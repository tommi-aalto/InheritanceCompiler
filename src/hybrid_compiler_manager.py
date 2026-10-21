"""
hybrid_compiler_manager.py -- build orchestration for a package of @compiled classes
==================================================================

LAYOUT
    myproject/
        __init__.py          <- the ONLY place you call the builder (see below)
        models/              <- your OOP source: any depth, any number of modules
            __init__.py
            base.py          (contains @compiled classes)
            tanks.py         (contains @compiled classes, may inherit from base.py's)
            helpers.py       (no @compiled -> never scanned, stays ordinary Python)
        compiled.py          <- GENERATED   (what you import at runtime)
        compiled.pyi         <- GENERATED   (IDE stub: re-exports your ORIGINAL classes)

    myproject/__init__.py:
        from hybrid_compiler_manager import HybridCompilerManager
        HybridCompilerManager(__file__, __name__, source_dir="models").build()

    your program:
        from myproject.compiled import Entity, CommandTank

WHAT build() DOES (runs when the package is first imported)
    1. scan   every .py under source_dir with the AST (no user code is imported) and keep
              the modules that contain at least one @compiled class
    2. hash   those files' contents + the compiler's own source -> compare with the hash
              stored in the first line of compiled.py
    3. if unchanged: do nothing (fast path, just reading + hashing a few files)
       if changed:   merge all modules into ONE compilation unit, compile it, delete the
                     stale Numba cache, and write compiled.py and compiled.pyi atomically.
       If compilation fails, the previous compiled.py is left untouched and the error
       (with file and line) is raised from `import myproject`.

THE MERGED NAMESPACE
    All @compiled modules are compiled together, because classes inherit across modules and share
    one type-id table.  Consequences:
      * `from .base import Entity` / `from myproject.models.base import Entity` between
        @compiled modules are dropped (the names live in the merged module).
      * `import myproject.models.base` (plain import of a @compiled module) is rejected.
      * Imports of NON-@compiled modules (helpers.py, numpy, ...) are kept; relative ones are
        rewritten to absolute because compiled.py lives in a different package level.
      * Top-level class/function names must be unique across the scanned modules.
"""
from __future__ import annotations

import ast
import hashlib
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hybrid_compiler
from hybrid_compiler import CompileError, HybridCompiler, _is_marked

HASH_PREFIX = "# hybrid-hash: "


@dataclass
class SourceModule:
    path: Path
    text: str
    dotted: str            # absolute dotted name, e.g. "myproject.models.base"
    rel_dotted: str        # relative to the package dir, e.g. "models.base" (for the stub)
    is_init: bool          # file is a package __init__.py
    tree: ast.Module
    compiled_classes: list[str]


class HybridCompilerManager:
    def __init__(self, init_file: str, package: Optional[str] = None, source_dir: str = "models",
                 output: str = "compiled", verbose: bool = True):
        """
        init_file   pass `__file__` from the package's __init__.py
        package     pass `__name__` (needed to rewrite relative imports to absolute ones)
        source_dir  subfolder holding the user's OOP source
        output      base name of the generated module (-> compiled.py / compiled.pyi)
        """
        self.pkg_dir = Path(init_file).resolve().parent
        self.package = package or self.pkg_dir.name
        self.src_dir = self.pkg_dir / source_dir
        self.output = output
        self.out_py = self.pkg_dir / f"{output}.py"
        self.out_pyi = self.pkg_dir / f"{output}.pyi"
        self.verbose = verbose
        if not self.src_dir.is_dir():
            raise CompileError(f"source directory not found: {self.src_dir}")

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def build(self, force: bool = False) -> bool:
        """Compile if anything changed.  Returns True if it (re)compiled."""
        mods = self.scan()
        if not mods:
            warnings.warn(f"hybrid: no @compiled classes found under {self.src_dir}; nothing compiled")
            return False
        digest = self.digest(mods)
        if not force and self.out_py.exists() and self.out_pyi.exists() and self.stored_digest() == digest:
            return False

        ordered = self._order(mods)
        code = self._compile(ordered)                 # raises CompileError -> old output stays intact
        stub = self._stub(ordered)

        removed = self._clean_numba_cache()           # source changed: drop stale JIT cache
        self._atomic_write(self.out_py, f"{HASH_PREFIX}{digest}\n{code}")
        self._atomic_write(self.out_pyi, stub)

        if f"{self.package}.{self.output}" in sys.modules:
            warnings.warn(f"hybrid: {self.output}.py was regenerated after it was imported in this process; "
                          f"restart Python to use the new version")
        if self.verbose:
            n = sum(len(m.compiled_classes) for m in mods)
            print(f"hybrid: compiled {n} classes from {len(mods)} modules -> {self.out_py.name} "
                  f"(cleared {removed} numba cache files)", file=sys.stderr)
        return True

    # ------------------------------------------------------------------
    # step 1: scan (AST only; user code is never imported or executed)
    # ------------------------------------------------------------------
    def scan(self) -> list[SourceModule]:
        mods = []
        for path in sorted(self.src_dir.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(text, filename=str(path))
            except SyntaxError as e:
                raise CompileError(f"{path}: {e}") from e
            classes = [n.name for n in tree.body if isinstance(n, ast.ClassDef) and _is_marked(n.decorator_list)]
            if not classes:
                continue
            for n in tree.body:
                if isinstance(n, ast.ClassDef):
                    n._hybrid_file = str(path)           # lets the compiler name the file in errors
            parts = list(path.relative_to(self.pkg_dir).with_suffix("").parts)
            is_init = parts[-1] == "__init__"
            if is_init:
                parts = parts[:-1]
            mods.append(SourceModule(path, text, ".".join([self.package] + parts), ".".join(parts),
                                     is_init, tree, classes))
        return mods

    # ------------------------------------------------------------------
    # step 2: change detection
    # ------------------------------------------------------------------
    def digest(self, mods: list[SourceModule]) -> str:
        """Hash of every @compiled source file (path + content) and of the compiler itself, so
        upgrading the compiler also triggers a rebuild."""
        h = hashlib.sha256()
        h.update(Path(hybrid_compiler.__file__).read_bytes())
        for m in sorted(mods, key=lambda m: m.rel_dotted):
            h.update(m.rel_dotted.encode() + b"\0" + m.text.encode() + b"\0")
        return h.hexdigest()

    def stored_digest(self) -> Optional[str]:
        try:
            with open(self.out_py, encoding="utf-8") as fh:
                first = fh.readline()
        except OSError:
            return None
        return first[len(HASH_PREFIX):].strip() if first.startswith(HASH_PREFIX) else None

    # ------------------------------------------------------------------
    # step 3: merge modules into one compilation unit
    # ------------------------------------------------------------------
    def _resolve(self, node: ast.ImportFrom, m: SourceModule) -> Optional[str]:
        """Absolute dotted name of the module an ImportFrom refers to."""
        if node.level == 0:
            return node.module
        base = m.dotted.split(".") if m.is_init else m.dotted.split(".")[:-1]
        up = node.level - 1
        if up > len(base):
            raise CompileError(f"{m.path}:{node.lineno}: relative import goes above the top-level package")
        base = base[:len(base) - up]
        return ".".join(base + ([node.module] if node.module else []))

    def _order(self, mods: list[SourceModule]) -> list[SourceModule]:
        """Dependencies first (cycles are tolerated: everything ends up in one namespace anyway)."""
        by = {m.dotted: m for m in mods}
        order, seen = [], set()

        def deps(m):
            for n in m.tree.body:
                if isinstance(n, ast.ImportFrom):
                    t = self._resolve(n, m)
                    if t in by:
                        yield by[t]
                    for a in n.names:
                        if f"{t}.{a.name}" in by:
                            yield by[f"{t}.{a.name}"]

        def visit(m):
            if m.dotted in seen:
                return
            seen.add(m.dotted)
            for d in deps(m):
                visit(d)
            order.append(m)

        for m in mods:
            visit(m)
        return order

    def _rewrite_import(self, node, m: SourceModule, compiled: set[str]):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in compiled:
                    raise CompileError(f"{m.path}:{node.lineno}: 'import {a.name}' of another @compiled module is not "
                                       f"supported; use 'from {a.name} import <names>'")
            return node
        if isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                return node
            target = self._resolve(node, m)
            if target in compiled:
                return None                            # names live in the merged module
            names = [a for a in node.names if f"{target}.{a.name}" not in compiled]
            if not names:
                return None                            # `from . import base` style
            new = ast.ImportFrom(module=target, names=names, level=0)
            return ast.copy_location(new, node)
        return node

    def _compile(self, ordered: list[SourceModule]) -> str:
        compiled = {m.dotted for m in ordered}
        body, owner = [], {}
        for m in ordered:
            for node in m.tree.body:
                node = self._rewrite_import(node, m, compiled)
                if node is None:
                    continue
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    if node.name in owner:
                        raise CompileError(f"'{node.name}' is defined in both {owner[node.name]} and {m.path}; "
                                           f"all @compiled modules share one namespace")
                    owner[node.name] = m.path
                body.append(node)
        return HybridCompiler().compile_tree(ast.Module(body=body, type_ignores=[]))

    # ------------------------------------------------------------------
    # step 4: IDE stub -- points the generated module's names at the ORIGINAL classes
    # ------------------------------------------------------------------
    @staticmethod
    def _top_level_names(tree: ast.Module) -> list[str]:
        names = []
        for n in tree.body:
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)):
                names.append(n.name)
            elif isinstance(n, ast.Assign):
                names += [t.id for t in n.targets if isinstance(t, ast.Name)]
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                names.append(n.target.id)
        return [x for x in names if not (x.startswith("__") and x.endswith("__"))]

    def _stub(self, ordered: list[SourceModule]) -> str:
        lines = ["# AUTO-GENERATED by hybrid_compiler_manager.py -- IDE stub, do not edit.",
                 f"# Editors and type checkers read this instead of {self.output}.py, so they see your",
                 "# ORIGINAL classes (real signatures, docstrings, field types).  At runtime the",
                 f"# compiled implementations in {self.output}.py are what you actually get.", ""]
        for m in ordered:
            names = self._top_level_names(m.tree)
            if names and m.rel_dotted:
                inner = ",\n".join(f"    {n} as {n}" for n in names)
                lines.append(f"from .{m.rel_dotted} import (\n{inner},\n)")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # step 5: housekeeping
    # ------------------------------------------------------------------
    def _clean_numba_cache(self) -> int:
        """Numba's on-disk cache (*.nbi index, *.nbc data) does not notice changes to
        generated globals or to callees, so any rebuild drops it.  Files are named
        '<module>.<function>-<line>.<pyver>.nbi|nbc' and live in __pycache__ next to the
        module.  With NUMBA_CACHE_DIR set the layout differs (assumed:
        <dir>/<package-dir-name>_<hash>/...); I have not verified that on every Numba version."""
        removed = 0
        patterns = [f"{self.output}.*.nbi", f"{self.output}.*.nbc", f"{self.output}.*.pyc"]
        roots = [(self.pkg_dir / "__pycache__", "")]
        env = os.environ.get("NUMBA_CACHE_DIR")
        if env:
            roots.append((Path(env), f"{self.pkg_dir.name}_*/"))
        for root, sub in roots:
            if not root.is_dir():
                continue
            for pat in patterns:
                for f in root.glob(sub + pat):
                    try:
                        f.unlink()
                        removed += 1
                    except OSError:
                        pass
        return removed

    @staticmethod
    def _atomic_write(path: Path, text: str):
        """Write via a temp file + rename so a crash or a concurrent import never sees half a file."""
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
