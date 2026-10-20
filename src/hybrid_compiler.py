"""
hybrid_compiler.py -- "Python OOP -> flat Numba code + numpy-backed proxies"
============================================================================

WHAT THE USER WRITES
    Ordinary Python OOP (multiple inheritance, super(), polymorphism, numpy arrays).
        from hybrid_compiler import c       # marker decorator, does nothing at runtime

        @c                                   # <- class is compiled
        class Entity:
            health: float                    # typed fields become numpy-dtype columns
            @c                               # <- method is JIT-compiled (numba)
            def take_damage(self, amount: float): ...
            def describe(self): ...          # undecorated -> stays plain Python
    Undecorated classes are copied through untouched.

WHAT THE COMPILER PRODUCES (one .py file)
    * one numpy dtype per compiled class ("row" = one-element array of that dtype)
    * "projected" dtypes: the fields of ANCESTOR A, placed at the offsets they have
      inside DESCENDANT D.  A parent-typed pointer to a child row can then be viewed
      as a parent row, so parent methods / parent field accesses work on any child.
      (Projections that are byte-identical are de-duplicated -> one Numba
      specialisation per distinct layout, not per class.)
    * @njit functions  Class__method(self, self_addr, ...)
          `self` is a numpy RECORD (rows[0]), so inside a method the user's own
          `self.health += 1` is plain record attribute access: NO rewriting.
    * dispatchers      dispatch_K_m(type, addr, ...)  virtual call on a K-typed pointer
    * super routers    super_from_X_m(type, addr, ...) follows the RUNTIME object's MRO
    * Python proxy classes that OWN their row (and every buffer) and expose the
      fields as properties.  The Python garbage collector therefore keeps every
      pointer target alive (proxies keep strong refs to what they point to).

ONE MECHANISM FOR EVERYTHING THAT IS NOT `self`
    Numba cannot hold two different record types in one variable, so any access through a
    (type_id, addr) pointer must branch on the type id and use the matching view.  Instead
    of a second code generator for fields, field access reuses the method machinery:
        other.health        ->  dispatch_Entity__hy_get_health(t, a)
        other.health = v    ->  dispatch_Entity__hy_set_health(t, a, v)
        other.arr           ->  dispatch_Entity__hy_arr_arr(t, a)
        other.CONST         ->  dispatch_Entity__hy_const_CONST(t, a)      (only if overridden)
    where  Entity___hy_get_health(self, self_addr)  is a generated method whose whole body
    is  `return self.health`.  So a pointer access is "a method call through the dispatcher",
    and what runs inside is the same plain `self.x` attribute access as everywhere else.

POINTERS ("fat pointer" = two int64)
    (type_id, addr).  addr is the address of the target's row; 0 means None.
    A list of objects is an int64 array field of shape (cap, 2) holding (type_id, addr)
    pairs plus a count field -- i.e. it is read exactly like any numpy array field.
    Inside JIT code an address becomes a record again with
        carray(_to_voidptr(addr), 1, some_projected_dtype)[0]
    (`_to_voidptr` is a 1-instruction LLVM intrinsic: inttoptr).

DEBUG MODE
    Set the environment variable HYBRID_DEBUG=1 before importing the generated module and
    it runs WITHOUT Numba: every @njit function is an ordinary Python function (pdb, print,
    breakpoints), `carray` is ctypes over the address, and records are a thin attribute
    wrapper over numpy void scalars.  Same code, same semantics -- it is also how the tests
    run here.  It verifies the compiler's logic, NOT that Numba accepts the generated code.

RULES THE COMPILER ENFORCES
    1. JIT writes DATA, Python writes POINTERS.  JIT code may not re-point refs,
       assign lists / arrays, append to lists, or construct compiled objects.
    2. A @c method may only call other @c methods (checked line by line).
    3. A @c method may only touch typed fields (float/int/bool, compiled-class refs,
       list[Class], np.ndarray[dtype, ndim]).  Strings/dicts etc. are Python-only
       "metadata" fields; touching one from JIT code is a compile error.
    4. An attribute name has ONE type across a class's MRO (Python has one slot per
       name).  Conflicts are a compile error.
    5. Compiled classes may only inherit from compiled classes; undecorated
       subclasses of compiled classes are rejected.
    6. An override preserves its parent method's parameter names, types, and order; it may
       append parameters only when each appended parameter has a default.

KNOWN LIMITS / UNTESTED
    * Methods that return objects are not supported (return numbers/bools/None).
    * Mutual recursion between @c methods may need explicit Numba signatures.
    * Numba-specific pieces (carray over a voidptr with a record dtype, records with
      explicit offsets, passing a record from carray(...)[0] to another njit function and
      mutating it) are written from documentation: run numba_smoke_test.py on a machine
      with Numba to verify them on your version.
"""
from __future__ import annotations

import ast
import copy
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Optional, TypeVar

import numpy as np


class CompileError(Exception):
    """Raised for any user error detected while transpiling."""


_T = TypeVar("_T")


def compiled(obj: _T) -> _T:
    """Marker decorator.  Does nothing at runtime; the compiler finds it in the AST.
    Typed as identity so IDEs/type checkers keep seeing the original class or method."""
    return obj


# =====================================================================
# 1. METADATA STRUCTURES
# =====================================================================
# Python annotation -> numpy dtype name
PRIMS = {"float": "float64", "int": "int64", "bool": "bool_"}
# size == alignment of every column type we ever emit
SIZE = {"float64": 8, "int64": 8, "bool_": 1, "float32": 4, "int32": 4,
        "int16": 2, "int8": 1, "uint8": 1, "uint16": 2, "uint32": 4, "uint64": 8}


def _round_up(x: int, to: int) -> int:
    return (x + to - 1) // to * to


@dataclass(frozen=True)
class Field:
    """One typed attribute of a compiled class.

    kind:
      'prim'    float/int/bool           -> 1 column
      'ref'     another compiled class   -> 2 columns  (name_type, name_addr)
      'objlist' list[Class(, cap)]       -> 3 columns  (name_data, name_shape[2], name_count)
                                           an int64 array of shape (cap, 2) holding
                                           (type_id, addr) pairs: it IS an array field
      'array'   np.ndarray[dtype, ndim]  -> 2 columns  (name_data, name_shape[ndim])
    """
    name: str
    kind: str
    dtype: str = ""        # prim / array element dtype
    target: str = ""       # ref / objlist element class
    cap: int = 0           # objlist initial capacity (Python may grow it later)
    ndim: int = 0          # array rank

    def sig(self) -> tuple:
        """Identity used to detect conflicting redeclarations (cap is only a default)."""
        return (self.name, self.kind, self.dtype, self.target, self.ndim)

    def columns(self) -> list[tuple[str, str, int]]:
        """(column name, numpy dtype name, sub-array length or 0 for a scalar)."""
        n = self.name
        if self.kind == "prim":
            return [(n, self.dtype, 0)]
        if self.kind == "ref":
            return [(f"{n}_type", "int64", 0), (f"{n}_addr", "int64", 0)]
        if self.kind == "objlist":
            return [(f"{n}_data", "int64", 0), (f"{n}_shape", "int64", 2), (f"{n}_count", "int64", 0)]
        return [(f"{n}_data", "int64", 0), (f"{n}_shape", "int64", self.ndim)]

    def array_helper(self) -> str:
        """Name of the generated `(addr, shape) -> numpy view` function for array-like fields."""
        dt, nd = (self.dtype, self.ndim) if self.kind == "array" else ("int64", 2)
        return f"_arr_{dt}_{nd}d"


@dataclass
class Param:
    """One parameter of a @c method (after `self`)."""
    name: str
    kind: str                  # 'prim' | 'ent' | 'pass'  (pass = e.g. np.ndarray, handed through)
    ann: str                   # annotation text
    ent: str = ""              # class name when kind == 'ent'
    default: Optional[ast.expr] = None

    def flat(self) -> list[str]:
        """Names of the JIT parameters this expands to (objects become 2 ints)."""
        return [f"{self.name}_type", f"{self.name}_addr"] if self.kind == "ent" else [self.name]


@dataclass
class ClassInfo:
    name: str
    node: ast.ClassDef
    tid: int
    bases: list[str]
    decls: dict[str, Field] = field(default_factory=dict)          # fields declared HERE
    py_fields: set[str] = field(default_factory=set)               # python-only annotated names
    consts: dict[str, tuple[str, str]] = field(default_factory=dict)  # attr -> (global name, source)
    jit_methods: dict[str, ast.FunctionDef] = field(default_factory=dict)
    synth: dict[str, ast.FunctionDef] = field(default_factory=dict)   # generated accessor methods
    sigs: dict[str, list[Param]] = field(default_factory=dict)
    py_nodes: list[ast.stmt] = field(default_factory=list)         # copied verbatim into the proxy
    file: str = "<source>"                                         # for error messages


@dataclass
class Layout:
    cols: list[tuple[str, str, int, int]]   # (column, dtype, subarray len, byte offset)
    itemsize: int
    order: list[Field]                      # fields in layout order (first base's order is a prefix)


@dataclass
class Ptr:
    """Compile-time description of an object-valued expression: its static class
    plus SOURCE TEXT for its runtime type id and address."""
    cls: str
    t: str
    a: str
    is_self: bool = False


def _is_marked(decorators: list[ast.expr]) -> bool:
    """True if the decorator list contains `c` or `c()`."""
    for d in decorators:
        if isinstance(d, ast.Name) and d.id == "compiled":
            return True
        if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "compiled":
            return True
    return False


# =====================================================================
# 2. THE COMPILER
# =====================================================================
class HybridCompiler:
    DEFAULT_LIST_CAPACITY = 16

    def __init__(self):
        self._reset()

    def _reset(self):
        self.classes: dict[str, ClassInfo] = {}
        self.imports: list[ast.stmt] = []
        self.other_nodes: list[ast.stmt] = []
        self.mro: dict[str, list[str]] = {}
        self.descendants: dict[str, list[str]] = {}
        self.fields: dict[str, dict[str, Field]] = {}       # MRO-resolved fields per class
        self.layouts: dict[str, Layout] = {}
        self.proj_of: dict[tuple[str, str], str] = {}       # (descendant, ancestor) -> dtype var
        self.proj_specs: dict[str, tuple[list, int]] = {}   # dtype var -> (cols, itemsize)
        self._proj_by_key: dict[tuple, str] = {}
        # (static class, method name) pairs the lowering pass needs a dispatcher for
        # (user methods AND generated accessor methods share this one mechanism)
        self.need_dispatch: set[tuple[str, str]] = set()
        self.need_super: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------
    # driver
    # ------------------------------------------------------------------
    def compile(self, source: str) -> str:
        return self.compile_tree(ast.parse(source))

    def compile_tree(self, tree: ast.Module) -> str:
        """Compile an already-parsed (possibly merged multi-module) tree.  ClassDef nodes may
        carry an `_hybrid_file` attribute, used to attribute errors to their source file."""
        self._reset()
        self._parse(tree)
        self._build_mro()
        self._resolve_fields()
        self._check_signatures()
        self._build_layouts()
        self._build_projections()
        return self._emit()

    # ------------------------------------------------------------------
    # step 1: parse, classify, validate
    # ------------------------------------------------------------------
    def _parse(self, tree: ast.Module):
        compiled = [n for n in tree.body if isinstance(n, ast.ClassDef) and _is_marked(n.decorator_list)]
        self.class_names = {n.name for n in compiled}
        upper = [n.name.upper() for n in compiled]
        if len(set(upper)) != len(upper):
            raise CompileError("Class names must differ by more than case (TYPE_<NAME> constants would clash)")

        tid = 1
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in self.class_names:
                self._parse_class(node, tid)
                tid += 1
            elif isinstance(node, ast.ClassDef):
                # undecorated class: must not inherit from a compiled class (rule 5)
                for b in node.bases:
                    if isinstance(b, ast.Name) and b.id in self.class_names:
                        raise CompileError(
                            f"Undecorated class '{node.name}' inherits from compiled class '{b.id}'. "
                            f"It would share the parent's TYPE_ID and JIT code would silently ignore its "
                            f"overrides. Decorate it with @c.")
                self.other_nodes.append(node)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                self._keep_import(node)
            else:
                self.other_nodes.append(node)

    def _keep_import(self, node):
        """Keep user imports, except the marker `c` and __future__ (we emit our own)."""
        if isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                return
            names = [a for a in node.names if (a.asname or a.name) != "c"]
            if not names:
                return
            node = ast.ImportFrom(module=node.module, names=names, level=node.level)
        self.imports.append(node)

    @staticmethod
    def _ann_name(ann: Optional[ast.AST]) -> str:
        """Annotation -> bare name ('float', 'Entity', 'ndarray'); '' if not a simple name."""
        if isinstance(ann, ast.Name):
            return ann.id
        if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
            return ann.value
        if isinstance(ann, ast.Attribute):
            return ann.attr
        return ""

    @staticmethod
    def _np_dtype_name(node: ast.AST) -> str:
        name = HybridCompiler._ann_name(node)
        name = {"float": "float64", "int": "int64", "bool": "bool_"}.get(name, name)
        try:
            np.dtype(name.rstrip("_") if name == "bool_" else name)
        except TypeError:
            raise CompileError(f"Unsupported numpy dtype '{name}'")
        if name not in SIZE:
            raise CompileError(f"Unsupported numpy dtype '{name}'")
        return name

    def _field_from_annotation(self, name: str, ann: ast.AST) -> Optional[Field]:
        """Return a Field, or None when the annotation means 'python-only metadata'."""
        s = self._ann_name(ann)
        if isinstance(ann, (ast.Name, ast.Constant)):
            if s in PRIMS:
                return Field(name, "prim", dtype=PRIMS[s])
            if s in self.class_names:
                return Field(name, "ref", target=s)
            return None
        if isinstance(ann, ast.Subscript):
            base = self._ann_name(ann.value)
            sl = ann.slice
            if base == "list":
                if isinstance(sl, ast.Tuple) and len(sl.elts) == 2 and isinstance(sl.elts[1], ast.Constant):
                    elem, cap = self._ann_name(sl.elts[0]), int(sl.elts[1].value)
                else:
                    elem, cap = self._ann_name(sl), self.DEFAULT_LIST_CAPACITY
                if elem in self.class_names:
                    return Field(name, "objlist", target=elem, cap=cap)
                if elem in PRIMS:
                    raise CompileError(
                        f"Field '{name}': list[{elem}] cannot be read by JIT code. Use np.ndarray[{elem}, 1] "
                        f"(add an `int` count field for a growable buffer), or annotate it as plain `list` for Python-only use.")
                return None                                  # plain python list -> metadata
            if base == "ndarray":
                if not (isinstance(sl, ast.Tuple) and len(sl.elts) == 2 and isinstance(sl.elts[1], ast.Constant)):
                    raise CompileError(f"Field '{name}': write np.ndarray[dtype, ndim], e.g. np.ndarray[float, 2]")
                return Field(name, "array", dtype=self._np_dtype_name(sl.elts[0]), ndim=int(sl.elts[1].value))
        return None

    def _make_sig(self, cls: str, fn: ast.FunctionDef) -> list[Param]:
        a = fn.args
        if a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
            raise CompileError(f"{cls}.{fn.name}: *args/**kwargs/keyword-only/positional-only params are not supported in @c methods")
        if not a.args or a.args[0].arg != "self":
            raise CompileError(f"{cls}.{fn.name}: first parameter of a @c method must be named 'self'")
        defaults = [None] * (len(a.args) - len(a.defaults)) + list(a.defaults)
        params = []
        for arg, dflt in zip(a.args[1:], defaults[1:]):
            if arg.annotation is None:
                raise CompileError(f"{cls}.{fn.name}: parameter '{arg.arg}' needs a type annotation")
            s = self._ann_name(arg.annotation)
            if s in PRIMS:
                params.append(Param(arg.arg, "prim", s, default=dflt))
            elif s in self.class_names:
                params.append(Param(arg.arg, "ent", s, ent=s, default=dflt))
            else:
                params.append(Param(arg.arg, "pass", ast.unparse(arg.annotation), default=dflt))
        if fn.returns is not None and self._ann_name(fn.returns) in self.class_names:
            raise CompileError(f"{cls}.{fn.name}: returning compiled objects from @c methods is not supported")
        return params

    def _parse_class(self, node: ast.ClassDef, tid: int):
        if node.keywords:
            raise CompileError(f"{node.name}: class keywords/metaclasses are not supported")
        for d in node.decorator_list:
            if not _is_marked([d]):
                raise CompileError(f"{node.name}: only the @c class decorator is supported")
        bases = []
        for b in node.bases:
            if isinstance(b, ast.Name) and b.id == "object":
                continue
            if isinstance(b, ast.Name) and b.id in self.class_names:
                bases.append(b.id)
            else:
                raise CompileError(f"{node.name}: compiled classes may only inherit from other @c classes, got {ast.unparse(b)}")
        info = ClassInfo(node.name, node, tid, bases)
        info.file = getattr(node, "_hybrid_file", "<source>")

        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                attr = item.target.id
                if item.value is not None:
                    ann = self._ann_name(item.annotation)
                    if ann in PRIMS or isinstance(item.value, ast.Constant):
                        # class constant: hoisted to a module global so Numba can freeze it
                        info.consts[attr] = (f"CONST_{node.name}_{attr}", ast.unparse(item.value))
                    else:
                        info.py_nodes.append(item)
                    continue
                f = self._field_from_annotation(attr, item.annotation)
                if f is None:
                    info.py_fields.add(attr)           # metadata: plain python attribute
                else:
                    info.decls[attr] = f
            elif (isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name)
                  and isinstance(item.value, ast.Constant) and isinstance(item.value.value, (int, float, bool))):
                attr = item.targets[0].id            # unannotated numeric constant -> also hoisted
                info.consts[attr] = (f"CONST_{node.name}_{attr}", ast.unparse(item.value))
            elif isinstance(item, ast.FunctionDef) and _is_marked(item.decorator_list):
                if item.name.startswith("__"):
                    raise CompileError(f"{node.name}.{item.name}: dunder methods cannot be @c (keep __init__ plain Python)")
                if len(item.decorator_list) != 1:
                    raise CompileError(f"{node.name}.{item.name}: a @c method cannot carry other decorators")
                info.jit_methods[item.name] = item
                info.sigs[item.name] = self._make_sig(node.name, item)
            else:
                info.py_nodes.append(item)            # plain python method / docstring / etc.
        self.classes[node.name] = info

    # ------------------------------------------------------------------
    # step 2: C3 MRO + descendants
    # ------------------------------------------------------------------
    def _c3(self, name: str) -> list[str]:
        bases = self.classes[name].bases
        if not bases:
            return [name]
        seqs = [list(self._c3(b)) for b in bases] + [list(bases)]
        res = [name]
        while True:
            seqs = [s for s in seqs if s]
            if not seqs:
                return res
            for s in seqs:
                cand = s[0]
                if not any(cand in o[1:] for o in seqs):
                    break
            else:
                raise CompileError(f"Inconsistent method resolution order for class '{name}'")
            res.append(cand)
            for s in seqs:
                if s[0] == cand:
                    s.pop(0)

    def _build_mro(self):
        for n in self.classes:
            self.mro[n] = self._c3(n)
        for k in self.classes:
            self.descendants[k] = [d for d in self.classes if k in self.mro[d]]  # id order, includes k

    # ------------------------------------------------------------------
    # step 3: MRO-resolved fields (conflict detection) + helpers
    # ------------------------------------------------------------------
    def _resolve_fields(self):
        for cls, mro in self.mro.items():
            resolved: dict[str, tuple[Field, str]] = {}
            for anc in mro:
                for name, f in self.classes[anc].decls.items():
                    if name in resolved and resolved[name][0].sig() != f.sig():
                        other = resolved[name]
                        raise CompileError(
                            f"Attribute '{name}' has conflicting types in the MRO of {cls}: "
                            f"{other[1]} declares {other[0].sig()[1:]}, {anc} declares {f.sig()[1:]}. "
                            f"Python has a single slot per attribute name; rename one of them.")
                    resolved.setdefault(name, (f, anc))
            self.fields[cls] = {n: f for n, (f, _) in resolved.items()}

    def const_for(self, cls: str, attr: str) -> Optional[str]:
        for anc in self.mro[cls]:
            if attr in self.classes[anc].consts:
                return self.classes[anc].consts[attr][0]
        return None

    def const_owner(self, cls: str, attr: str) -> str:
        return next(a for a in self.mro[cls] if attr in self.classes[a].consts)

    def decl_owner(self, cls: str, fname: str) -> str:
        """The class (in cls's MRO) that declares field `fname`."""
        return next(a for a in self.mro[cls] if fname in self.classes[a].decls)

    def find_method(self, cls: str, m: str) -> Optional[str]:
        """First class in cls's MRO that defines method m (what Python would call).
        Generated accessor methods count: they are resolved exactly like user methods."""
        return next((a for a in self.mro[cls]
                     if m in self.classes[a].jit_methods or m in self.classes[a].synth), None)

    def sig_of(self, cls: str, m: str) -> list[Param]:
        return self.classes[self.find_method(cls, m)].sigs[m]

    @staticmethod
    def flat_params(sig: list[Param]) -> list[str]:
        return [n for p in sig for n in p.flat()]

    # ---- generated accessor methods (field access through a pointer) ----------
    def _synth(self, owner: str, name: str, params: list[Param], body: str) -> str:
        """Register a tiny generated @njit method `Owner__name(self, self_addr, ...)` whose
        body is plain record attribute access.  Returns its method name so callers can ask for
        a dispatcher.  Because it is an ordinary method, ordinary dispatch picks the right
        view for every runtime class."""
        info = self.classes[owner]
        if name not in info.synth:
            flat = ["self", "self_addr"] + self.flat_params(params)
            src = f"@njit(cache=True)\ndef {owner}__{name}({', '.join(flat)}):\n    {body}\n"
            info.synth[name] = ast.parse(src).body[0]
            info.sigs[name] = params
        return name

    def getter(self, cls: str, col: str, fname: str) -> str:
        return self._synth(self.decl_owner(cls, fname), f"_hy_get_{col}", [], f"return self.{col}")

    def setter(self, cls: str, col: str, fname: str) -> str:
        return self._synth(self.decl_owner(cls, fname), f"_hy_set_{col}",
                           [Param("val", "pass", "")], f"self.{col} = val")

    def array_getter(self, cls: str, f: Field) -> str:
        return self._synth(self.decl_owner(cls, f.name), f"_hy_arr_{f.name}", [],
                           f"return {f.array_helper()}(self.{f.name}_data, self.{f.name}_shape)")

    def const_getter(self, cls: str, attr: str) -> str:
        """Class constant that some subclass overrides: Python's lookup depends on the RUNTIME
        class, so give every defining class a tiny method returning its own value."""
        for d in self.descendants[cls]:
            owner = self.const_owner(d, attr)
            self._synth(owner, f"_hy_const_{attr}", [], f"return {self.classes[owner].consts[attr][0]}")
        return f"_hy_const_{attr}"

    # ------------------------------------------------------------------
    # step 4: override signatures must preserve the parent-call prefix
    # ------------------------------------------------------------------
    def _check_signatures(self):
        key = lambda sig: [(p.name, p.kind, p.ann, p.ent) for p in sig]
        for cls, info in self.classes.items():
            for m, sig in info.sigs.items():
                for anc in self.mro[cls][1:]:
                    other = self.classes[anc].sigs.get(m)
                    if other is None:
                        continue
                    if len(sig) < len(other) or key(sig[:len(other)]) != key(other):
                        raise CompileError(f"{cls}.{m} overrides {anc}.{m} with a different signature "
                                           f"(parent parameter names/types/order must match so dispatch can forward arguments)")
                    if any(p.default is None for p in sig[len(other):]):
                        raise CompileError(f"{cls}.{m} extends {anc}.{m} with parameters that must have default values")

    # ------------------------------------------------------------------
    # step 5: memory layout
    # ------------------------------------------------------------------
    def _layout(self, cls: str):
        """Layout of one class.  The FIRST base's field order is a prefix of ours, so every
        field inherited through first bases sits at the same offset in all descendants
        (prefix layout).  Other bases' fields are appended and handled by projections."""
        if cls in self.layouts:
            return
        info = self.classes[cls]
        for b in info.bases:
            self._layout(b)
        order: list[Field] = []
        seen: set[str] = set()

        def add(f: Field):
            if f.name not in seen:
                seen.add(f.name)
                order.append(f)

        for b in info.bases:                         # first base first -> prefix property
            for f in self.layouts[b].order:
                add(f)
        for f in info.decls.values():
            add(f)

        cols = [("type_id", "int64", 0, 0)]          # type_id is ALWAYS at offset 0
        names = {"type_id"}
        off = 8
        for f in order:
            for col, dt, cnt in f.columns():
                if col in names:
                    raise CompileError(f"{cls}: column '{col}' (from field '{f.name}') collides with another "
                                       f"field's column; rename one of the fields")
                names.add(col)
                off = _round_up(off, SIZE[dt])
                cols.append((col, dt, cnt, off))
                off += SIZE[dt] * (cnt or 1)
        self.layouts[cls] = Layout(cols, _round_up(off, 8), order)

    def _build_layouts(self):
        for cls in self.classes:
            self._layout(cls)

    # ------------------------------------------------------------------
    # step 6: projected dtypes  (ancestor's fields at the descendant's offsets)
    # ------------------------------------------------------------------
    def _build_projections(self):
        pairs = [(a, a) for a in self.classes]       # own dtypes first (they get the nice names)
        pairs += [(d, a) for d in self.classes for a in self.mro[d] if a != d]
        for d, a in pairs:
            d_off = {col: off for col, _, _, off in self.layouts[d].cols}
            cols = [(col, dt, cnt, d_off[col]) for col, dt, cnt, _ in self.layouts[a].cols]
            end = max(off + SIZE[dt] * (cnt or 1) for _, dt, cnt, off in cols)
            itemsize = _round_up(end, 8)
            key = (tuple(cols), itemsize)
            if key not in self._proj_by_key:
                var = f"{a.lower()}_dtype" if d == a else f"{a.lower()}_view{len(self.proj_specs)}"
                self._proj_by_key[key] = var
                self.proj_specs[var] = (cols, itemsize)
            self.proj_of[(d, a)] = self._proj_by_key[key]

    # ------------------------------------------------------------------
    # step 7: emit the module
    # ------------------------------------------------------------------
    def _emit(self) -> str:
        # --- lower every @c method first; this discovers which accessors/dispatchers must exist ---
        jit_src: list[str] = []
        for cls, info in self.classes.items():
            for m, fn in info.jit_methods.items():
                jit_src.append(f"# {info.file}:{fn.lineno}  {cls}.{m}")
                jit_src.append(ast.unparse(Lowerer(self, cls, m).lower(fn)))
                jit_src.append("")
        synth_src: list[str] = []
        for cls, info in self.classes.items():
            for name, fn in info.synth.items():
                synth_src.append(ast.unparse(fn))
                synth_src.append("")

        out: list[str] = [_HEADER]
        for imp in self.imports:
            out.append(ast.unparse(imp))
        out.append("")

        out.append("# --- Type ids and hoisted class constants " + "-" * 30)
        for n, info in self.classes.items():
            out.append(f"TYPE_{n.upper()} = {info.tid}")
        for info in self.classes.values():
            for _, (g, src) in info.consts.items():
                out.append(f"{g} = {src}")

        out.append("\n# --- Dtypes: one per class + projections onto each ancestor " + "-" * 10)
        for var, (cols, itemsize) in self.proj_specs.items():
            out.extend(self._dtype_src(var, cols, itemsize))

        n = len(self.classes) + 1
        table = np.zeros((n, n), dtype=bool)
        for d, mro in self.mro.items():
            for a in mro:
                table[self.classes[d].tid, self.classes[a].tid] = True
        out.append("\n# --- isinstance / issubclass: O(1) ancestry matrix " + "-" * 20)
        out.append(f"ancestry_table = np.array({table.astype(int).tolist()}, dtype=np.bool_)")
        out.append(_ANCESTRY_FUNCS)

        out.append(self._array_helpers())
        out.append("# --- @c methods lowered to flat numba functions (self is a numpy record) " + "-" * 8)
        out.extend(jit_src)
        out.append("# --- generated accessor methods: field access through a pointer = a method call " + "-" * 3)
        out.extend(synth_src)

        out.append("# --- Generated routers (virtual dispatch and super()) " + "-" * 15)
        for k, m in sorted(self.need_super):
            out.append(self._gen_super(k, m))
        for k, m in sorted(self.need_dispatch):
            out.append(self._gen_dispatch(k, m))

        out.append(_RUNTIME)
        for node in self.other_nodes:
            out.append(ast.unparse(node))
            out.append("")
        out.append("# --- Proxy classes (own their row; plain python outside the hot path) " + "-" * 5)
        for cls in sorted(self.classes, key=lambda x: (len(self.mro[x]), self.classes[x].tid)):
            out.append(self._gen_proxy(cls))
        return "\n".join(out) + "\n"

    @staticmethod
    def _dtype_src(var, cols, itemsize) -> list[str]:
        fmts = [f"np.{dt}" if cnt == 0 else f"(np.{dt}, ({cnt},))" for _, dt, cnt, _ in cols]
        return [f"{var} = np.dtype({{",
                f"    'names': {[c[0] for c in cols]!r},",
                f"    'formats': [{', '.join(fmts)}],",
                f"    'offsets': {[c[3] for c in cols]!r},",
                f"    'itemsize': {itemsize},",
                "}, align=True)"]

    def _array_helpers(self) -> str:
        """One `rebuild numpy view from (addr, shape)` helper per (dtype, ndim) in use.
        Object lists are (cap, 2) int64 arrays, so they use the same helpers."""
        combos = set()
        for i in self.classes.values():
            for f in i.decls.values():
                if f.kind in ("array", "objlist"):
                    combos.add(("int64", 2) if f.kind == "objlist" else (f.dtype, f.ndim))
        out = ["# --- numpy-array field helpers: (data address, shape) -> array view " + "-" * 8]
        for dt, nd in sorted(combos):
            shape = ", ".join(f"shape[{i}]" for i in range(nd)) + ("," if nd == 1 else "")
            out += ["@njit(cache=True)", f"def _arr_{dt}_{nd}d(addr, shape):",
                    "    if addr == 0:", "        raise ValueError('numpy array field is not assigned')",
                    f"    return carray(_to_voidptr(addr), ({shape}), np.{dt})", ""]
        return "\n".join(out)

    # ---- router generators --------------------------------------------------
    def _chain(self, groups: dict, body_for, final: str) -> list[str]:
        """Emit `if self_type == A or self_type == B: <body>` for each group.
        A single group needs no switch at all (every possible runtime type behaves alike)."""
        lines = []
        if len(groups) == 1:
            (key, _), = groups.items()
            return ["    " + l for l in body_for(key)]
        for key, ds in groups.items():
            cond = " or ".join(f"self_type == TYPE_{d.upper()}" for d in ds)
            lines.append(f"    if {cond}:")
            lines += ["        " + l for l in body_for(key)]
        lines.append(f"    raise ValueError({final!r})")
        return lines

    def _gen_dispatch(self, k: str, m: str) -> str:
        """Virtual call on a pointer whose STATIC class is k: pick the implementation
        Python's MRO would pick for the runtime class, and hand it the row viewed as that
        implementation's class (a record: carray(...)[0]). Child-only parameters are supplied
        with their declared defaults so calls through a parent-typed pointer remain valid."""
        sig = self.sig_of(k, m)
        params = self.flat_params(sig)
        groups: dict[tuple[str, str], list[str]] = {}
        for d in self.descendants[k]:
            impl = self.find_method(d, m)
            if impl:
                groups.setdefault((impl, self.proj_of[(d, impl)]), []).append(d)
        args = ", ".join(params)

        def body(key):
            call_args = self._forwarded_method_args(sig, self.classes[key[0]].sigs[m])
            return [f"return {key[0]}__{m}(carray(p, 1, {key[1]})[0], self_addr"
                    f"{', ' + ', '.join(call_args) if call_args else ''})"]

        lines = ["@njit(cache=True)", f"def dispatch_{k}_{m}(self_type, self_addr{', ' + args if args else ''}):",
                 f"    if self_addr == 0:", f"        raise ValueError('null reference calling {k}.{m}')",
                 "    p = _to_voidptr(self_addr)"]
        lines += self._chain(groups, body, f"bad type id in dispatch_{k}_{m}")
        return "\n".join(lines) + "\n"

    def _gen_super(self, x: str, m: str) -> str:
        """super().m() written inside class x.  super() follows the RUNTIME object's MRO,
        so for every possible runtime class D (x or a descendant) find the next
        implementation after x in MRO(D)."""
        params = self.flat_params(self._super_sig(x, m))
        groups: dict[tuple[str, str], list[str]] = {}
        for d in self.descendants[x]:
            mro = self.mro[d]
            nxt = next((a for a in mro[mro.index(x) + 1:] if m in self.classes[a].jit_methods), None)
            if nxt:
                groups.setdefault((nxt, self.proj_of[(d, nxt)]), []).append(d)
        sig = self._super_sig(x, m)
        args = ", ".join(params)

        def body(key):
            call_args = self._forwarded_method_args(sig, self.classes[key[0]].sigs[m])
            return [f"return {key[0]}__{m}(carray(p, 1, {key[1]})[0], self_addr"
                    f"{', ' + ', '.join(call_args) if call_args else ''})"]

        lines = ["@njit(cache=True)", f"def super_from_{x}_{m}(self_type, self_addr{', ' + args if args else ''}):",
                 "    p = _to_voidptr(self_addr)"]
        lines += self._chain(groups, body, f"super().{m}(): no implementation after {x} in the MRO")
        return "\n".join(lines) + "\n"

    def _forwarded_method_args(self, caller_sig: list[Param], implementation_sig: list[Param]) -> list[str]:
        """Flatten caller parameters and append an implementation's defaulted suffix."""
        args = self.flat_params(caller_sig)
        for param in implementation_sig[len(caller_sig):]:
            if param.default is None:
                raise CompileError("overridden method has an appended parameter without a default value")
            if param.kind == "ent":
                if not (isinstance(param.default, ast.Constant) and param.default.value is None):
                    raise CompileError(
                        f"object parameter '{param.name}' added by an override must default to None")
                args.extend(("0", "0"))
            else:
                args.append(ast.unparse(param.default))
        return args

    def _super_sig(self, x: str, m: str) -> list[Param]:
        for a in self.mro[x][1:]:
            if m in self.classes[a].jit_methods:
                return self.classes[a].sigs[m]
        raise CompileError(f"super().{m}() in {x}: no @c method '{m}' in any base class")

    # ---- proxy classes --------------------------------------------------------
    def _prop_src(self, f: Field) -> str:
        n = f.name
        if f.kind == "prim":
            return (f"@property\ndef {n}(self):\n    return self._row[{n!r}][0].item()\n"
                    f"@{n}.setter\ndef {n}(self, value):\n    self._row[{n!r}][0] = value\n")
        if f.kind == "ref":
            return (f"@property\ndef {n}(self):\n    return self._refs.get({n!r})\n"
                    f"@{n}.setter\ndef {n}(self, value):\n    _set_ref(self, {n!r}, value, {f.target!r})\n")
        if f.kind == "objlist":
            return (f"@property\ndef {n}(self):\n    return self._lists[{n!r}]\n"
                    f"@{n}.setter\ndef {n}(self, value):\n    _set_list(self, {n!r}, value)\n")
        return (f"@property\ndef {n}(self):\n    return self._arrays.get({n!r})\n"
                f"@{n}.setter\ndef {n}(self, value):\n    _set_array(self, {n!r}, value, {f.dtype!r}, {f.ndim})\n")

    def _wrapper_src(self, cls: str, m: str) -> str:
        """Python-facing method: coerce arguments, hand the JIT function this object's row
        viewed as `cls` (a record) plus its address.  Python's own MRO already chose the
        override, so no dispatcher is needed at the entry point."""
        fn = self.classes[cls].jit_methods[m]
        args = copy.deepcopy(fn.args)
        for a in args.args:
            a.annotation = None
        call = [f"self._rec({cls!r})", "self._addr"]
        for p in self.classes[cls].sigs[m]:
            if p.kind == "prim":
                call.append(f"{p.ann}({p.name})")
            elif p.kind == "ent":
                call.append(f"*_ent_pair({p.name}, {p.ent!r})")
            else:
                call.append(p.name)
        return f"def {m}({ast.unparse(args)}):\n    return {cls}__{m}({', '.join(call)})\n"

    def _gen_proxy(self, cls: str) -> str:
        info = self.classes[cls]
        bases = ", ".join(info.bases) if info.bases else "HybridRoot"
        # NOT f"{cls.lower()}_dtype": a class without new fields shares its parent's dtype
        own = self.proj_of[(cls, cls)]
        proj = ", ".join(f"{a!r}: {self.proj_of[(cls, a)]}" for a in self.mro[cls])
        lists = tuple((f.name, f.cap, f.target) for f in self.fields[cls].values() if f.kind == "objlist")
        prims = tuple(f.name for f in self.fields[cls].values() if f.kind == "prim")
        lines = [f"class {cls}({bases}):",
                 f"    TYPE_ID = TYPE_{cls.upper()}",
                 f"    _DTYPE = {own}",
                 f"    _ITEMSIZE = {own}.itemsize",
                 f"    _PROJ = {{{proj}}}",
                 f"    _OBJ_LIST_SPECS = {lists!r}",
                 f"    _PRIM_COLS = {prims!r}"]
        for attr, (g, _) in info.consts.items():
            lines.append(f"    {attr} = {g}")
        for f in info.decls.values():
            lines.append(textwrap.indent(self._prop_src(f), "    "))
        for m in info.jit_methods:
            lines.append(textwrap.indent(self._wrapper_src(cls, m), "    "))
        node = ast.parse("\n".join(lines)).body[0]
        node.body.extend(copy.deepcopy(info.py_nodes))       # user's plain-python members, verbatim
        return ast.unparse(node) + "\n"


# =====================================================================
# 3. LOWERING OF ONE @c METHOD BODY  (OOP python -> flat numba python)
# =====================================================================
class Lowerer(ast.NodeTransformer):
    """Rewrites one method body.

    `self` is a numpy record, so `self.x` on a primitive field is left EXACTLY as the user
    wrote it.  Everything else that touches an object is rewritten by asking `ptr()`:
    object-valued expressions are never materialised as Python objects; `ptr()` describes
    them as source text for (type, addr).  Every visitor that can legitimately receive an
    object (attribute access, method call, comparison, truth test, isinstance, for-loop over
    a list field, local assignment, call argument) asks `ptr()` first; if an object reaches
    any OTHER position, the visit_Name / visit_Attribute fallbacks raise a clear CompileError.
    """

    def __init__(self, comp: HybridCompiler, cls: str, method: str):
        self.c = comp
        self.cls = cls
        self.method = method
        self.locals: dict[str, str] = {}      # local/param name -> static class (object-valued)
        self.tmp = 0

    # ---- small utilities ---------------------------------------------------
    def _err(self, node, msg):
        file = self.c.classes[self.cls].file
        raise CompileError(f"{file}, line {getattr(node, 'lineno', '?')}: {self.cls}.{self.method}: {msg}")

    @staticmethod
    def _e(src: str) -> ast.expr:
        return ast.parse(src, mode="eval").body

    def src(self, node: ast.AST) -> str:
        """Lower an ordinary (non-object) expression and return its source text."""
        return ast.unparse(self.visit(copy.deepcopy(node)))

    def block(self, stmts: list[ast.stmt]) -> list[ast.stmt]:
        out: list[ast.stmt] = []
        for s in stmts:
            r = self.visit(s)
            if r is None:
                continue
            out.extend(r if isinstance(r, list) else [r])
        return out or [ast.Pass()]

    def declare(self, name: str, cls: str, node):
        if name in self.locals and self.locals[name] != cls:
            self._err(node, f"variable '{name}' is used as {self.locals[name]} and as {cls}; JIT locals need ONE static class")
        self.locals[name] = cls

    # ---- entry point -------------------------------------------------------
    def lower(self, fn: ast.FunctionDef) -> ast.FunctionDef:
        sig = self.c.classes[self.cls].sigs[self.method]
        for p in sig:
            if p.kind == "ent":
                self.locals[p.name] = p.ent
        new = copy.deepcopy(fn)
        new.name = f"{self.cls}__{self.method}"
        new.returns = None
        new.decorator_list = [self._e("njit(cache=True)")]
        flat = ["self", "self_addr"] + self.c.flat_params(sig)
        new.args = ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in flat],
                                 vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
        prologue = ast.parse("self_type = self.type_id").body
        new.body = prologue + self.block(new.body)
        return new

    # ---- reading columns / arrays of an object --------------------------------
    def col(self, owner: Ptr, f: Field, col: str) -> str:
        """Source text that READS column `col` (of field `f`) of `owner`.
        self: plain record attribute.  Anything else: a dispatched generated getter."""
        if owner.is_self:
            return f"self.{col}"
        m = self.c.getter(owner.cls, col, f.name)
        self.c.need_dispatch.add((owner.cls, m))
        return f"dispatch_{owner.cls}_{m}({owner.t}, {owner.a})"

    def arr_src(self, owner: Ptr, f: Field) -> str:
        """Source text for the numpy view of an array-like field (arrays and object lists)."""
        if owner.is_self:
            return f"{f.array_helper()}(self.{f.name}_data, self.{f.name}_shape)"
        m = self.c.array_getter(owner.cls, f)
        self.c.need_dispatch.add((owner.cls, m))
        return f"dispatch_{owner.cls}_{m}({owner.t}, {owner.a})"

    def const_src(self, owner: Ptr, attr: str) -> str:
        names = {self.c.const_for(d, attr) for d in self.c.descendants[owner.cls]}
        if len(names) == 1:
            return names.pop()
        m = self.c.const_getter(owner.cls, attr)         # overridden below: depends on runtime class
        self.c.need_dispatch.add((owner.cls, m))
        return f"dispatch_{owner.cls}_{m}({owner.t}, {owner.a})"

    # ---- object-expression analysis ------------------------------------------
    def ptr(self, node) -> Optional[Ptr]:
        """If `node` evaluates to a compiled object (or None-able reference), describe it."""
        if isinstance(node, ast.Name):
            if node.id == "self":
                return Ptr(self.cls, "self_type", "self_addr", True)
            if node.id in self.locals:
                return Ptr(self.locals[node.id], f"{node.id}_type", f"{node.id}_addr")
            return None
        if isinstance(node, ast.Attribute):
            owner = self.ptr(node.value)
            if owner is not None:
                f = self.c.fields[owner.cls].get(node.attr)
                if f is not None and f.kind == "ref":
                    return Ptr(f.target, self.col(owner, f, f"{f.name}_type"), self.col(owner, f, f"{f.name}_addr"))
            return None
        if isinstance(node, ast.Subscript):
            lst = self.list_of(node.value)
            if lst is not None:
                owner, f = lst
                if isinstance(node.slice, ast.Slice):
                    self._err(node, "slicing object lists is not supported in JIT code")
                arr = self.arr_src(owner, f)
                cnt = self.col(owner, f, f"{f.name}_count")
                i = f"_idx({self.src(node.slice)}, {cnt})"      # Python list indexing rules
                return Ptr(f.target, f"{arr}[{i}, 0]", f"{arr}[{i}, 1]")
        return None

    def list_of(self, node) -> Optional[tuple[Ptr, Field]]:
        if isinstance(node, ast.Attribute):
            owner = self.ptr(node.value)
            if owner is not None:
                f = self.c.fields[owner.cls].get(node.attr)
                if f is not None and f.kind == "objlist":
                    return owner, f
        return None

    def truth(self, node) -> ast.expr:
        """Python truthiness: an object is truthy iff it is not None."""
        p = self.ptr(node)
        if p is not None:
            return self._e(f"{p.a} != 0")
        return self.visit(node)

    def _why_not_field(self, owner: Ptr, attr: str) -> str:
        if any(attr in self.c.classes[a].py_fields for a in self.c.mro[owner.cls]):
            why = "a python-only (metadata) field"
        else:
            why = "not a declared typed field/constant (JIT code cannot use undeclared attributes or call non-@c methods)"
        return f"'{owner.cls}.{attr}' is {why}"

    # ---- fallbacks that catch objects used in unsupported positions ----------
    _OBJ_MSG = ("object references and object lists can only be used as: an attribute owner (self.ref.x), a "
                "method receiver, an argument for an object parameter, in ==/is/!= comparisons, in a truth "
                "test, as a for-loop iterable (lists), in len(), in isinstance(), or assigned to a local")

    def visit_Name(self, node: ast.Name):
        if node.id == "self" or node.id in self.locals:
            self._err(node, f"'{node.id}' is an object; {self._OBJ_MSG}")
        return node

    def visit_Attribute(self, node: ast.Attribute):
        # Class.CONSTANT  (static lookup, like Python)
        if isinstance(node.value, ast.Name) and node.value.id in self.c.classes and node.value.id not in self.locals:
            g = self.c.const_for(node.value.id, node.attr)
            if g is None:
                self._err(node, f"{node.value.id}.{node.attr} is not a class constant")
            return ast.Name(id=g, ctx=ast.Load())
        owner = self.ptr(node.value)
        if owner is None:
            return self.generic_visit(node)           # np.pi, arr.shape, ...
        attr = node.attr
        if any(attr in self.c.classes[a].consts for a in self.c.mro[owner.cls]):
            return self._e(self.const_src(owner, attr))
        f = self.c.fields[owner.cls].get(attr)
        if f is None:
            self._err(node, self._why_not_field(owner, attr))
        if f.kind == "prim":
            if owner.is_self:
                return node                            # `self.x` is record attribute access: untouched
            if isinstance(node.ctx, ast.Load):
                return self._e(self.col(owner, f, attr))
            self._err(node, "unsupported assignment target")
        if f.kind == "array":
            if not isinstance(node.ctx, ast.Load):
                self._err(node, f"cannot re-point array field '{attr}' in JIT code (JIT writes data, Python writes pointers); "
                                f"write into it instead: self.{attr}[i] = ...")
            return self._e(self.arr_src(owner, f))
        self._err(node, f"'{attr}' is an object {'reference' if f.kind == 'ref' else 'list'}; {self._OBJ_MSG}")

    # ---- statements ----------------------------------------------------------
    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) != 1:
            self._err(node, "chained assignment is not supported in JIT code")
        tgt = node.targets[0]
        if isinstance(tgt, ast.Name):
            if tgt.id == "self":
                self._err(node, "cannot rebind self")
            p = self.ptr(node.value)
            if p is not None:                           # x = <object>   -> two ints
                self.declare(tgt.id, p.cls, node)
                return ast.parse(f"{tgt.id}_type, {tgt.id}_addr = {p.t}, {p.a}").body[0]
            if isinstance(node.value, ast.Constant) and node.value.value is None and tgt.id in self.locals:
                return ast.parse(f"{tgt.id}_type, {tgt.id}_addr = 0, 0").body[0]
        if isinstance(tgt, ast.Attribute):
            owner = self.ptr(tgt.value)
            if owner is not None:
                f = self.c.fields[owner.cls].get(tgt.attr)
                if f is None:
                    self._err(node, self._why_not_field(owner, tgt.attr))
                if f.kind != "prim":
                    self._err(node, f"JIT code cannot assign {f.kind} field '{tgt.attr}' (JIT writes data, "
                                    f"Python writes pointers). Assign it from Python, or write into it in place.")
                if not owner.is_self:                   # other.x = v  -> dispatch_K__hy_set_x(type, addr, v)
                    m = self.c.setter(owner.cls, tgt.attr, f.name)
                    self.c.need_dispatch.add((owner.cls, m))
                    return ast.Expr(value=ast.Call(func=ast.Name(id=f"dispatch_{owner.cls}_{m}", ctx=ast.Load()),
                                                   args=[self._e(owner.t), self._e(owner.a), self.visit(node.value)], keywords=[]))
        if isinstance(tgt, ast.Subscript) and self.list_of(tgt.value) is not None:
            self._err(node, "JIT code cannot assign list elements (Python writes pointers)")
        return self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        s = HybridCompiler._ann_name(node.annotation)
        if s in self.c.classes and isinstance(node.target, ast.Name):
            self.declare(node.target.id, s, node)
            if node.value is None:
                return ast.Pass()
        if node.value is None:
            return ast.Pass()
        return self.visit(ast.Assign(targets=[node.target], value=node.value, lineno=node.lineno))

    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.target, ast.Attribute):
            owner = self.ptr(node.target.value)
            if owner is not None:
                f = self.c.fields[owner.cls].get(node.target.attr)
                if f is None:
                    self._err(node, self._why_not_field(owner, node.target.attr))
                if f.kind != "prim":
                    self._err(node, f"'{node.target.attr}' cannot be updated in JIT code")
                if not owner.is_self:                   # other.x += v  -> set(get() op v)
                    attr = node.target.attr
                    m = self.c.setter(owner.cls, attr, f.name)
                    self.c.need_dispatch.add((owner.cls, m))
                    get = self._e(self.col(owner, f, attr))
                    new = ast.BinOp(left=get, op=node.op, right=self.visit(node.value))
                    return ast.Expr(value=ast.Call(func=ast.Name(id=f"dispatch_{owner.cls}_{m}", ctx=ast.Load()),
                                                   args=[self._e(owner.t), self._e(owner.a), new], keywords=[]))
        return self.generic_visit(node)

    def visit_For(self, node: ast.For):
        lst = self.list_of(node.iter)
        if lst is None:
            return self.generic_visit(node)
        owner, f = lst
        if not isinstance(node.target, ast.Name):
            self._err(node, "iterating an object list needs a simple loop variable")
        var = node.target.id
        self.tmp += 1
        n, r, i = f"_n{self.tmp}", f"_r{self.tmp}", f"_i{self.tmp}"
        self.declare(var, f.target, node)
        stmts = ast.parse(
            f"{n} = {self.col(owner, f, f.name + '_count')}\n"
            f"{r} = {self.arr_src(owner, f)}\n"
            f"for {i} in range({n}):\n"
            f"    {var}_type = {r}[{i}, 0]\n"
            f"    {var}_addr = {r}[{i}, 1]\n").body
        loop = stmts[2]
        loop.body = loop.body + self.block(node.body)
        loop.orelse = self.block(node.orelse) if node.orelse else []
        return stmts

    def visit_If(self, node: ast.If):
        node.test = self.truth(node.test)
        node.body = self.block(node.body)
        node.orelse = self.block(node.orelse) if node.orelse else []
        return node

    def visit_While(self, node: ast.While):
        node.test = self.truth(node.test)
        node.body = self.block(node.body)
        node.orelse = self.block(node.orelse) if node.orelse else []
        return node

    def visit_IfExp(self, node: ast.IfExp):
        node.test = self.truth(node.test)
        node.body = self.visit(node.body)
        node.orelse = self.visit(node.orelse)
        return node

    def visit_BoolOp(self, node: ast.BoolOp):
        node.values = [self.truth(v) for v in node.values]
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            p = self.ptr(node.operand)
            if p is not None:
                return self._e(f"{p.a} == 0")
        return self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        operands = [node.left] + node.comparators
        ptrs = [self.ptr(o) for o in operands]
        if not any(ptrs):
            return self.generic_visit(node)
        if len(node.ops) != 1:
            self._err(node, "chained comparisons on objects are not supported")
        op, (lp, rp) = node.ops[0], ptrs
        is_none = lambda n: isinstance(n, ast.Constant) and n.value is None
        eq = isinstance(op, (ast.Is, ast.Eq))
        if not isinstance(op, (ast.Is, ast.IsNot, ast.Eq, ast.NotEq)):
            self._err(node, "objects only support ==, !=, is, is not")
        sym = "==" if eq else "!="
        if lp and is_none(node.comparators[0]):
            return self._e(f"{lp.a} {sym} 0")
        if rp and is_none(node.left):
            return self._e(f"{rp.a} {sym} 0")
        if lp and rp:
            return self._e(f"{lp.a} {sym} {rp.a}")         # same address <=> same object
        self._err(node, "cannot compare an object with a non-object")

    # ---- calls ---------------------------------------------------------------
    def _bind(self, node: ast.Call, sig: list[Param], what: str) -> list[tuple[Param, ast.expr]]:
        """Match call arguments to the signature (positional, keyword, defaults)."""
        if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
            self._err(node, f"*args/**kwargs are not supported in calls to {what}")
        if len(node.args) > len(sig):
            self._err(node, f"too many arguments in call to {what}")
        vals = {p.name: a for p, a in zip(sig, node.args)}
        names = {p.name for p in sig}
        for k in node.keywords:
            if k.arg not in names:
                self._err(node, f"{what} has no parameter '{k.arg}'")
            if k.arg in vals:
                self._err(node, f"parameter '{k.arg}' given twice in call to {what}")
            vals[k.arg] = k.value
        out = []
        for p in sig:
            if p.name not in vals:
                if p.default is None:
                    self._err(node, f"missing argument '{p.name}' in call to {what}")
                vals[p.name] = copy.deepcopy(p.default)
            out.append((p, vals[p.name]))
        return out

    def _expand(self, node, bound) -> list[ast.expr]:
        """Call arguments -> flat JIT arguments (object args become (type, addr))."""
        args: list[ast.expr] = []
        for p, a in bound:
            if p.kind == "ent":
                pt = self.ptr(a)
                if pt is not None:
                    if p.ent not in self.c.mro[pt.cls]:
                        self._err(node, f"argument '{p.name}' expects {p.ent}, got {pt.cls}")
                    args += [self._e(pt.t), self._e(pt.a)]
                elif isinstance(a, ast.Constant) and a.value is None:
                    args += [ast.Constant(value=0), ast.Constant(value=0)]
                else:
                    self._err(node, f"argument '{p.name}' must be an object of class {p.ent} or None")
            else:
                args.append(self.visit(a))
        return args

    def visit_Call(self, node: ast.Call):
        f = node.func
        # isinstance(obj, C) / isinstance(obj, (A, B))
        if isinstance(f, ast.Name) and f.id == "isinstance" and len(node.args) == 2:
            p = self.ptr(node.args[0])
            if p is None:
                self._err(node, "isinstance() is only supported on compiled objects in JIT code")
            cls_nodes = node.args[1].elts if isinstance(node.args[1], ast.Tuple) else [node.args[1]]
            names = []
            for cn in cls_nodes:
                if not (isinstance(cn, ast.Name) and cn.id in self.c.classes):
                    self._err(node, "isinstance() second argument must be compiled class name(s)")
                names.append(cn.id)
            return self._e(" or ".join(f"is_instance({p.t}, {p.a}, TYPE_{n.upper()})" for n in names))
        if isinstance(f, ast.Name) and f.id == "issubclass" and len(node.args) == 2:
            a, b = node.args
            if not all(isinstance(x, ast.Name) and x.id in self.c.classes for x in (a, b)):
                self._err(node, "issubclass() needs compiled class names")
            return self._e(f"is_subclass(TYPE_{a.id.upper()}, TYPE_{b.id.upper()})")
        if isinstance(f, ast.Name) and f.id == "len" and len(node.args) == 1:
            lst = self.list_of(node.args[0])
            if lst is not None:
                return self._e(self.col(lst[0], lst[1], lst[1].name + "_count"))
        if isinstance(f, ast.Name) and f.id in self.c.classes:
            self._err(node, f"cannot construct compiled objects ('{f.id}(...)') in JIT code; create them in Python")
        # super().m(...)
        if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Call)
                and isinstance(f.value.func, ast.Name) and f.value.func.id == "super"):
            m = f.attr
            sig = self.c._super_sig(self.cls, m)
            self.c.need_super.add((self.cls, m))
            args = self._expand(node, self._bind(node, sig, f"super().{m}"))
            return ast.Call(func=ast.Name(id=f"super_from_{self.cls}_{m}", ctx=ast.Load()),
                            args=[ast.Name(id="self_type", ctx=ast.Load()), ast.Name(id="self_addr", ctx=ast.Load())] + args, keywords=[])
        # obj.method(...)  /  self.method(...)  -> virtual dispatch
        if isinstance(f, ast.Attribute):
            owner = self.ptr(f.value)
            if owner is not None:
                m = f.attr
                if self.c.find_method(owner.cls, m) is None:
                    self._err(node, f"'{owner.cls}.{m}' is not a @c method; JIT methods may only call other @c methods")
                self.c.need_dispatch.add((owner.cls, m))
                args = self._expand(node, self._bind(node, self.c.sig_of(owner.cls, m), f"{owner.cls}.{m}"))
                return ast.Call(func=ast.Name(id=f"dispatch_{owner.cls}_{m}", ctx=ast.Load()),
                                args=[self._e(owner.t), self._e(owner.a)] + args, keywords=[])
        return self.generic_visit(node)


# =====================================================================
# 4. FIXED TEXT BLOCKS COPIED INTO EVERY OUTPUT MODULE
# =====================================================================
_HEADER = '''\
# AUTO-GENERATED by hybrid_compiler.py -- do not edit.  Delete the numba cache (__pycache__/*.nbi,*.nbc)
# whenever this file is regenerated: Numba's cache does not notice changed globals/dtypes.
from __future__ import annotations
import os
import numpy as np

# HYBRID_DEBUG=1 -> run everything as plain Python (no Numba needed): pdb, print and breakpoints work.
_DEBUG = os.environ.get("HYBRID_DEBUG") == "1"

if _DEBUG:
    import ctypes

    def njit(*a, **k):
        """Debug stand-in: the decorated function stays an ordinary Python function."""
        return a[0] if a and callable(a[0]) else (lambda f: f)

    def _to_voidptr(addr):
        return int(addr)

    class _Rec:
        """Debug stand-in for a Numba record: attribute access reads/writes the numpy void
        scalar it wraps (which is a VIEW of the row's memory, so writes go through).
        Not np.record: numpy's own attributes (size, data, ...) would shadow field names."""
        __slots__ = ("_v",)

        def __init__(self, v):
            object.__setattr__(self, "_v", v)

        def __getattr__(self, name):             # only runs when normal lookup fails
            return self._v[name]

        def __setattr__(self, name, value):
            self._v[name] = value

        def __repr__(self):
            return f"_Rec({self._v!r})"

    class _RecArray:
        """carray(...) over a record dtype: indexing gives a _Rec."""
        __slots__ = ("_a",)

        def __init__(self, a):
            self._a = a

        def __getitem__(self, i):
            return _Rec(self._a[i])

    def carray(ptr, shape, dtype=None):
        """Debug stand-in for numba.carray: a numpy view over raw memory (ctypes)."""
        dt = np.dtype(dtype)
        shp = (shape,) if isinstance(shape, (int, np.integer)) else tuple(int(s) for s in shape)
        nbytes = int(np.prod(shp)) * dt.itemsize
        buf = (ctypes.c_char * nbytes).from_address(int(ptr))
        arr = np.frombuffer(buf, dtype=dt).reshape(shp)
        return _RecArray(arr) if dt.names else arr
else:
    from numba import njit, types, carray
    from numba.extending import intrinsic
    from llvmlite import ir

    _Rec = None

    @intrinsic
    def _to_voidptr(typingctx, addr):
        """int address -> void*.  Numba has no pure-python way to do this; the LLVM
        instruction `inttoptr` does exactly it.  `ir.IntType(8).as_pointer()` is LLVM's
        generic byte pointer (i8*), which is what Numba's `voidptr` maps to."""
        def codegen(context, builder, signature, args):
            return builder.inttoptr(args[0], ir.IntType(8).as_pointer())
        return types.voidptr(types.intp), codegen

'''

_ANCESTRY_FUNCS = '''
@njit(cache=True)
def is_instance(obj_type, obj_addr, target_type_id):
    """isinstance() on a (type, addr) pointer; None (addr 0) is never an instance."""
    if obj_addr == 0:
        return False
    return ancestry_table[obj_type, target_type_id]


@njit(cache=True)
def is_subclass(child_type_id, parent_type_id):
    return ancestry_table[child_type_id, parent_type_id]


@njit(cache=True)
def _idx(i, n):
    """Python list indexing rules: negative wraps around, out of range raises."""
    if i < 0:
        i += n
    if i < 0 or i >= n:
        raise IndexError('list index out of range')
    return i

'''

_RUNTIME = '''
# =====================================================================
# Python runtime: proxy base class + helpers.  Runs outside the hot path.
# =====================================================================
from collections.abc import MutableSequence


def _ent_pair(value, cls_name):
    """proxy (or None) -> the (type_id, address) fat pointer JIT code expects.
    The isinstance check is the only thing standing between a wrong object and
    silent memory corruption, so it is not optional."""
    if value is None:
        return (0, 0)
    if not isinstance(value, globals()[cls_name]):
        raise TypeError(f"expected {cls_name} or None, got {type(value).__name__}")
    return (value.TYPE_ID, value._addr)


def _set_ref(owner, name, value, cls_name):
    """Python is the only writer of pointers.  Store the strong reference FIRST-CLASS in
    owner._refs (keeps the target alive) and mirror it into the row as (type, addr)."""
    t, a = _ent_pair(value, cls_name)
    if value is None:
        owner._refs.pop(name, None)
    else:
        owner._refs[name] = value
    owner._row[name + '_type'][0] = t
    owner._row[name + '_addr'][0] = a


def _set_array(owner, name, value, dtype, ndim):
    """np.ndarray fields: forced C-contiguous (JIT rebuilds the view with carray).
    If `value` is already contiguous with the right dtype it is NOT copied, so the
    caller's array and the field share memory."""
    row = owner._row
    if value is None:
        owner._arrays.pop(name, None)
        row[name + '_data'][0] = 0
        row[name + '_shape'][0, :] = 0
        return
    arr = np.ascontiguousarray(value, dtype=dtype)
    if arr.ndim != ndim:
        raise ValueError(f"{name}: expected {ndim}-d array, got {arr.ndim}-d")
    owner._arrays[name] = arr
    row[name + '_data'][0] = arr.ctypes.data
    row[name + '_shape'][0, :] = arr.shape


def _set_list(owner, name, value):
    wrapper = owner._lists[name]
    if value is wrapper:
        return
    wrapper._commit([] if value is None else list(value))


class _ObjList(MutableSequence):
    """list[Class] field.  Python keeps a normal list of strong references (`_items`);
    the row describes an int64 array of shape (cap, 2) -- (type_id, addr) per element --
    exactly like a numpy array field (name_data, name_shape), plus name_count.
    EVERY mutation builds a new item list, validates it, and rewrites the buffer (O(n),
    but always consistent).  Growth just allocates a bigger buffer and updates the row --
    JIT code never resizes."""

    def __init__(self, owner, name, cap, elem):
        self._owner, self._name, self._elem = owner, name, elem
        self._items = []
        self._resize(max(1, cap))
        self._sync()

    def _resize(self, cap):
        self._cap = cap
        self._buf = np.zeros((cap, 2), dtype=np.int64)      # owned here -> address stays valid

    def _sync(self):
        items = self._items
        if len(items) > self._cap:
            self._resize(max(len(items), 2 * self._cap))
        buf = self._buf
        buf[:] = 0
        for i, it in enumerate(items):
            if it is not None:
                buf[i, 0] = it.TYPE_ID
                buf[i, 1] = it._addr
        row = self._owner._row
        row[self._name + '_data'][0] = buf.ctypes.data
        row[self._name + '_shape'][0, :] = buf.shape
        row[self._name + '_count'][0] = len(items)

    def _commit(self, new_items):
        cls = globals()[self._elem]
        for it in new_items:
            if it is not None and not isinstance(it, cls):
                raise TypeError(f"{self._name}: expected {self._elem} or None, got {type(it).__name__}")
        self._items = new_items
        self._sync()

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __setitem__(self, i, v):
        new = list(self._items); new[i] = v; self._commit(new)

    def __delitem__(self, i):
        new = list(self._items); del new[i]; self._commit(new)

    def insert(self, i, v):
        new = list(self._items); new.insert(i, v); self._commit(new)

    def extend(self, values):
        self._commit(self._items + list(values))

    def clear(self):
        self._commit([])

    def reverse(self):
        self._commit(self._items[::-1])

    def sort(self, *, key=None, reverse=False):
        self._commit(sorted(self._items, key=key, reverse=reverse))

    def __repr__(self):
        return f"_ObjList({self._items!r})"


_INTERNAL_ATTRS = frozenset(('_mem', '_row', '_addr', '_refs', '_arrays', '_views', '_recs', '_lists'))


class HybridRoot:
    """Root of every generated proxy class (the user's classes are rewritten to inherit
    from it).  Owns the memory, the strong references and pickling.

        _mem    int64 buffer, the actual storage (8-byte aligned)
        _row    1-element array of this class's dtype over _mem   (NEVER reassigned)
        _addr   address of _mem: what other objects store as a pointer
        _refs   strong refs to every object this one points to (keeps targets alive)
        _arrays strong refs to numpy arrays whose addresses are stored in the row
        _lists  _ObjList wrappers
        _views  cache of projected views (_row seen as an ancestor)
        _recs   cache of those views' first record: what a JIT method receives as `self`
    """

    def __new__(cls, *args, **kwargs):
        # Because we override __new__, object.__init__ would silently swallow constructor
        # arguments when no class in the MRO defines __init__.  Plain Python raises instead.
        if (args or kwargs) and cls.__init__ is object.__init__:
            raise TypeError(f"{cls.__name__}() takes no arguments")
        self = object.__new__(cls)
        self._mem = np.zeros(cls._ITEMSIZE // 8, dtype=np.int64)
        self._row = np.frombuffer(self._mem, dtype=cls._DTYPE, count=1)
        self._addr = self._mem.ctypes.data
        self._row['type_id'][0] = cls.TYPE_ID
        self._refs, self._arrays, self._views, self._recs = {}, {}, {}, {}
        self._lists = {n: _ObjList(self, n, cap, elem) for n, cap, elem in cls._OBJ_LIST_SPECS}
        return self

    def _view(self, ancestor):
        """This object's memory viewed as `ancestor`'s dtype (a 1-element array)."""
        v = self._views.get(ancestor)
        if v is None:
            v = np.frombuffer(self._mem, dtype=self._PROJ[ancestor], count=1)
            self._views[ancestor] = v
        return v

    def _rec(self, ancestor):
        """The record a JIT method defined in `ancestor` receives as `self` (a VIEW of this
        object's memory: writes through it land in the row)."""
        r = self._recs.get(ancestor)
        if r is None:
            r = self._view(ancestor)[0]
            if _DEBUG:
                r = _Rec(r)
            self._recs[ancestor] = r
        return r

    # --- pickle / deepcopy: addresses are meaningless in a new process, so rebuild
    # --- every pointer through the property setters (which recompute addresses).
    def __getstate__(self):
        return {
            'prims': {n: self._row[n][0].item() for n in self._PRIM_COLS},
            'refs': dict(self._refs),
            'lists': {n: list(l) for n, l in self._lists.items()},
            'arrays': dict(self._arrays),
            'py': {k: v for k, v in self.__dict__.items() if k not in _INTERNAL_ATTRS},
        }

    def __setstate__(self, state):
        for n, v in state['prims'].items():
            self._row[n][0] = v
        for n, v in state['refs'].items():
            setattr(self, n, v)
        for n, v in state['lists'].items():
            setattr(self, n, v)
        for n, v in state['arrays'].items():
            setattr(self, n, v)
        self.__dict__.update(state['py'])

'''


# =====================================================================
# 5. COMMAND LINE:  python hybrid_compiler.py my_classes.py  ->  my_classes_compiled.py
# =====================================================================
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python hybrid_compiler.py <source.py>")
    src_path = sys.argv[1]
    out_path = src_path[:-3] + "_compiled.py"
    with open(src_path) as fh:
        code = HybridCompiler().compile(fh.read())
    with open(out_path, "w") as fh:
        fh.write(code)
    print(f"wrote {out_path}")
