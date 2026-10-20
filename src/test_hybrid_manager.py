"""End-to-end test of HybridCompilerManager in a throw-away project (debug mode: no Numba needed)."""
import importlib
import os
import shutil
import sys
import tempfile
import textwrap
import warnings
from pathlib import Path

os.environ["HYBRID_DEBUG"] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hybrid_compiler import CompileError  # noqa: E402
from hybrid_compiler_manager import HybridCompilerManager  # noqa: E402

root = Path(tempfile.mkdtemp())
sys.path.insert(0, str(root))
pkg = root / "demo_pkg"
(pkg / "models").mkdir(parents=True)


def w(rel, text):
    (pkg / rel).write_text(textwrap.dedent(text))


w("__init__.py", """
    from hybrid_compiler_manager import HybridCompilerManager
    HybridCompilerManager(__file__, __name__, source_dir="models", verbose=False).build()
""")
w("models/__init__.py", "")
w("models/helpers.py", """
    def clamp(x, lo, hi):
        return max(lo, min(hi, x))
""")
w("models/base.py", """
    from __future__ import annotations
    from hybrid_compiler import c
    from .helpers import clamp                      # NOT a @c module: kept, made absolute

    @c
    class Entity:
        MAX: float = 100.0
        health: float
        @c
        def take_damage(self, amount: float) -> None:
            self.health -= amount
        @c
        def update(self, dt: float) -> None:
            self.health += dt
        def safe_health(self):
            return clamp(self.health, 0.0, 50.0)
""")
w("models/tanks.py", """
    from __future__ import annotations
    import numpy as np
    from hybrid_compiler import c
    from .base import Entity                        # another @c module: dropped, merged namespace

    @c
    class Tank(Entity):
        heat: np.ndarray[float, 1]
        @c
        def update(self, dt: float) -> None:
            super().update(dt)
            self.heat[0] += dt
""")


def fresh_import():
    for k in [k for k in sys.modules if k.startswith("demo_pkg")]:
        del sys.modules[k]
    importlib.invalidate_caches()
    return importlib.import_module("demo_pkg.compiled")


# 1. importing the package compiles on first use
mod = fresh_import()
assert (pkg / "compiled.py").exists() and (pkg / "compiled.pyi").exists()
first_line = (pkg / "compiled.py").read_text().splitlines()[0]
assert first_line.startswith("# hybrid-hash: ")
t = mod.Tank(); t.health = 10.0; t.heat = [0.0, 0.0]
t.update(2.0)
assert t.health == 12.0 and t.heat[0] == 2.0 and isinstance(t, mod.Entity)    # cross-module inheritance works
t.health = 99.0
assert t.safe_health() == 50.0                                                # helper import survived
print("ok  first import compiled two modules into one namespace; cross-module inheritance works")

# 2. the IDE stub points at the ORIGINAL classes
stub = (pkg / "compiled.pyi").read_text()
assert "from .models.base import" in stub and "Entity as Entity" in stub and "Tank as Tank" in stub
assert "helpers" not in stub
print("ok  .pyi re-exports the original classes")

# 3. unchanged -> no rebuild; changed -> rebuild + numba cache purge
b = HybridCompilerManager(pkg / "__init__.py", "demo_pkg", "models", verbose=False)
assert b.build() is False
cache = pkg / "__pycache__"
cache.mkdir(exist_ok=True)
stale = [cache / "compiled.dispatch_Entity_update-12.py312.nbi", cache / "compiled.dispatch_Entity_update-12.py312.1.nbc"]
keep = cache / "other.thing-3.py312.nbi"
for f in stale + [keep]:
    f.write_text("x")
w("models/base.py", (pkg / "models/base.py").read_text().replace("100.0", "200.0"))
assert b.build() is True
assert not any(f.exists() for f in stale) and keep.exists()
assert "= 200.0" in (pkg / "compiled.py").read_text()
assert b.build() is False
print("ok  unchanged skips; edit recompiles, purges only this module's numba cache files")

# 4. warns when regenerated after import in the same process
w("models/base.py", (pkg / "models/base.py").read_text().replace("200.0", "300.0"))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    b.build()
assert any("restart" in str(x.message) for x in caught)
print("ok  warns about regenerating an already-imported module")

# 5. compile errors: named file + line, previous output left intact
good = (pkg / "compiled.py").read_text()
w("models/bad.py", """
    from hybrid_compiler import c
    @c
    class Bad:
        name: str
        @c
        def f(self) -> None:
            self.name = "x"
""")
try:
    b.build()
    raise SystemExit("expected CompileError")
except CompileError as e:
    assert "bad.py" in str(e) and "python-only" in str(e), e
    print("ok  error names the file:", str(e)[-90:])
assert (pkg / "compiled.py").read_text() == good
(pkg / "models/bad.py").unlink()

# 6. duplicate names across modules are rejected
w("models/dup.py", """
    from hybrid_compiler import c
    @c
    class Entity:
        x: float
""")
try:
    b.build()
    raise SystemExit("expected CompileError")
except CompileError as e:
    assert "share one namespace" in str(e)
    print("ok  duplicate class name across modules rejected")

shutil.rmtree(root, ignore_errors=True)
print("\nALL MANAGER TESTS PASSED")
