"""
Tests for hybrid_compiler.

They run in DEBUG MODE (HYBRID_DEBUG=1): the generated "JIT" functions execute as plain
Python, `carray` is ctypes over the address, and records are attribute wrappers over numpy
void scalars.  That verifies the COMPILER (layouts, projections, dispatch, super routing,
pointer lowering, proxies, pickling) against a plain-python reference run -- but NOT that
Numba accepts the generated code.  Run numba_smoke_test.py on a machine with Numba for that.
"""
import copy
import gc
import importlib.util
import os
import pickle
import sys

os.environ["HYBRID_DEBUG"] = "1"
import numpy as np

sys.path.insert(0, ".")
from hybrid_compiler import HybridCompiler, CompileError  # noqa: E402

SRC = open("example_classes.py").read()


def load(code, name):
    open(f"{name}.py", "w").write(code)
    spec = importlib.util.spec_from_file_location(name, f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


import example_classes as ref                       # plain python reference
code = HybridCompiler().compile(SRC)
jit = load(code, "_compiled_engine")                # compiled


def scenario(m):
    """Same operations on reference and compiled module; returns observable state."""
    e1 = m.Entity(); e1.health = 100.0; e1.name = "grunt"
    au = m.ArmoredUnit(); au.health = 200.0; au.armor = 30.0
    apc = m.APC(); apc.health = 300.0; apc.armor = 10.0; apc.fuel = 50.0; apc.state = 0.0; apc.speed = 0.0
    t = m.CommandTank()
    t.squad = [e1, au]
    t.squad.append(apc)
    t.squad.append(None)
    t.squad.append(e1)                              # 5th element: exceeds the annotated capacity 4
    for _ in range(5):
        t.update(0.5)
    apc.update(1.0)
    t.engage(au)
    t.engage(e1)
    t.engage(None)
    e1.apply_hazard(10.0)                           # Entity.apply_hazard -> Entity.take_damage
    t.apply_hazard(30.0)                            # virtual: CommandTank.take_damage (halved) via ArmoredUnit
    au.apply_hazard(50.0)
    return dict(
        tank=(t.health, t.armor, t.fuel, t.speed, t.state, t.is_visible, t.visual_state, t.mesh_id),
        heat=tuple(t.heat), e1=e1.health, au=(au.health, au.armor),
        apc=(apc.health, apc.armor, apc.fuel, apc.speed, apc.state),
        squad_health=t.squad_health(), threat=t.threat(au), threat_none=t.threat(None),
        commander_is_self=t.commander is t, describe=e1.describe(),
    )


def approx(a, b):
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(approx(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return abs(a - b) < 1e-9
    return a == b


def test_behaviour_matches_python():
    r, j = scenario(ref), scenario(jit)
    for k in r:
        assert approx(r[k], j[k]), f"MISMATCH {k}: python={r[k]} compiled={j[k]}"
    print("ok  behaviour identical to plain python:", len(r), "observables")


def test_generated_code_shape():
    """`self` is a record: the user's own self.x lines survive untouched; pointer access is
    always a dispatched generated method, never hand-written carray in lowered code."""
    assert "def Entity__take_damage(self, self_addr, amount):" in code
    assert "    self.health -= amount" in code                      # untouched user line
    assert "self_row" not in code
    assert "return self.health" in code                             # generated getter body
    assert "dispatch_Entity__hy_set_health(unit_type, unit_addr" in code
    lowered = code[code.index("def CommandTank__squad_health"):code.index("def CommandTank__update")]
    assert "carray" not in lowered and "_list_type" not in code
    print("ok  generated code: user self.x lines untouched, pointer access via dispatched accessors")


def test_layout_and_projections():
    comp = HybridCompiler(); comp.compile(SRC)
    off = lambda cls, col: {c[0]: c[3] for c in comp.layouts[cls].cols}[col]
    assert off("CommandTank", "health") == off("Entity", "health") == off("APC", "health")
    ent_views = {comp.proj_of[(d, "Entity")] for d in comp.descendants["Entity"]}
    print(f"ok  Entity viewed from {len(comp.descendants['Entity'])} classes uses {len(ent_views)} dtype(s)")
    assert len(ent_views) == 1
    assert comp.proj_of[("APC", "ArmoredUnit")] != comp.proj_of[("ArmoredUnit", "ArmoredUnit")]
    print("ok  diamond base gets a distinct projected dtype:", comp.proj_of[("APC", "ArmoredUnit")])
    assert jit.armoredunit_dtype.isalignedstruct


def test_projected_view_reads_child_memory():
    t = jit.CommandTank(); t.armor = 77.0
    v = t._view("ArmoredUnit")                      # APC-layout memory seen as ArmoredUnit
    assert v[0]["armor"] == 77.0 and v[0]["type_id"] == jit.TYPE_COMMANDTANK
    assert t._rec("ArmoredUnit").armor == 77.0       # the record a JIT method would receive as self
    print("ok  ArmoredUnit view over CommandTank memory sees the right bytes")


def test_gc_lifetime_and_lists():
    t = jit.CommandTank()
    e = jit.Entity(); e.health = 5.0
    t.squad.append(e)
    addr = e._addr
    del e
    gc.collect()
    assert t.squad[0]._addr == addr and t.squad_health() == 5.0        # tank keeps it alive
    t.squad.clear(); t.squad.extend([jit.Entity() for _ in range(40)])  # grows past cap
    assert len(t.squad) == 40 and t._row["squad_shape"][0, 0] >= 40
    try:
        t.squad.append(3.14); raise SystemExit("should have rejected non-Entity")
    except TypeError:
        pass
    try:
        t.commander = jit.Renderable(); raise SystemExit("should have rejected wrong class")
    except TypeError:
        pass
    print("ok  strong refs keep targets alive; lists grow; wrong types rejected")


def test_pickle_and_deepcopy():
    t = jit.CommandTank(); t.name = "orig"
    a = jit.Entity(); a.health = 11.0
    t.squad = [a, t]                                 # list containing the owner itself (cycle)
    t.heat = np.array([1.0, 2.0, 3.0])
    for clone in (pickle.loads(pickle.dumps(t)), copy.deepcopy(t)):
        assert clone is not t and clone._addr != t._addr
        assert clone.name == "orig" and clone.health == t.health
        assert clone.commander is clone                              # cycle preserved, re-linked
        assert clone.squad[1] is clone and clone.squad[0].health == 11.0
        assert clone._row["squad_count"][0] == 2 and list(clone.heat) == [1.0, 2.0, 3.0]
        clone.update(1.0)                                            # JIT path works on the clone
        assert t.heat[0] == 1.0                                      # original untouched (own row/array)
    print("ok  pickle + deepcopy re-link all pointers (incl. cycles)")


def test_subclass_without_new_fields():
    """Regression: a subclass that only overrides methods has the parent's exact layout,
    so its dtype is de-duplicated into the parent's.  The proxy must use the shared dtype."""
    src = ("from hybrid_compiler import c\n@c\nclass A:\n    x: float\n    @c\n    def f(self) -> None:\n        self.x += 1.0\n"
           "@c\nclass B(A):\n    @c\n    def f(self) -> None:\n        super().f()\n        self.x += 10.0\n")
    m = load(HybridCompiler().compile(src), "_compiled_nofields")
    b = m.B(); b.f()
    assert b.x == 11.0 and b._row["type_id"][0] == m.TYPE_B and m.A().x == 0.0
    print("ok  subclass without new fields shares its parent's dtype but keeps its own type id")


def test_override_extends_parent_signature():
    """isinstance narrows the receiver so a guarded call can pass child-only arguments."""
    src = ("from hybrid_compiler import compiled\n"
           "@compiled\nclass A:\n"
           "    total: int\n"
           "    @compiled\n"
           "    def hit(self, amount: int) -> None:\n"
           "        self.total += amount\n"
           "@compiled\nclass B(A):\n"
           "    @compiled\n"
           "    def hit(self, amount: int, bonus: int = 3) -> None:\n"
           "        super().hit(amount + bonus)\n")
    src += ("@compiled\nclass Holder:\n"
            "    target: A\n"
            "    @compiled\n"
            "    def invoke(self, amount: int) -> None:\n"
            "        if isinstance(self.target, B):\n"
            "            self.target.hit(amount, bonus=4)\n"
            "        else:\n"
            "            self.target.hit(amount)\n")
    compiled = HybridCompiler().compile(src)
    assert "dispatch_B_hit" in compiled
    unguarded_src = src.replace(
        "if isinstance(self.target, B):\n"
        "            self.target.hit(amount, bonus=4)\n"
        "        else:\n"
        "            self.target.hit(amount)",
        "self.target.hit(amount, bonus=4)")
    try:
        HybridCompiler().compile(unguarded_src)
    except CompileError as error:
        assert "no parameter 'bonus'" in str(error)
    else:
        raise AssertionError("child-only arguments must require an isinstance guard")
    m = load(compiled, "_compiled_extended_override")
    b = m.B()
    holder = m.Holder(); holder.target = b
    holder.invoke(5)
    assert b.total == 9
    holder.target = m.A()
    holder.invoke(5)
    assert holder.target.total == 5
    b.hit(2, bonus=4)
    assert b.total == 15
    print("ok  isinstance narrows an object field for child-only override arguments")


def test_init_resolution():
    """__init__ is found through Python's own MRO; stray constructor args are rejected."""
    src = ("from hybrid_compiler import c\n"
           "@c\nclass A:\n    h: float\n    def __init__(self, h: float = 1.0):\n        self.h = h\n"
           "@c\nclass B(A):\n    k: float\n"
           "@c\nclass N:\n    n: float\n")
    m = load(HybridCompiler().compile(src), "_compiled_init")
    assert m.B(5.0).h == 5.0 and m.B().h == 1.0
    try:
        m.N(5); raise SystemExit("N(5) should raise TypeError")
    except TypeError:
        pass
    print("ok  __init__ resolved through the MRO; stray constructor args rejected")


def test_overridden_constant_and_field_names():
    """Class constants overridden in a subclass resolve by RUNTIME class (own self and through
    a pointer); field names like `size`/`data` work (they are not shadowed by numpy)."""
    src = ("from hybrid_compiler import c\n"
           "@c\nclass A:\n    X: float = 1.0\n    v: float\n    size: float\n    data: float\n"
           "    @c\n    def f(self) -> None:\n        self.v = self.X\n        self.size += 1.0\n        self.data = 7.0\n"
           "    @c\n    def g(self, o: A) -> None:\n        self.v = o.X\n"
           "@c\nclass B(A):\n    X: float = 2.0\n")
    m = load(HybridCompiler().compile(src), "_compiled_const")
    a, b = m.A(), m.B()
    a.f(); b.f()
    assert (a.v, b.v) == (1.0, 2.0) and b.size == 1.0 and b.data == 7.0
    a.g(b); assert a.v == 2.0
    b.g(a); assert b.v == 1.0
    print("ok  overridden class constant resolves by runtime class; fields named size/data work")


def expect_error(src, needle):
    try:
        HybridCompiler().compile(src)
    except CompileError as e:
        assert needle in str(e), f"wrong message: {e}"
        print(f"ok  rejected: {str(e)[:95]}")
        return
    raise SystemExit(f"expected CompileError containing {needle!r}")


def test_errors():
    base = "from hybrid_compiler import c\n"
    expect_error(base + "@c\nclass A:\n    state: float\n@c\nclass B:\n    state: int\n@c\nclass C(A, B):\n    pass\n",
                 "conflicting types")
    expect_error(base + "@c\nclass A:\n    pass\nclass B(A):\n    pass\n", "Undecorated class 'B'")
    expect_error(base + "@c\nclass A:\n    name: str\n    @c\n    def f(self) -> None:\n        self.name = 'x'\n",
                 "python-only")
    expect_error(base + "@c\nclass A:\n    xs: list[A]\n    @c\n    def f(self, o: A) -> None:\n        self.xs.append(o)\n",
                 "list")
    expect_error(base + "@c\nclass A:\n    r: A\n    @c\n    def f(self, o: A) -> None:\n        self.r = o\n",
                 "Python writes pointers")
    expect_error(base + "@c\nclass A:\n    @c\n    def f(self) -> None:\n        A()\n", "cannot construct")
    expect_error(base + "@c\nclass A:\n    def g(self):\n        pass\n    @c\n    def f(self) -> None:\n        self.g()\n",
                 "not a @c method")
    expect_error(base + "@c\nclass A:\n    @c\n    def f(self, x: float) -> None:\n        pass\n"
                        "@c\nclass B(A):\n    @c\n    def f(self, y: float) -> None:\n        pass\n", "different signature")
    expect_error(base + "@c\nclass A:\n    @c\n    def f(self, x: int) -> None:\n        pass\n"
                        "@c\nclass B(A):\n    @c\n    def f(self, x: int, y: int) -> None:\n        pass\n", "default values")
    expect_error(base + "@c\nclass A:\n    @c\n    def f(self, x: int) -> None:\n        pass\n"
                        "@c\nclass B(A):\n    @c\n    def f(self, x: float, y: int = 0) -> None:\n        pass\n", "different signature")
    expect_error(base + "@c\nclass A:\n    xs: list[float]\n", "np.ndarray[float, 1]")
    expect_error(base + "@c\nclass A:\n    @c\n    def f(self, x) -> None:\n        pass\n", "needs a type annotation")
    expect_error(base + "@c\nclass A:\n    r: A\n    r_type: int\n", "collides")


if __name__ == "__main__":
    print("DEBUG MODE (HYBRID_DEBUG=1): generated code runs as plain Python")
    test_behaviour_matches_python()
    test_generated_code_shape()
    test_layout_and_projections()
    test_projected_view_reads_child_memory()
    test_gc_lifetime_and_lists()
    test_pickle_and_deepcopy()
    test_subclass_without_new_fields()
    test_override_extends_parent_signature()
    test_init_resolution()
    test_overridden_constant_and_field_names()
    test_errors()
    print("\nALL TESTS PASSED")
