"""
numba_smoke_test.py -- run this on a machine WITH Numba (I could not run it where I wrote it).

It checks, one by one, the Numba behaviours hybrid_compiler's generated code relies on.
Each check prints PASS/FAIL, so you see exactly which assumption (if any) breaks on your version.
"""
import numpy as np
from numba import carray, njit, types
from numba.extending import intrinsic
from llvmlite import ir

results = []


def check(name, fn):
    try:
        ok = bool(fn())
    except Exception as e:                       # noqa: BLE001  (we want to see any failure)
        print(f"FAIL  {name}: {type(e).__name__}: {str(e).splitlines()[0][:150]}")
        results.append(False)
        return
    print(("PASS  " if ok else "FAIL  ") + name)
    results.append(ok)


@intrinsic
def _to_voidptr(typingctx, addr):
    def codegen(context, builder, signature, args):
        return builder.inttoptr(args[0], ir.IntType(8).as_pointer())
    return types.voidptr(types.intp), codegen


# a row with a hole in it (offset 16 unused) and a "view" that only sees some fields
full = np.dtype({'names': ['type_id', 'a', 'b'], 'formats': [np.int64, np.float64, np.float64],
                 'offsets': [0, 8, 24], 'itemsize': 32}, align=True)
view = np.dtype({'names': ['type_id', 'b'], 'formats': [np.int64, np.float64],
                 'offsets': [0, 24], 'itemsize': 32}, align=True)
mem = np.zeros(4, dtype=np.int64)
row = np.frombuffer(mem, dtype=full, count=1)
addr = mem.ctypes.data


@njit
def bump(self, self_addr):                       # `self` is a record, as in generated methods
    self.a += 1.0
    self.b += 2.0


def t1():                                        # np.void from Python: write-through?
    bump(row[0], addr)
    return row["a"][0] == 1.0 and row["b"][0] == 2.0


@njit
def inner(self, self_addr):
    self.b += 100.0


@njit
def via_ptr(a):                                  # int -> voidptr -> carray over a RECORD dtype -> record -> other njit fn
    p = _to_voidptr(a)
    inner(carray(p, 1, view)[0], a)
    carray(p, 1, view)[0].b += 10.0


def t2():
    via_ptr(addr)
    return row["b"][0] == 2.0 + 100.0 + 10.0 and row["a"][0] == 1.0       # `a` is not in the view: untouched


@njit
def two_views(t, a):                             # per-branch access (different record types never share a variable)
    p = _to_voidptr(a)
    if t == 1:
        return carray(p, 1, view)[0].b
    if t == 2:
        return carray(p, 1, full)[0].b
    raise ValueError('bad type')


def t3():
    return two_views(1, addr) == two_views(2, addr) == row["b"][0]


@njit
def read_shape(self):                            # record sub-array field via attribute
    return self.sh[1]


sub = np.dtype({'names': ['type_id', 'sh'], 'formats': [np.int64, (np.int64, (2,))],
                'offsets': [0, 8], 'itemsize': 24}, align=True)
mem2 = np.zeros(3, dtype=np.int64)
row2 = np.frombuffer(mem2, dtype=sub, count=1)
row2["sh"][0, :] = [5, 7]


def t4():
    return read_shape(row2[0]) == 7


@njit
def arr_from_fields(self):                       # array rebuilt from (addr, shape) record fields, as _arr_*_Nd helpers do
    return carray(_to_voidptr(self.d), (self.s[0], self.s[1]), np.int64)


store = np.arange(6, dtype=np.int64).reshape(3, 2)
arec = np.dtype({'names': ['type_id', 'd', 's'], 'formats': [np.int64, np.int64, (np.int64, (2,))],
                 'offsets': [0, 8, 16], 'itemsize': 32}, align=True)
mem3 = np.zeros(4, dtype=np.int64)
row3 = np.frombuffer(mem3, dtype=arec, count=1)
row3["d"][0] = store.ctypes.data
row3["s"][0, :] = store.shape


def t5():
    return np.array_equal(arr_from_fields(row3[0]), store)


check("1. np.void passed from Python, `self.a += 1` writes through to the array", t1)
check("2. carray(voidptr, 1, record dtype)[0] -> record passed to another njit fn, writes land", t2)
check("3. per-branch records of two different dtypes (the dispatcher pattern)", t3)
check("4. record sub-array field read by attribute", t4)
check("5. numpy array rebuilt from (addr, shape) record fields", t5)
print("\nALL PASSED" if all(results) else "\nSOME CHECKS FAILED -- tell me which and I will adapt the generated code")
