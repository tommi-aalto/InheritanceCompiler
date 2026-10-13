This is a compiler/transpiler which takes Object oriented python code and transpiles it to "fat classes"
These classes don't have any inheritance and can therefore be jit-compiles using the @jitclass decorator

The python codebase to be compiled has to be strictly typed. Type annotations are not necessary, as the compiler will extract all object attribute assignments
and add them to the class annotations automatically. Method calls have to strictl adhere to the specified types

NOT SUPPORTED:
@jitclass or @njit in the base code, as this is to be added later
@staticmethod, or @classmethod, as numba does not support calling methods on classes
getattr(), setattr()

SUPPORTED:
Inheritance, multiple inheritance, diamond-style inheritance
super(), isinstance(object, Class), issubclass (object, Class)
