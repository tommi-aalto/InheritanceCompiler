This is a compiler/transpiler which takes Object oriented python code and transpiles it to "fat classes"
These classes don't have any inheritance and can therefore be jit-compiles using the @jitclass decorator

The python codebase to be compiled has to be strictly typed. Type annotations are not necessary, as the compiler will extract all object attribute assignments
and add them to the class annotations automatically. Method calls have to strictl adhere to the specified types

NOT SUPPORTED:
@jitclass or @njit in the base code, as this is to be added later
@staticmethod, or @classmethod, as numba does not support calling methods on classes
getattr(), setattr()

SUPPORTED:
Inheritance, multiple inheritance, diamond-style inheritance, and multiple
inheritance across otherwise separate root hierarchies
super(), isinstance(object, Class), issubclass (object, Class)

When a class combines separate inheritance trees, the compiler creates an
artificial squash root for those trees. Original class MROs are retained when
specializing `super()` calls, so calls stop at the original roots as before.

`compile_module(package_directory)` scans Python source files throughout the
package tree, combines them into one compilation unit, and writes
`_compiled_aot.py` and `_compiled_jit.py` at the package root. Relative imports
between those source files are removed because their definitions are combined.
Import compiled classes from the aggregate module, for example:

```python
from ._compiled_jit import MyClass
```
