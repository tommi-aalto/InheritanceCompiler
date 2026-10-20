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
An override may append parameters after the parent's parameters when each appended
parameter has a default. Parent parameter names, types, and order must be preserved.

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

In the JIT output, fields annotated with a user-defined class or a list of
user-defined classes are stored as `int64` instance IDs (`field_id` or
`field_ids`) and resolved through a typed registry. Constructor facades still
accept instances and lists of instances; compiled methods transparently resolve
those references and thread the registry through internal calls. When calling
a compiled jitclass method directly from Python, pass the module's `__registry`
as the first argument after the instance. Optional lists (or optional entries
within these lists) are not supported; use an empty list instead of `None`.
