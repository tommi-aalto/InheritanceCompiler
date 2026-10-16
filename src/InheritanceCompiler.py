import ast
import copy
import collections
import os
import hashlib

# ==============================================================================
# PASS 1: TYPE CHECK INTERCEPTION
# ==============================================================================

class IsInstanceTransformer(ast.NodeTransformer):
    """Intercepts `isinstance(obj, Class)` -> `is_instance(obj, TYPE_CLASS)`"""
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        # Find any function call of isinstance(object, Class) or issubclass(Class, Class)
        # and replace it with is_instance or is_subclass(object, TYPE_CLASS)
        # These are custom made methods looking up the result in an inheritance table
        if isinstance(node.func, ast.Name) and node.func.id in ('isinstance', 'issubclass'):
            if len(node.args) == 2 and isinstance(node.args[1], ast.Name):
                target = node.args[1].id
                new_func = 'is_instance' if node.func.id == 'isinstance' else 'is_subclass'
                return ast.Call(
                    func=ast.Name(id=new_func, ctx=ast.Load()),
                    args=[node.args[0], ast.Name(id=f"TYPE_{target.upper()}", ctx=ast.Load())],
                    keywords=[]
                )
        return node


# ==============================================================================
# PASS 2: SUPER() REWRITER
# ==============================================================================

class SuperCallRewriter(ast.NodeTransformer):
    """
    Rewrites `super().method(args)` -> `self.super_method(TYPE_CALLER, args)`
    Python will insert self to make it self.super_method(self, TYPE_CALLER, args)
    the super_method will look up the mro to determine who is the parent
    based on the type of self and the class the super is called from
    """
    def __init__(self, current_class_name: str):
        self.current_class_name = current_class_name

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        # Find all super() calls
        if isinstance(node.func, ast.Attribute) and \
           isinstance(node.func.value, ast.Call) and \
           isinstance(node.func.value.func, ast.Name) and \
           node.func.value.func.id == 'super':
            
            method_name = node.func.attr

            # Take the function which was called on super() and rewrite it to be a method of self
            # with this name schema: self.super_{method_name}
            node.func = ast.Attribute(
                value=ast.Name(id='self', ctx=ast.Load()),
                attr=f"super_{method_name}",
                ctx=ast.Load()
            )
            
            # Inject TYPE_CURRENT_CLASS as the first argument (caller_class_id)
            caller_type = ast.Name(id=f"TYPE_{self.current_class_name.upper()}", ctx=ast.Load())
            node.args.insert(0, caller_type)
            
        return node


class MethodSuperSpecializer(ast.NodeTransformer):
    def __init__(self, concrete_class, owner_class, method_name, mro, method_impls):
        self.concrete_class = concrete_class
        self.owner_class = owner_class
        self.method_name = method_name
        self.mro = mro
        self.method_impls = method_impls

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'self'
            and node.func.attr == f'super_{self.method_name}'
        ):
            return node

        next_owner = next(
            (
                ancestor
                for ancestor in self.mro[self.mro.index(self.owner_class) + 1:]
                if ancestor in self.method_impls
            ),
            None,
        )
        if next_owner is None:
            return ast.Constant(value=Ellipsis)

        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='self', ctx=ast.Load()),
                attr=f"_{next_owner}_{self.method_name}_for_{self.concrete_class}",
                ctx=ast.Load(),
            ),
            args=node.args[1:],
            keywords=node.keywords,
        )


class ClassAnnotationNormalizer(ast.NodeTransformer):
    def __init__(self, root_map: dict[str, str]):
        self.root_map = root_map

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = self.root_map.get(node.id, node.id)
        return node


class ForwardClassReferences(ast.NodeTransformer):
    def __init__(self, class_names: set[str]):
        self.class_names = class_names

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.class_names:
            return ast.Constant(value=node.id)
        return node


class OptionalAnnotationNormalizer(ast.NodeTransformer):
    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        node.annotation = self.visit(node.annotation)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        if node.annotation is not None:
            node.annotation = self.visit(node.annotation)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node.args)
        if node.returns is not None:
            node.returns = self.visit(node.returns)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        if isinstance(node.op, ast.BitOr):
            if isinstance(node.left, ast.Constant) and node.left.value is None:
                return ast.Subscript(
                    value=ast.Name(id='Optional', ctx=ast.Load()),
                    slice=self.visit(node.right),
                    ctx=ast.Load(),
                )
            if isinstance(node.right, ast.Constant) and node.right.value is None:
                return ast.Subscript(
                    value=ast.Name(id='Optional', ctx=ast.Load()),
                    slice=self.visit(node.left),
                    ctx=ast.Load(),
                )
        return self.generic_visit(node)


class AOTForwardAnnotation(ast.NodeTransformer):
    def __init__(self):
        self.class_names = set()

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.class_names = {
            statement.name
            for statement in node.body
            if isinstance(statement, ast.ClassDef)
        }
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        normalizer = ForwardClassReferences(self.class_names)
        for item in node.body:
            if isinstance(item, ast.AnnAssign):
                item.annotation = normalizer.visit(
                    OptionalAnnotationNormalizer().visit(item.annotation)
                )
            elif isinstance(item, ast.FunctionDef):
                for argument in (
                    item.args.posonlyargs
                    + item.args.args
                    + item.args.kwonlyargs
                    + ([item.args.vararg] if item.args.vararg else [])
                    + ([item.args.kwarg] if item.args.kwarg else [])
                ):
                    if argument.annotation is not None:
                        argument.annotation = normalizer.visit(
                            OptionalAnnotationNormalizer().visit(argument.annotation)
                        )
                if item.returns is not None:
                    item.returns = normalizer.visit(
                        OptionalAnnotationNormalizer().visit(item.returns)
                    )
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        normalizer = ForwardClassReferences(self.class_names)
        for argument in (
            node.args.posonlyargs
            + node.args.args
            + node.args.kwonlyargs
            + ([node.args.vararg] if node.args.vararg else [])
            + ([node.args.kwarg] if node.args.kwarg else [])
        ):
            if argument.annotation is not None:
                argument.annotation = normalizer.visit(
                    OptionalAnnotationNormalizer().visit(argument.annotation)
                )
        if node.returns is not None:
            node.returns = normalizer.visit(
                OptionalAnnotationNormalizer().visit(node.returns)
            )
        node.body = [self.visit(statement) for statement in node.body]
        return node


class OptionalUnwrapper(ast.NodeTransformer):
    """
    As we use Optional[Type] for all complex types in the fat struct,
    we need to make sure that any method which uses these attributes
    """

    def __init__(self):
        self.optional_fields = set()

    def visit(self, node):
        if isinstance(node, ast.ClassDef):
            return self.visit_ClassDef(node)
        if isinstance(node, ast.FunctionDef):
            return self.visit_FunctionDef(node)
        if isinstance(node, ast.stmt):
            return self._visit_statement(node)
        return super().visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        '''
        Collect all optional fields in the class and store them in self.optional_fields
        '''
        previous_fields = self.optional_fields
        self.optional_fields = {
            item.target.id
            for item in node.body
            if isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and SquashCompiler._is_optional_annotation(item.annotation)
        }
        node = self.generic_visit(node)
        self.optional_fields = previous_fields
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if not node.name == '__init__':
            return self.generic_visit(node)
        init_body = copy.deepcopy(node.body)
        new_init_body = []

        # Helper function to recursively process lists of statements (handles nested ifs)
        def transform_statements(stmts):
            new_stmts = []
            for stmt in stmts:
                
                # 1. If it's an 'if' statement, recursively process its inner body and orelse blocks
                if isinstance(stmt, ast.If):
                    stmt.body = transform_statements(stmt.body)
                    stmt.orelse = transform_statements(stmt.orelse)
                    new_stmts.append(stmt)
                    continue
                
                # 2. Check if the statement (Expr or Return) holds a function/method call
                call_node = None
                if isinstance(stmt, (ast.Expr, ast.Return)) and isinstance(stmt.value, ast.Call):
                    call_node = stmt.value
                
                # 3. If a call is found, inject assertions BEFORE the statement
                if call_node:
                    for arg in call_node.args:
                        if isinstance(arg, ast.Name):
                            arg_name = arg.id
                            new_stmts.append(
                                ast.Assert(
                                    test=ast.Compare(
                                        left=ast.Name(id=arg_name, ctx=ast.Load()),
                                        ops=[ast.IsNot()],
                                        comparators=[ast.Constant(value=None)],
                                    ),
                                    msg=None,
                                )
                            )
                
                # Append the original statement itself
                new_stmts.append(stmt)
            return new_stmts

        node.body = transform_statements(node.body)
        return self.generic_visit(node)

    def _visit_statement(self, node):
        '''
        insert assert self.attr is not None before any method call which uses self.attr
        '''
        loaded_attributes = []
        method_attributes = set()
        optional_fields = self.optional_fields

        class AttributeCollector(ast.NodeVisitor):
            def visit(self, current):
                if current is not node and isinstance(current, ast.stmt):
                    return
                return super().visit(current)

            def visit_Call(self, call):
                if isinstance(call.func, ast.Attribute) and \
                   isinstance(call.func.value, ast.Name) and \
                   call.func.value.id == 'self':
                    method_attributes.add(id(call.func))
                self.generic_visit(call)

            def visit_Attribute(self, attribute):
                if id(attribute) in method_attributes:
                    return
                if isinstance(attribute.ctx, ast.Load) and \
                   isinstance(attribute.value, ast.Name) and \
                   attribute.value.id == 'self' and \
                   attribute.attr != 'type_id' and \
                   attribute.attr in optional_fields and \
                   attribute.attr not in loaded_attributes:
                    loaded_attributes.append(attribute.attr)
                self.generic_visit(attribute)

        # find all method calls which use attribues which are optional and will be used in this method
        # so we can assert they are not None before use
        collector = AttributeCollector()
        collector.visit(node)

        for field_name, value in ast.iter_fields(node):
            if isinstance(value, list):
                transformed = []
                for child in value:
                    if isinstance(child, ast.AST):
                        result = self.visit(child)
                        transformed.extend(result if isinstance(result, list) else [result])
                    else:
                        transformed.append(child)
                setattr(node, field_name, transformed)
            elif isinstance(value, ast.AST):
                setattr(node, field_name, self.visit(value))

        # inject the assert self.attr is Not None before using it in a method call
        # So that the method does not need to take care of Optional[Type]
        # but can use Type directly, and numba will not complain about type mismatch
        assertions = [
            ast.Assert(
                test=ast.Compare(
                    left=ast.Attribute(
                        value=ast.Name(id='self', ctx=ast.Load()),
                        attr=attribute_name,
                        ctx=ast.Load(),
                    ),
                    ops=[ast.IsNot()],
                    comparators=[ast.Constant(value=None)],
                ),
                msg=None,
            )
            for attribute_name in loaded_attributes
        ]
        return assertions + [node]


# ==============================================================================
# PASS 3: FLATTENING COMPILER
# ==============================================================================

class FlatteningCompiler(ast.NodeTransformer):
    """
    This class takes all inheritance out of the child classes by manually injecting all
    inherited parent methods into the child class
    """
    def __init__(self):
        self.class_registry = {}
        self.class_ids = {}
        self.ancestry_matrix = []
        self.ordered_classes = []
        self.artificial_root_groups = {}

    def _c3_merge(self, seqs):
        """
        C3 Linearization algorithm for computing the MRO of inherited python classes
        """
        res = []
        while True:
            seqs = [s for s in seqs if s]
            if not seqs: return res
            for seq in seqs:
                head = seq[0]
                if not any(head in s[1:] for s in seqs): break
            else: raise Exception("Inconsistent MRO")
            res.append(head)
            for seq in seqs:
                if seq[0] == head: del seq[0]

    def _compute_mro(self, cls_name: str) -> list:
        # In case we did not find the class in the registry, we assume it is a base clas
        # (like object) and return it as the only element in the MRO
        if cls_name not in self.class_registry: return [cls_name, 'object']

        # Retrieve the Class node based on its name
        cls_node = self.class_registry[cls_name]

        #Get the classes it directly inherits from
        bases = [b.id for b in cls_node.bases if isinstance(b, ast.Name)]

        # MRO is the class itself + the merge of the MROs of its bases + the list of bases
        return [cls_name] + self._c3_merge([self._compute_mro(b) for b in bases] + [bases])

    def _add_artificial_roots(self) -> list[ast.ClassDef]:
        def tree_root(cls_name: str) -> str | None:
            for ancestor in reversed(self._compute_mro(cls_name)):
                if ancestor in self.class_registry:
                    return ancestor
            return None

        parent_roots = {}
        all_roots = set()
        for cls_name, cls_node in self.class_registry.items():
            roots = {
                root
                for base in cls_node.bases
                if isinstance(base, ast.Name)
                and (root := tree_root(base.id)) is not None
            }
            parent_roots[cls_name] = roots
            all_roots.update(roots)

        parent = {root: root for root in all_roots}

        def find(root: str) -> str:
            while parent[root] != root:
                parent[root] = parent[parent[root]]
                root = parent[root]
            return root

        def union(first: str, second: str) -> None:
            first_root = find(first)
            second_root = find(second)
            if first_root != second_root:
                parent[second_root] = first_root

        crossover_roots = set()
        for roots in parent_roots.values():
            if len(roots) > 1:
                roots_in_order = [
                    name for name in self.ordered_classes if name in roots
                ]
                crossover_roots.update(roots_in_order)
                for root in roots_in_order[1:]:
                    union(roots_in_order[0], root)

        components = collections.defaultdict(list)
        for root in self.ordered_classes:
            if root in crossover_roots:
                components[find(root)].append(root)

        created = []
        reserved_names = set(self.class_registry)
        for roots in components.values():
            if len(roots) < 2:
                continue
            artificial_name = "_ArtificialRoot_" + "_".join(roots)
            suffix = 2
            while artificial_name in reserved_names:
                artificial_name = "_ArtificialRoot_" + "_".join(roots) + f"_{suffix}"
                suffix += 1
            reserved_names.add(artificial_name)

            artificial_root = ast.ClassDef(
                name=artificial_name,
                bases=[ast.Name(id=root, ctx=ast.Load()) for root in roots],
                keywords=[],
                body=[ast.Pass()],
                decorator_list=[],
            )
            self.class_registry[artificial_name] = copy.deepcopy(artificial_root)
            self.class_ids[artificial_name] = len(self.class_ids)
            self.ordered_classes.append(artificial_name)
            self.artificial_root_groups[artificial_name] = roots
            created.append(artificial_root)
        return created

    def visit_Module(self, node: ast.Module) -> ast.Module:
        for stmt in node.body:
            # look for all classes in the module
            if isinstance(stmt, ast.ClassDef):
                self.class_registry[stmt.name] = copy.deepcopy(stmt)
                self.ordered_classes.append(stmt.name)
                # and assign class ids based on the order they were found
                self.class_ids[stmt.name] = len(self.class_ids)

        artificial_roots = self._add_artificial_roots()

        # Compute the MRO for each class and build the ancestry matrix
        for cls in self.ordered_classes:
            mro = self._compute_mro(cls)
            self.ancestry_matrix.append(tuple((c in mro) for c in self.ordered_classes))

        node.body.extend(artificial_roots)
        self.generic_visit(node)

        # Generate the module level injected code

        # Generate TYPE_CLASS constants for each class
        injected = [ast.Assign(targets=[ast.Name(id=f"TYPE_{name.upper()}", ctx=ast.Store())], 
                               value=ast.Constant(value=cid)) for name, cid in self.class_ids.items()]
        
        # Generate the tuple of tuples representing the ancestry matrix
        # It will look like this:
        # ((True, False, False), (False, True, False), (False, False, True))
        # And should be indexed using the TYPE_CLASS constants to determine if a class is a subclass of another
        matrix_ast = ast.Tuple(elts=[ast.Tuple(elts=[ast.Constant(value=v) for v in row], ctx=ast.Load()) 
                                     for row in self.ancestry_matrix], ctx=ast.Load())
        injected.append(ast.Assign(targets=[ast.Name(id="ANCESTRY_TABLE", ctx=ast.Store())], value=matrix_ast))


        # Generate the is_instance and is_subclass lookups
        utils = """
def is_instance(obj, target_id: int) -> bool: return ANCESTRY_TABLE[obj.type_id][target_id]
def is_subclass(cls_id: int, target_id: int) -> bool: return ANCESTRY_TABLE[cls_id][target_id]
"""
        # Inject at module level
        injected.extend(ast.parse(utils).body)
        node.body = injected + node.body
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        # Remove all bases from the class definition, since we are flattening the hierarchy
        # and already put all parent functionality into the children
        node.bases = []
        return node


# ==============================================================================
# PASS 4: SQUASH COMPILER (Fat Structs, Properties & Padding)
# ==============================================================================

class SquashCompiler(ast.NodeTransformer):
    """
    To allow for child-to-parent compability (Object owners accepting children instead of their parents)
    This class will squash the entire inheritance hierarchy into a single "fat struct" class,
    which will contain all the fields and methods of all the classes in the hierarchy.
    """
    def __init__(self, flattening_compiler: FlatteningCompiler):
        self.fc: FlatteningCompiler = flattening_compiler
        # We need to know the inheritance extracted from the flattening compiler
        # As the flattening compiler already stripped the inheritance from the classes
        self.bases_map = {name: [b.id for b in node.bases if isinstance(b, ast.Name)] 
                          for name, node in self.fc.class_registry.items()}
        self.mro_map = {name: self.fc._compute_mro(name) for name in self.fc.ordered_classes}
        # and map the root of each hierarchy to the classes that belong to it:
        self.root_map = {}
        for name, mro in self.mro_map.items():
            for ancestor in reversed(mro):  # Start from oldest ancestor
                if ancestor in self.fc.class_registry:
                    self.root_map[name] = ancestor
                    break

        for artificial_root, roots in self.fc.artificial_root_groups.items():
            for name, mro in self.mro_map.items():
                if name == artificial_root or any(root in mro for root in roots):
                    self.root_map[name] = artificial_root

        # group all children to be squashed into the root of the inheritance tree
        self.squash_groups = collections.defaultdict(list)
        for cls, root in self.root_map.items():
            self.squash_groups[root].append(cls)

    def _merge_arguments(self, signatures):
        """
        There will be a rule for the user when overwriting parent methods:
        accept all arguments the parent accepted in this method first, with the same order
        afterwards, new arguments can be appended

        the root of the tree will have to accept every argument that any child class ever accepts
        So this method will merge all the arguments of the same method across the hierarchy into a single signature
        """
        merged = copy.deepcopy(signatures[0])
        merged.args = []
        defaults_by_name = {}
        present_in = collections.defaultdict(set)

        # Loop through all signatures and collect the arguments and defaults
        for signature_index, signature in enumerate(signatures):
            # The default value will be overwritten by the first signature that has it, and the rest will be ignored
            defaulted_args = signature.args[-len(signature.defaults):] if signature.defaults else []
            defaults_by_name.update(
                (arg.arg, default)
                for arg, default in zip(defaulted_args, signature.defaults)
                if arg.arg not in defaults_by_name
            )
            # Append all found args to the merged.args if not already present, and track which signature they were found in
            for arg in signature.args:
                present_in[arg.arg].add(signature_index)
                if all(existing.arg != arg.arg for existing in merged.args):
                    merged.args.append(copy.deepcopy(arg))

        first_default = next(
            (
                index
                for index, arg in enumerate(merged.args)
                if arg.arg in defaults_by_name
                or len(present_in[arg.arg]) != len(signatures)
            ),
            len(merged.args),
        )
        merged.defaults = [
            copy.deepcopy(
                defaults_by_name.get(arg.arg, ast.Constant(value=None))
                if arg.arg in defaults_by_name
                or len(present_in[arg.arg]) != len(signatures)
                else ast.Constant(value=None)
            )
            for arg in merged.args[first_default:]
        ]
        return merged

    @staticmethod
    def _is_optional_annotation(annotation):
        # like this -> Optional[int] or this -> int | None
        if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
            return any(isinstance(side, ast.Constant) and side.value is None for side in (annotation.left, annotation.right))
        if isinstance(annotation, ast.Subscript):
            optional_name = annotation.value
            return (isinstance(optional_name, ast.Name) and optional_name.id == 'Optional') or \
                   (isinstance(optional_name, ast.Attribute) and optional_name.attr == 'Optional')
        return False

    @staticmethod
    def _is_primitive_annotation(annotation):
        return isinstance(annotation, ast.Name) and annotation.id in {
            'bool', 'bytes', 'complex', 'float', 'int', 'str',
        }

    @classmethod
    def _optional_annotation(cls, annotation: ast.AST | None):
        if annotation is None:
            return None
        # Only make optional if not already optional or assigning a primitive type
        if cls._is_optional_annotation(annotation) or cls._is_primitive_annotation(annotation):
            return annotation
        return ast.Subscript(
            value=ast.Name(id='Optional', ctx=ast.Load()),
            slice=copy.deepcopy(annotation),
            ctx=ast.Load(),
        )

    @staticmethod
    def _initial_value(annotation):
        """
        primitives are initialized to 0, '', False, etc.
        complex types as None
        """
        if isinstance(annotation, ast.Name):
            primitive_defaults = {
                'bool': False,
                'bytes': b'',
                'complex': 0j,
                'float': 0.0,
                'int': 0,
                'str': '',
            }
            if annotation.id in primitive_defaults:
                return ast.Constant(value=primitive_defaults[annotation.id])
        return ast.Constant(value=None)

    def _call_args_for_signature(self, signature, available_args, property_registry=None):
        argument_values = {arg.arg: ast.Name(id=arg.arg, ctx=ast.Load()) for arg in available_args.args[1:]}
        defaults = {}
        if signature.defaults:
            defaulted_args = signature.args[-len(signature.defaults):]
            defaults.update((arg.arg, default) for arg, default in zip(defaulted_args, signature.defaults))

        call_args = []
        for arg in signature.args[1:]:
            value = argument_values[arg.arg]
            default = defaults.get(arg.arg)
            field = property_registry.get(arg.arg) if property_registry is not None else None
            annotation = field.annotation if field is not None else arg.annotation

            if default is not None:
                fallback = copy.deepcopy(default)
            elif self._is_primitive_annotation(annotation):
                fallback = self._initial_value(annotation)
            else:
                call_args.append(value)
                continue

            call_args.append(
                ast.IfExp(
                    test=ast.Compare(
                        left=copy.deepcopy(value),
                        ops=[ast.IsNot()],
                        comparators=[ast.Constant(value=None)],
                    ),
                    body=value,
                    orelse=fallback,
                )
            )
        return call_args

    def _build_method_dispatcher(self, method_name, best_args, all_classes, impls, is_getter, is_setter):
        """
        Make an if-elif chain, looking for the correct type_id and calling the respective method
        """
        outer_else = [ast.Return(value=ast.Constant(value=None))]
        suffix = "_set" if is_setter else ("_get" if is_getter else "")

        for concrete_class in reversed(all_classes):
            owner = next(
                (ancestor for ancestor in self.mro_map[concrete_class] if ancestor in impls),
                None,
            )
            if owner is None:
                continue
            call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='self', ctx=ast.Load()),
                    attr=f"_{owner}_{method_name}{suffix}_for_{concrete_class}",
                    ctx=ast.Load(),
                ),
                args=self._call_args_for_signature(impls[owner].args, best_args),
                keywords=[],
            )
            outer_else = [
                ast.If(
                    test=ast.Compare(
                        left=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='type_id', ctx=ast.Load()),
                        ops=[ast.Eq()],
                        comparators=[ast.Name(id=f"TYPE_{concrete_class.upper()}", ctx=ast.Load())],
                    ),
                    body=[ast.Return(value=call)],
                    orelse=outer_else,
                )
            ]

        decorator_list = []
        if is_getter:
            decorator_list.append(ast.Name(id='property', ctx=ast.Load()))
        elif is_setter:
            decorator_list.append(
                ast.Attribute(value=ast.Name(id=method_name, ctx=ast.Load()), attr='setter', ctx=ast.Load())
            )
        return ast.FunctionDef(
            name=method_name,
            args=copy.deepcopy(best_args),
            body=outer_else,
            decorator_list=decorator_list,
        )

    def visit_Module(self, node: ast.Module) -> ast.Module:
        '''
        Squash all inheritance trees found into a singe fat-struct like class
        '''
        new_module_body = []
        for stmt in node.body:
            if not isinstance(stmt, ast.ClassDef):
                new_module_body.append(stmt)
                continue

            # Just look at the roots of the inheritance trees, we will find the children in this pass
            cls_name = stmt.name
            root_name = self.root_map.get(cls_name)
            if cls_name != root_name: continue

            # Make a new class based on this root class, which will be the fat struct
            fat_struct = copy.deepcopy(stmt)
            # Squash all child classes in to the root
            all_classes = self.squash_groups[root_name]
            
            method_registry = collections.defaultdict(dict) 
            property_methods = set()
            property_registry = {'type_id': ast.AnnAssign(target=ast.Name(id='type_id', ctx=ast.Store()), 
                                                          annotation=ast.Name(id='int', ctx=ast.Load()), simple=1)}
            

            # 1. GATHER DATA & NAMESPACE
            for current_cls in all_classes:
                # This is the child class were looking at
                child_node = self.fc.class_registry[current_cls]
                for item in child_node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        # We found an annotation assignment
                        property_node = copy.deepcopy(item)
                        property_node.annotation = self._normalize_annotation(property_node.annotation)
                        # primitives are taken directly; complex types can be initialized with None
                        property_node.annotation = self._optional_annotation(property_node.annotation)
                        property_registry[item.target.id] = property_node
                    elif isinstance(item, ast.FunctionDef):
                        # Parse all functions
                        # identify properties and setters by their decorators
                        is_getter = any(isinstance(d, ast.Name) and d.id == 'property' for d in item.decorator_list)
                        is_setter = any(isinstance(d, ast.Attribute) and d.attr == 'setter' for d in item.decorator_list)

                        # mark them accordingly
                        registry_key = f"{item.name}_SET" if is_setter else (f"{item.name}_GET" if is_getter else item.name)
                        if is_getter or is_setter:
                            property_methods.add(registry_key)
                            item.decorator_list = []

                        # Rewrite super() calls to be self.super_method(TYPE_CALLER, args)
                        rewritten = SuperCallRewriter(current_cls).visit(copy.deepcopy(item))
                        for argument in (
                            rewritten.args.posonlyargs
                            + rewritten.args.args
                            + rewritten.args.kwonlyargs
                            + ([rewritten.args.vararg] if rewritten.args.vararg else [])
                            + ([rewritten.args.kwarg] if rewritten.args.kwarg else [])
                        ):
                            argument.annotation = self._normalize_annotation(argument.annotation)
                        rewritten.returns = self._normalize_annotation(rewritten.returns)
                        method_registry[registry_key][current_cls] = rewritten

            # 2. BUILD FAT STRUCT PROPERTIES (not the @property, but the annotations)
            fat_struct.body = list(property_registry.values())
            
            # 4. GENERATE V-TABLES
            for registry_key, impls in method_registry.items():
                # for properties, we annotated the names so strip them to get the pure name
                method_name = registry_key.replace('_SET', '').replace('_GET', '')
                best_args = self._merge_arguments([m.args for m in impls.values()])
                # but we need to remember if its a property getter or setter
                is_getter = registry_key.endswith('_GET')
                is_setter = registry_key.endswith('_SET')
                

                suffix = "_set" if is_setter else ("_get" if is_getter else "")
                for concrete_class in all_classes:
                    mro = self.mro_map[concrete_class]
                    for owner in mro:
                        if owner not in impls:
                            continue
                        specialized = copy.deepcopy(impls[owner])
                        # rewrite all methods to "_Parent_method_for_Child"
                        # So that they can be explicitly called (after rewriting super by the SuperSpecializer)
                        specialized.name = f"_{owner}_{method_name}{suffix}_for_{concrete_class}"
                        specialized = MethodSuperSpecializer(
                            concrete_class,
                            owner,
                            method_name,
                            mro,
                            impls,
                        ).visit(specialized)
                        fat_struct.body.append(specialized)

                # the original method name will be a dispatcher calling the correct "Class method for Class" method
                dispatcher:ast.FunctionDef = self._build_method_dispatcher(
                    method_name,
                    best_args,
                    all_classes,
                    impls,
                    is_getter,
                    is_setter,
                )
                #The init just needs one additional argument "type_id" and to initialize the type id
                #and all annotations with dummy values
                if method_name == '__init__':
                    # insert the new type_id argument
                    dispatcher.args.args.insert(1, ast.arg(arg='type_id', annotation=ast.Name(id='int', ctx=ast.Load())))
                    # and in the __init__ body, assign it to self.type_id
                    fat_init_args = copy.deepcopy(dispatcher.args)
                    dispatcher.body.insert(0,ast.Assign(targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='type_id', ctx=ast.Store())], 
                                            value=ast.Name(id='type_id', ctx=ast.Load())))
                    # Initialize all other properties with their default values (0, '', None)
                    # Taken care of in _initial_value
                    # But except type_id
                    for prop_name in property_registry:
                        if prop_name == 'type_id':
                            continue
                        dispatcher.body.insert(1, ast.Assign(
                                                    targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr=prop_name, ctx=ast.Store())],
                                                    value=self._initial_value(property_registry[prop_name].annotation),
                                                ))
                
                fat_struct.body.append(dispatcher)
            
            new_module_body.append(fat_struct)
            
            # 5. GENERATE FACADES
            # Make methods for child classes initializing the root like this
            # def Child(arg1, arg2):
            # #     return Parent(type_id=TYPE_CHILD, arg1=arg1, arg2=arg2)
            for child_cls in all_classes:
                # the root does not need one, this is still a class with a proper __init__
                if child_cls == root_name: continue

                facade_args = ast.arguments()
                #Take the original child class init arguments
                for i in range(0, len(self.mro_map[child_cls])):
                    try:
                        facade_args = copy.deepcopy(
                            method_registry['__init__'][self.mro_map[child_cls][i]].args
                        )
                        break
                    except KeyError:
                        #child has no init, take the parents init
                        continue
                # if we found an init, strip self from the arguments, else empty arguments
                facade_args.args = [a for a in facade_args.args[1:]]  if len(facade_args.args) > 0 else []
                default_args = []
                for arg in fat_init_args.args:
                    if not any(arg.arg == facade_arg.arg for facade_arg in facade_args.args)\
                        and arg.arg not in ('type_id', 'self'):
                        default_args.append(
                            ast.keyword(
                            arg=arg.arg,              # The name of the parameter
                            value=ast.Constant(value=None)       # The value assigned to it (e.g., None)
                        ))
                
                facade = ast.FunctionDef(
                    name=child_cls, args=facade_args,
                    body=[ast.Return(value=ast.Call(
                        func=ast.Name(id=root_name, ctx=ast.Load()),
                        args=[ast.Name(id=f"TYPE_{child_cls.upper()}", ctx=ast.Load())],
                        keywords=[
                            ast.keyword(arg=arg.arg, value=ast.Name(id=arg.arg, ctx=ast.Load()))
                            for arg in facade_args.args
                        ] + default_args,
                    ))], decorator_list=[]
                )
                new_module_body.append(facade)


        node.body = new_module_body
        return node

    def _normalize_annotation(self, annotation: ast.AST | None) -> ast.AST | None:
        if annotation is None:
            return None
        return ClassAnnotationNormalizer(self.root_map).visit(copy.deepcopy(annotation))

# ==============================================================================
# PASS 5: JIT INJECTION 
# ==============================================================================

class JITInjector(ast.NodeTransformer):
    '''
    inject the @jitclass or @njit decorators into the generated code, so that the JIT version can be compiled by numba
    '''
    def __init__(self):
        self.compiled_class_names = set()

    def visit_Module(self, node: ast.Module) -> ast.Module:
        _insert_module_imports(
            node,
            "from typing import Optional\nfrom numba.experimental import jitclass\nfrom numba import njit, deferred_type, optional",
        )
        self.compiled_class_names = {
            statement.name
            for statement in node.body
            if isinstance(statement, ast.ClassDef)
        }
        import_index = 0
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            import_index = 1
        while import_index < len(node.body) and isinstance(
            node.body[import_index], (ast.Import, ast.ImportFrom)
        ):
            import_index += 1
        declarations = [
            ast.Assign(
                targets=[ast.Name(id=self._deferred_name(class_name), ctx=ast.Store())],
                value=ast.Call(func=ast.Name(id='deferred_type', ctx=ast.Load()), args=[], keywords=[]),
            )
            for class_name in sorted(self.compiled_class_names)
        ]
        node.body[import_index:import_index] = declarations
        self.generic_visit(node)
        return node

    @staticmethod
    def _deferred_name(class_name: str) -> str:
        return f"_DEFERRED_TYPE_{class_name.upper()}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if node.name in ('is_instance', 'is_subclass'):
            node.decorator_list.append(ast.Name(id='njit', ctx=ast.Load()))
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> list[ast.stmt]:
        for item in node.body:
            if not isinstance(item, ast.AnnAssign):
                continue
            annotation = item.annotation
            optional_annotation = (
                isinstance(annotation, ast.Subscript)
                and isinstance(annotation.value, ast.Name)
                and annotation.value.id == 'Optional'
            )
            annotation_type = annotation.slice if optional_annotation else annotation
            if (
                isinstance(annotation_type, ast.Name)
                and annotation_type.id in self.compiled_class_names
            ):
                deferred_type = ast.Name(
                    id=self._deferred_name(annotation_type.id),
                    ctx=ast.Load(),
                )
                item.annotation = (
                    ast.Call(
                        func=ast.Name(id='optional', ctx=ast.Load()),
                        args=[deferred_type],
                        keywords=[],
                    )
                    if optional_annotation
                    else deferred_type
                )

        class_annotation_normalizer = ForwardClassReferences(self.compiled_class_names)
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                for argument in (
                    item.args.posonlyargs
                    + item.args.args
                    + item.args.kwonlyargs
                    + ([item.args.vararg] if item.args.vararg else [])
                    + ([item.args.kwarg] if item.args.kwarg else [])
                ):
                    if argument.annotation is not None:
                        argument.annotation = class_annotation_normalizer.visit(argument.annotation)
                if item.returns is not None:
                    item.returns = class_annotation_normalizer.visit(item.returns)

        node.decorator_list.append(ast.Call(func=ast.Name(id='jitclass', ctx=ast.Load()), args=[], keywords=[]))
        return [
            node,
            ast.Expr(
                value=ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=self._deferred_name(node.name), ctx=ast.Load()),
                        attr='define',
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Attribute(
                            value=ast.Attribute(
                                value=ast.Name(id=node.name, ctx=ast.Load()),
                                attr='class_type',
                                ctx=ast.Load(),
                            ),
                            attr='instance_type',
                            ctx=ast.Load(),
                        )
                    ],
                    keywords=[],
                )
            ),
        ]


def _insert_module_imports(node: ast.Module, import_source: str) -> None:
    import_index = 0
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        import_index = 1
    while (
        import_index < len(node.body)
        and isinstance(node.body[import_index], ast.ImportFrom)
        and node.body[import_index].module == '__future__'
    ):
        import_index += 1
    node.body[import_index:import_index] = ast.parse(import_source).body


class CompiledImportRewriter(ast.NodeTransformer):
    def __init__(self, suffix: str):
        self.suffix = suffix

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
        if node.level > 0 and node.module:
            module_parts = node.module.rsplit('.', 1)
            module_parts[-1] += self.suffix
            node.module = '.'.join(module_parts)
        return node


class RelativeImportRemover(ast.NodeTransformer):
    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.level > 0:
            return None
        return node


def _combine_module_sources(module_sources: list[tuple[str, str]]) -> str:
    parsed_modules = []
    module_indexes = {}
    for filename, source_code in module_sources:
        module_name = os.path.splitext(filename)[0].replace(os.sep, '.')
        module_indexes[module_name] = len(parsed_modules)
        parsed_modules.append((module_name, filename, ast.parse(source_code, filename=filename)))

    dependencies = [set() for _ in parsed_modules]
    for index, (module_name, _, module) in enumerate(parsed_modules):
        package_parts = module_name.split('.')[:-1]
        for statement in ast.walk(module):
            if not isinstance(statement, ast.ImportFrom) or statement.level == 0:
                continue
            package_length = len(package_parts) - statement.level + 1
            if package_length < 0:
                continue
            target_parts = package_parts[:package_length]
            if statement.module:
                target_parts.extend(statement.module.split('.'))
                target = '.'.join(target_parts)
                if target in module_indexes:
                    dependencies[index].add(module_indexes[target])
            else:
                for imported_name in statement.names:
                    target = '.'.join(target_parts + [imported_name.name])
                    if target in module_indexes:
                        dependencies[index].add(module_indexes[target])

    ordered_indexes = []
    remaining_indexes = set(range(len(parsed_modules)))
    while remaining_indexes:
        ready_indexes = sorted(
            index
            for index in remaining_indexes
            if dependencies[index].isdisjoint(remaining_indexes)
        )
        if not ready_indexes:
            ready_indexes = sorted(remaining_indexes)
        ordered_indexes.extend(ready_indexes)
        remaining_indexes.difference_update(ready_indexes)

    body = []
    future_imports = []
    module_docstring = None
    for index in ordered_indexes:
        _, _, module = parsed_modules[index]
        module = copy.deepcopy(module)
        if module.body and isinstance(module.body[0], ast.Expr) and isinstance(module.body[0].value, ast.Constant) and isinstance(module.body[0].value.value, str):
            if module_docstring is None:
                module_docstring = module.body[0]
            module.body.pop(0)
        module = RelativeImportRemover().visit(module)
        for statement in module.body:
            if isinstance(statement, ast.ImportFrom) and statement.module == '__future__':
                future_imports.append(statement)
            else:
                body.append(statement)

    combined_body = ([module_docstring] if module_docstring is not None else []) + future_imports + body
    return ast.unparse(ast.Module(body=combined_body, type_ignores=[]))


def transpile_code(source_code: str):
    """Executes the pipeline in memory and returns (AOT_code, JIT_code)"""
    tree = ast.parse(source_code)
    
    # Run AST Pipeline
    tree = IsInstanceTransformer().visit(tree)
    flattening_pass = FlatteningCompiler()
    tree = flattening_pass.visit(tree)
    #ast.fix_missing_locations(tree)
    #flat = ast.unparse(tree)
    tree = SquashCompiler(flattening_pass).visit(tree)
    ast.fix_missing_locations(tree)
    
    aot_tree = AOTForwardAnnotation().visit(copy.deepcopy(tree))
    aot_tree = CompiledImportRewriter('_aot').visit(aot_tree)
    _insert_module_imports(aot_tree, "from typing import Optional")
    aot_code = ast.unparse(aot_tree)

    #Jit methods need an additional treatments to assert that instance attributes are not none before being used 
    #by assert self.attr is not None. Then numba will not treat is as Optional(Type) but as Type and static typing will work
    jit_tree = OptionalUnwrapper().visit(copy.deepcopy(tree))
    jit_tree = OptionalAnnotationNormalizer().visit(jit_tree)
    jit_tree = CompiledImportRewriter('_jit').visit(jit_tree)
    jit_tree = JITInjector().visit(jit_tree)
    ast.fix_missing_locations(jit_tree)
    jit_code = ast.unparse(jit_tree)
    
    return aot_code, jit_code#, flat

def compile_module(directory_path: str):
    """
    Scans a package tree for OOP Python files, combines them, and dynamically
    compiles the combined source into _compiled_aot.py and _compiled_jit.py.
    Designed to be called at the top of an __init__.py file.
    """
    print(f"[Engine] Scanning directory for OOP files: {directory_path}")
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")

    with open(__file__, 'rb') as compiler_file:
        compiler_hash = hashlib.md5(compiler_file.read()).digest()

    source_paths = []
    for current_directory, subdirectories, filenames in os.walk(directory_path):
        subdirectories[:] = sorted(
            name for name in subdirectories if name != '__pycache__'
        )
        for filename in sorted(filenames):
            if (
                not filename.endswith('.py')
                or filename == '__init__.py'
                or filename.endswith(('_aot.py', '_jit.py'))
                or filename in ('_compiled_aot.py', '_compiled_jit.py')
            ):
                continue
            source_paths.append(os.path.join(current_directory, filename))

    if not source_paths:
        print("[Engine] No source modules found, skipping compilation.")
        return

    module_sources = []
    source_hash = hashlib.md5(compiler_hash)
    for source_path in source_paths:
        with open(source_path, 'r', encoding='utf-8') as source_file:
            source_code = source_file.read()
        relative_path = os.path.relpath(source_path, directory_path)
        source_hash.update(relative_path.encode('utf-8'))
        source_hash.update(b'\0')
        source_hash.update(source_code.encode('utf-8'))
        source_hash.update(b'\0')
        module_sources.append((relative_path, source_code))

    aot_path = os.path.join(directory_path, '_compiled_aot.py')
    jit_path = os.path.join(directory_path, '_compiled_jit.py')
    hash_path = os.path.join(directory_path, '.compiled.hash')
    current_hash = source_hash.hexdigest()

    if os.path.exists(hash_path) and os.path.exists(aot_path) and os.path.exists(jit_path):
        with open(hash_path, 'r', encoding='utf-8') as hash_file:
            if hash_file.read().strip() == current_hash:
                print("[Engine] Cache hit for combined modules, skipping compilation.")
                return

    print(f"[Engine] Compiling {len(module_sources)} source modules into combined output...")

    try:
        combined_source = _combine_module_sources(module_sources)
        aot_code, jit_code = transpile_code(combined_source)

        with open(aot_path, 'w', encoding='utf-8') as aot_file:
            aot_file.write(aot_code)

        with open(jit_path, 'w', encoding='utf-8') as jit_file:
            jit_file.write(jit_code)

        with open(hash_path, 'w', encoding='utf-8') as hash_file:
            hash_file.write(current_hash)

    except Exception as error:
        print(f"[Engine] Combined compilation failed: {error}")
        raise