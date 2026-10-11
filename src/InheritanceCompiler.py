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
    """Rewrites `super().method(args)` -> `self.super_method(TYPE_CALLER, args)`"""
    def __init__(self, current_class_name: str):
        self.current_class_name = current_class_name

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and \
           isinstance(node.func.value, ast.Call) and \
           isinstance(node.func.value.func, ast.Name) and \
           node.func.value.func.id == 'super':
            
            method_name = node.func.attr
            
            # Rewrite to: self.super_{method_name}
            node.func = ast.Attribute(
                value=ast.Name(id='self', ctx=ast.Load()),
                attr=f"super_{method_name}",
                ctx=ast.Load()
            )
            
            # Inject TYPE_CURRENT_CLASS as the first argument (caller_class_id)
            caller_type = ast.Name(id=f"TYPE_{self.current_class_name.upper()}", ctx=ast.Load())
            node.args.insert(0, caller_type)
            
        return node


class InitSuperSpecializer(ast.NodeTransformer):
    def __init__(self, concrete_class, owner_class, mro, init_impls):
        self.concrete_class = concrete_class
        self.owner_class = owner_class
        self.mro = mro
        self.init_impls = init_impls

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'self'
            and node.func.attr == 'super___init__'
        ):
            return node

        next_owner = next(
            (
                ancestor
                for ancestor in self.mro[self.mro.index(self.owner_class) + 1:]
                if ancestor in self.init_impls
            ),
            None,
        )
        if next_owner is None:
            return ast.Constant(value=None)

        args = node.args[1:] if node.args else []
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='self', ctx=ast.Load()),
                attr=f"_{next_owner}_init_for_{self.concrete_class}",
                ctx=ast.Load(),
            ),
            args=args,
            keywords=node.keywords,
        )


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
            return ast.Constant(value=None)

        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id='self', ctx=ast.Load()),
                attr=f"_{next_owner}_{self.method_name}_for_{self.concrete_class}",
                ctx=ast.Load(),
            ),
            args=node.args[1:],
            keywords=node.keywords,
        )


class OptionalUnwrapper(ast.NodeTransformer):
    """Insert assertions that narrow Optional self attributes before use."""

    def __init__(self):
        self.optional_fields = set()

    def visit(self, node):
        if isinstance(node, ast.ClassDef):
            return self.visit_ClassDef(node)
        if isinstance(node, ast.stmt):
            return self._visit_statement(node)
        return super().visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
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

    def _visit_statement(self, node):
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
    """Computes MRO, builds the Ancestry Matrix, and assigns TYPE IDs."""
    def __init__(self):
        self.class_registry = {}
        self.class_ids = {}
        self.ancestry_matrix = []
        self.ordered_classes = []

    def _c3_merge(self, seqs):
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
        if cls_name not in self.class_registry: return [cls_name, 'object']
        cls_node = self.class_registry[cls_name]
        bases = [b.id for b in cls_node.bases if isinstance(b, ast.Name)]
        return [cls_name] + self._c3_merge([self._compute_mro(b) for b in bases] + [bases])

    def visit_Module(self, node: ast.Module) -> ast.Module:
        for stmt in node.body:
            if isinstance(stmt, ast.ClassDef):
                self.class_registry[stmt.name] = copy.deepcopy(stmt)
                self.ordered_classes.append(stmt.name)
                self.class_ids[stmt.name] = len(self.class_ids)
                
        for cls in self.ordered_classes:
            mro = self._compute_mro(cls)
            self.ancestry_matrix.append(tuple((c in mro) for c in self.ordered_classes))

        self.generic_visit(node)
        
        injected = [ast.Assign(targets=[ast.Name(id=f"TYPE_{name.upper()}", ctx=ast.Store())], 
                               value=ast.Constant(value=cid)) for name, cid in self.class_ids.items()]
        
        matrix_ast = ast.Tuple(elts=[ast.Tuple(elts=[ast.Constant(value=v) for v in row], ctx=ast.Load()) 
                                     for row in self.ancestry_matrix], ctx=ast.Load())
        injected.append(ast.Assign(targets=[ast.Name(id="ANCESTRY_TABLE", ctx=ast.Store())], value=matrix_ast))

        utils = """
def is_instance(obj, target_id: int) -> bool: return ANCESTRY_TABLE[obj.type_id][target_id]
def is_subclass(cls_id: int, target_id: int) -> bool: return ANCESTRY_TABLE[cls_id][target_id]
"""
        injected.extend(ast.parse(utils).body)
        node.body = injected + node.body
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node.bases = []
        return node


# ==============================================================================
# PASS 4: SQUASH COMPILER (Fat Structs, Properties & Padding)
# ==============================================================================

class SquashCompiler(ast.NodeTransformer):
    def __init__(self, flattening_compiler):
        self.fc = flattening_compiler
        self.bases_map = {name: [b.id for b in node.bases if isinstance(b, ast.Name)] 
                          for name, node in self.fc.class_registry.items()}
        self.mro_map = {name: self.fc._compute_mro(name) for name in self.fc.ordered_classes}
                # Replace the old self.root_map line with this:
        self.root_map = {}
        for name, mro in self.mro_map.items():
            for ancestor in reversed(mro):  # Start from oldest ancestor
                if ancestor in self.fc.class_registry:
                    self.root_map[name] = ancestor
                    break

        
        self.squash_groups = collections.defaultdict(list)
        for cls, root in self.root_map.items():
            self.squash_groups[root].append(cls)

    def _merge_arguments(self, signatures):
        merged = copy.deepcopy(signatures[0])
        merged.args = []
        defaults_by_name = {}
        present_in = collections.defaultdict(set)

        for signature_index, signature in enumerate(signatures):
            defaulted_args = signature.args[-len(signature.defaults):] if signature.defaults else []
            defaults_by_name.update(
                (arg.arg, default)
                for arg, default in zip(defaulted_args, signature.defaults)
                if arg.arg not in defaults_by_name
            )
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
    def _optional_annotation(cls, annotation):
        if annotation is None:
            return None
        if cls._is_optional_annotation(annotation) or cls._is_primitive_annotation(annotation):
            return annotation
        return ast.Subscript(
            value=ast.Name(id='Optional', ctx=ast.Load()),
            slice=copy.deepcopy(annotation),
            ctx=ast.Load(),
        )

    @staticmethod
    def _initial_value(annotation):
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

    @staticmethod
    def _init_method_name(owner, concrete_class):
        return f"_{owner}_init_for_{concrete_class}"

    def _build_init_dispatcher(self, best_args, all_classes, init_impls, property_registry):
        outer_else = [ast.Return(value=ast.Constant(value=None))]

        for concrete_class in reversed(all_classes):
            owner = next(
                (ancestor for ancestor in self.mro_map[concrete_class] if ancestor in init_impls),
                None,
            )
            if owner is None:
                continue
            init_args = init_impls[owner].args.args[1:]
            call_args = []
            for init_arg in init_args:
                passed_arg = ast.Name(id=init_arg.arg, ctx=ast.Load())
                field = property_registry.get(init_arg.arg)
                annotation = field.annotation if field is not None else init_arg.annotation
                if self._is_primitive_annotation(annotation):
                    passed_arg = ast.IfExp(
                        test=ast.Compare(
                            left=copy.deepcopy(passed_arg),
                            ops=[ast.IsNot()],
                            comparators=[ast.Constant(value=None)],
                        ),
                        body=copy.deepcopy(passed_arg),
                        orelse=self._initial_value(annotation),
                    )
                call_args.append(passed_arg)
            call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='self', ctx=ast.Load()),
                    attr=self._init_method_name(owner, concrete_class),
                    ctx=ast.Load(),
                ),
                args=call_args,
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

        return ast.FunctionDef(name='_dispatch_init', args=copy.deepcopy(best_args), body=outer_else, decorator_list=[])

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
        new_module_body = [
            ast.ImportFrom(module='typing', names=[ast.alias(name='Optional')], level=0)
        ]
        for stmt in node.body:
            if not isinstance(stmt, ast.ClassDef):
                new_module_body.append(stmt)
                continue
                
            cls_name = stmt.name
            root_name = self.root_map.get(cls_name)
            if cls_name != root_name: continue
                
            fat_struct = copy.deepcopy(stmt)
            all_classes = self.squash_groups[root_name]
            
            method_registry = collections.defaultdict(dict) 
            property_methods = set()
            property_registry = {'type_id': ast.AnnAssign(target=ast.Name(id='type_id', ctx=ast.Store()), 
                                                          annotation=ast.Name(id='int', ctx=ast.Load()), simple=1)}
            
            # 1. GATHER DATA & NAMESPACE
            for current_cls in all_classes:
                child_node = self.fc.class_registry[current_cls]
                for item in child_node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        property_node = copy.deepcopy(item)
                        property_node.annotation = self._optional_annotation(property_node.annotation)
                        property_registry[item.target.id] = property_node
                    elif isinstance(item, ast.FunctionDef):
                        is_getter = any(isinstance(d, ast.Name) and d.id == 'property' for d in item.decorator_list)
                        is_setter = any(isinstance(d, ast.Attribute) and d.attr == 'setter' for d in item.decorator_list)
                        
                        registry_key = f"{item.name}_SET" if is_setter else (f"{item.name}_GET" if is_getter else item.name)
                        if is_getter or is_setter:
                            property_methods.add(registry_key)
                            item.decorator_list = []
                            
                        rewritten = SuperCallRewriter(current_cls).visit(copy.deepcopy(item))
                        method_registry[registry_key][current_cls] = rewritten

            # 2. BUILD FAT STRUCT PROPERTIES
            fat_struct.body = list(property_registry.values())
            
            # 3. FAT __init__ GENERATION
            init_impls = method_registry.get('__init__', {})
            
            if init_impls:
                init_best_args = self._merge_arguments([m.args for m in init_impls.values()])
            else:
                init_best_args = ast.arguments(posonlyargs=[], args=[ast.arg(arg='self')], kwonlyargs=[], kw_defaults=[], defaults=[])

            fat_init_args = copy.deepcopy(init_best_args)
            fat_init_args.args.insert(1, ast.arg(arg='type_id', annotation=ast.Name(id='int', ctx=ast.Load())))

            init_body = [ast.Assign(targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='type_id', ctx=ast.Store())], 
                                    value=ast.Name(id='type_id', ctx=ast.Load()))]
            init_body.extend(
                ast.Assign(
                    targets=[ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr=prop_name, ctx=ast.Store())],
                    value=self._initial_value(property_registry[prop_name].annotation),
                )
                for prop_name in property_registry
                if prop_name != 'type_id'
            )

            if init_impls:
                expected_init_args = [a.arg for a in init_best_args.args[1:]]
                dispatch_call = ast.Call(func=ast.Attribute(value=ast.Name(id='self', ctx=ast.Load()), attr='_dispatch_init', ctx=ast.Load()),
                                         args=[ast.Name(id=name, ctx=ast.Load()) for name in expected_init_args], keywords=[])
                init_body.append(ast.Expr(value=dispatch_call))
            fat_struct.body.append(ast.FunctionDef(name='__init__', args=fat_init_args, body=init_body, decorator_list=[]))

            # 4. GENERATE V-TABLES
            for registry_key, impls in method_registry.items():
                method_name = registry_key.replace('_SET', '').replace('_GET', '')
                best_args = self._merge_arguments([m.args for m in impls.values()])
                is_getter = registry_key.endswith('_GET')
                is_setter = registry_key.endswith('_SET')

                if method_name == '__init__':
                    for concrete_class in all_classes:
                        mro = self.mro_map[concrete_class]
                        for owner in mro:
                            if owner not in impls:
                                continue
                            specialized = copy.deepcopy(impls[owner])
                            specialized.name = self._init_method_name(owner, concrete_class)
                            specialized = InitSuperSpecializer(
                                concrete_class,
                                owner,
                                mro,
                                impls,
                            ).visit(specialized)
                            fat_struct.body.append(specialized)

                    fat_struct.body.append(
                        self._build_init_dispatcher(best_args, all_classes, impls, property_registry)
                    )
                    continue

                suffix = "_set" if is_setter else ("_get" if is_getter else "")
                for concrete_class in all_classes:
                    mro = self.mro_map[concrete_class]
                    for owner in mro:
                        if owner not in impls:
                            continue
                        specialized = copy.deepcopy(impls[owner])
                        specialized.name = f"_{owner}_{method_name}{suffix}_for_{concrete_class}"
                        specialized = MethodSuperSpecializer(
                            concrete_class,
                            owner,
                            method_name,
                            mro,
                            impls,
                        ).visit(specialized)
                        fat_struct.body.append(specialized)

                fat_struct.body.append(
                    self._build_method_dispatcher(
                        method_name,
                        best_args,
                        all_classes,
                        impls,
                        is_getter,
                        is_setter,
                    )
                )

            new_module_body.append(fat_struct)
            
            # 5. GENERATE FACADES
            for child_cls in all_classes:
                if child_cls == root_name: continue
                
                facade_args = copy.deepcopy(init_best_args)
                facade_args.args = [a for a in facade_args.args[1:]]
                
                facade = ast.FunctionDef(
                    name=child_cls, args=facade_args,
                    body=[ast.Return(value=ast.Call(
                        func=ast.Name(id=root_name, ctx=ast.Load()),
                        args=[ast.Name(id=f"TYPE_{child_cls.upper()}", ctx=ast.Load())] + \
                             [ast.Name(id=a.arg, ctx=ast.Load()) for a in facade_args.args],
                        keywords=[]
                    ))], decorator_list=[]
                )
                new_module_body.append(facade)


        node.body = new_module_body
        return node

# ==============================================================================
# PASS 5: JIT INJECTION 
# ==============================================================================

class JITInjector(ast.NodeTransformer):
    def visit_Module(self, node: ast.Module) -> ast.Module:
        node.body = ast.parse("from numba.experimental import jitclass\nfrom numba import njit").body + node.body
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if node.name in ('is_instance', 'is_subclass') or node.name[0].isupper():
            node.decorator_list.append(ast.Name(id='njit', ctx=ast.Load()))
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        node.decorator_list.append(ast.Call(func=ast.Name(id='jitclass', ctx=ast.Load()), args=[], keywords=[]))
        return node

def transpile_code(source_code: str):
    """Executes the pipeline in memory and returns (AOT_code, JIT_code)"""
    tree = ast.parse(source_code)
    
    # Run AST Pipeline
    tree = IsInstanceTransformer().visit(tree)
    flattening_pass = FlatteningCompiler()
    tree = flattening_pass.visit(tree)
    tree = SquashCompiler(flattening_pass).visit(tree)
    ast.fix_missing_locations(tree)
    
    aot_code = ast.unparse(tree)
    
    jit_tree = OptionalUnwrapper().visit(copy.deepcopy(tree))
    jit_tree = JITInjector().visit(jit_tree)
    ast.fix_missing_locations(jit_tree)
    jit_code = ast.unparse(jit_tree)
    
    return aot_code, jit_code

def compile_module(directory_path: str):
    """
    Scans a directory for OOP Python files, hashes them to detect changes,
    and dynamically compiles them into _aot.py and _jit.py files.
    Designed to be called at the top of an __init__.py file.
    """
    print(f"[Engine] Scanning directory for OOP files: {directory_path}")
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")

    with open(__file__, 'rb') as compiler_file:
        compiler_hash = hashlib.md5(compiler_file.read()).digest()

    for dir in os.listdir(directory_path):
        dir_path = os.path.join(directory_path, dir)
        if os.path.isdir(dir_path) and dir != '__pycache__':
            print(f"[Engine] Recursively compiling submodule: {dir_path}")
            compile_module(dir_path)  # Recursive call for subdirectories

    for filename in os.listdir(directory_path):
        if not filename.endswith('.py'):
            continue
            
        # Skip special files and previously compiled outputs
        if filename == '__init__.py' or filename.endswith('_aot.py') or filename.endswith('_jit.py'):
            continue
            
        source_path = os.path.join(directory_path, filename)
        base_name = filename[:-3]
        
        aot_path = os.path.join(directory_path, f"{base_name}_aot.py")
        jit_path = os.path.join(directory_path, f"{base_name}_jit.py")
        hash_path = os.path.join(directory_path, f".{base_name}.hash")
        
        with open(source_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
            
        current_hash = hashlib.md5(source_code.encode('utf-8') + compiler_hash).hexdigest()
        
        # Check cache hit
        if os.path.exists(hash_path) and os.path.exists(aot_path) and os.path.exists(jit_path):
            with open(hash_path, 'r', encoding='utf-8') as f:
                if f.read().strip() == current_hash:
                    print(f"[Engine] Cache hit for {filename}, skipping compilation.")
                    continue  # Cache hit, completely skip compilation
                
        print(f"[Engine] Compiling {filename} to C-Structs...")
        
        try:
            aot_code, jit_code = transpile_code(source_code)
            
            with open(aot_path, 'w', encoding='utf-8') as f:
                f.write(aot_code)
                
            with open(jit_path, 'w', encoding='utf-8') as f:
                f.write(jit_code)
                
            # Lock the cache only if compilation succeeded
            with open(hash_path, 'w', encoding='utf-8') as f:
                f.write(current_hash)
                
        except Exception as e:
            print(f"[Engine] Compilation failed for {filename}: {e}")