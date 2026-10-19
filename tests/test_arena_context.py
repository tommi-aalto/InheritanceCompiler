import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from InheritanceCompiler import transpile_code


class ArenaContextTests(unittest.TestCase):
    def test_registry_context_and_facades(self):
        source = '''
class Node:
    health: int
    target: Node | None

    def __init__(self, health: int, target: Node | None = None):
        self.health = health
        self.target = target

    def heal(self, amount: int):
        self.health += amount
        return self.health

    def attack_target(self, amount: int):
        if self.target is not None:
            return self.target.heal(amount)
        return self.heal(amount)

    def attack_self(self, amount: int):
        return self.heal(amount)

class Holder:
    node: Node

    def __init__(self, node: Node):
        self.node = node

    def value(self):
        return self.node.health
'''
        _, jit_code = transpile_code(source)
        self.assertIn('target_id: types.int64', jit_code)
        self.assertIn('def attack_target(self, __registry, amount: int):', jit_code)
        self.assertNotIn('deferred_type', jit_code)
        namespace = {'__name__': 'arena_context_test'}
        exec(jit_code, namespace)

        target = namespace['Node'](health=10)
        attacker = namespace['Node'](health=1, target=target)
        registry = namespace['__registry']

        self.assertEqual(attacker.attack_target(registry, 5), 15)
        self.assertEqual(target.health, 15)
        self.assertEqual(attacker.attack_self(registry, 4), 5)
        self.assertEqual(attacker.health, 5)
        self.assertEqual(namespace['Node'](health=0).target_id, -1)

        holder = namespace['Holder'](node=target)
        self.assertEqual(holder.value(registry), 15)

    def test_lists_of_user_classes_are_stored_as_ids(self):
        source = '''
from typing import List

class Item:
    value: int

    def __init__(self, value: int):
        self.value = value

class Inventory:
    items: List[Item]

    def __init__(self, items: List[Item]):
        self.items = items

    def first_value(self):
        return self.items[0].value

    def total(self):
        result = 0
        for item in self.items:
            result += item.value
        return result

    def append_item(self, item: Item):
        self.items.append(item)
'''
        _, jit_code = transpile_code(source)
        self.assertIn('items_ids: List[types.int64]', jit_code)
        self.assertIn('NumbaList.empty_list(types.int64)', jit_code)
        namespace = {'__name__': 'arena_list_test'}
        exec(jit_code, namespace)

        first = namespace['Item'](value=4)
        second = namespace['Item'](value=6)
        inventory = namespace['Inventory'](items=[first, second])
        registry = namespace['__registry']

        self.assertEqual(inventory.first_value(registry), 4)
        self.assertEqual(inventory.total(registry), 10)
        inventory.append_item(registry, namespace['Item'](value=9))
        self.assertEqual(inventory.total(registry), 19)

    def test_inline_object_annotation_is_registered_and_rewritten(self):
        source = '''
class Eye:
    value: int

    def __init__(self, value: int):
        self.value = value

class Owner:
    valid: bool

    def __init__(self, eye: Eye):
        self.eye: Eye = eye
        self.valid = isinstance(eye, Eye)

    def read(self):
        return self.eye.value
'''
        _, jit_code = transpile_code(source)
        self.assertIn('eye_id: types.int64', jit_code)
        self.assertNotIn('self.eye: Eye', jit_code)
        self.assertIn('__registry[0][eye]', jit_code)
        namespace = {'__name__': 'arena_inline_annotation_test'}
        exec(jit_code, namespace)

        eye = namespace['Eye'](value=12)
        owner = namespace['Owner'](eye=eye)
        registry = namespace['__registry']
        self.assertTrue(owner.valid)
        self.assertEqual(owner.read(registry), 12)

    def test_optional_object_lists_are_rejected_explicitly(self):
        source = '''
from typing import List, Optional

class Item:
    value: int

    def __init__(self, value: int):
        self.value = value

class Inventory:
    items: Optional[List[Item]]

    def __init__(self, items: Optional[List[Item]] = None):
        self.items = items
'''
        with self.assertRaisesRegex(ValueError, 'Optional lists of user-defined classes'):
            transpile_code(source)


if __name__ == '__main__':
    unittest.main()
