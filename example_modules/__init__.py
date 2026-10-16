import os
#import typing
from src.InheritanceCompiler import compile_module
from pathlib import Path
# 1. Trigger the dynamic compiler BEFORE importing the classes
_current_dir = Path(__file__).parent
compile_module(_current_dir)

# # 2. Route the imports for IDE vs Runtime
# if typing.TYPE_CHECKING:
#     # IDE Intelligence (reads your original OOP source)
#     from .magic_example.magic import Entity, Player
# else:
#     # Runtime Execution (reads the auto-generated JIT file)
from ._compiled_jit import Entity, Physical, Magical, Paladin, Weapon, Sword, Bow
from ._compiled_jit import is_instance as jit_isinstance, is_instance as jit_issubclass
__all__ = ["Entity", "Physical", "Magical", "Paladin", "Weapon", "Sword", "Bow", jit_isinstance, jit_issubclass]