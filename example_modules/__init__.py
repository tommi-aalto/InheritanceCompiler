from src.hybrid_compiler_manager import HybridCompilerManager

HybridCompilerManager(__file__, __name__, source_dir="magic_example").build()

from .compiled import Paladin, Sword, Entity, Magical, Bow