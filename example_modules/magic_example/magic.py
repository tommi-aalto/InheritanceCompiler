
# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

# 1. HIERARCHY ONE: Polymorphic Squashing & Argument Expansion

from hybrid_compiler import compiled
import numpy as np

@compiled
class Weapon:
    @compiled
    def attack(self, target: Entity, amount: int = 5) -> None:
       # print(f"Base weapon strikes {target}!")
        target.take_damage(amount)

@compiled
class Sword(Weapon):
    sharpness: int

    def __init__(self, sharpness: int):
        super().__init__()
        self.sharpness = sharpness
    @compiled
    def attack(self, target: Entity, amount: int = 5, bonus_damage: int = 5) -> None:
      #  print("Sword slashes !")
        super().attack(target, amount + bonus_damage + self.sharpness)
@compiled
class Bow(Weapon):
    arrows: np.ndarray[int, 1]

    def __init__(self, arrow_sizes: list[int]):
        self.arrows = np.zeros((len(arrow_sizes),))
        self.arrows[:] = arrow_sizes


    @compiled
    def attack(self, target: Entity, amount: int = 5) -> None:
        for i in range(self.arrows.size):
            target.take_damage(self.arrows[i])

# 2. HIERARCHY TWO: Diamond Inheritance & Dynamic Super() Cascades!
@compiled
class Entity:
    health: int
    def __init__(self):
        self.health = 100
        print("Entity initialized")
    @compiled
    def take_damage(self, amount: int) -> None:
        self.health -= amount
@compiled
class Physical(Entity):
    def __init__(self):
        super().__init__()
    @compiled
    def take_damage(self, amount: int) -> None:
      #  print("Physical Armor blocks 5")
        super().take_damage(max(0, amount - 5))
@compiled
class Magical(Entity):
    @compiled
    def take_damage(self, amount: int) -> None:
       # print("Magical Wards block 10")
        super().take_damage(max(0, amount - 10))
@compiled
class Paladin(Physical, Magical):
    weapon: Weapon  # Fat Struct polymorphic slot!
    
    def __init__(self, weapon: Weapon):
        super().__init__()
        self.weapon = weapon
    @compiled
    def take_damage(self, amount: int) -> None:
       # print("Paladin braces for impact!")
        super().take_damage(amount)
    @compiled
    def strike(self, target: Entity) -> int:
        # Executes V-Table dispatcher with expanded argument!
        if isinstance(self.weapon, Sword):
            return self.weapon.attack(target, bonus_damage = 20)
        return self.weapon.attack(target)
      #  print('total damage', dmg)