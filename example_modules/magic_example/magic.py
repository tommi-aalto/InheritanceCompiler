
# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

# 1. HIERARCHY ONE: Polymorphic Squashing & Argument Expansion

from hybrid_compiler import c
@c
class Weapon:
    @c
    def attack(self, target: Entity, amount: int = 5) -> None:
        print(f"Base weapon strikes {target}!")
        target.take_damage(amount)

@c
class Sword(Weapon):
    sharpness: int
    def __init__(self, sharpness: int):
        super().__init__()
        self.sharpness = sharpness
    @c
    def attack(self, target: str, amount: int = 5,  bonus_damage: int = 0) -> None:
        print("Sword slashes !")
        super().attack(target, amount + bonus_damage + self.sharpness)
@c
class Bow(Weapon):
    @c
    def attack(self, target: str) -> None:
        print("Bow fires at ", target, "!")
        target.take_damage(15)

# 2. HIERARCHY TWO: Diamond Inheritance & Dynamic Super() Cascades!
@c
class Entity:
    def __init__(self):
        self.health = 100
        print("Entity initialized")
    @c
    def take_damage(self, amount: int) -> None:
        self.health -= amount
@c
class Physical(Entity):
    def __init__(self):
        super().__init__()
    @c
    def take_damage(self, amount: int) -> None:
        print("Physical Armor blocks 5")
        super().take_damage(amount - 5)
@c
class Magical(Entity):
    @c
    def take_damage(self, amount: int) -> None:
        print("Magical Wards block 10")
        super().take_damage(amount - 10)
@c
class Paladin(Physical, Magical):
    weapon: Weapon  # Fat Struct polymorphic slot!
    
    def __init__(self, weapon: Weapon):
        super().__init__()
        self.weapon = weapon
    @c
    def take_damage(self, amount: int) -> None:
        print("Paladin braces for impact!")
        super().take_damage(amount)
    @c
    def strike(self, target: Entity):
        # Executes V-Table dispatcher with expanded argument!
        dmg = self.weapon.attack(target, bonus_damage=20)
        print('total damage', dmg)

if __name__ == '__main__':
    target = Magical()
    weapon = Sword(sharpness=10)
    paladin = Paladin(weapon)
    paladin.strike(target)