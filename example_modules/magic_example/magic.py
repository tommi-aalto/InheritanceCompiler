
# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

# 1. HIERARCHY ONE: Polymorphic Squashing & Argument Expansion
class Weapon:
    def attack(self, target: str) -> int:
        print(f"Base weapon strikes {target}!")
        return 5

class Sword(Weapon):
    sharpness: int
    def __init__(self, sharpness):
        super().__init__()
        self.sharpness = sharpness
    def attack(self, target: str, bonus_damage: int = 0) -> int:
        print("Sword slashes !")
        base_dmg = super().attack(target)
        return base_dmg + self.sharpness + bonus_damage

class Bow(Weapon):
    range_dist: int
    def __init__(self, range_dist):
        super().__init__()
        self.range_dist = range_dist
    def attack(self, target: str) -> int:
        print("Bow fires!")
        return 15

# 2. HIERARCHY TWO: Diamond Inheritance & Dynamic Super() Cascades!
class Entity:
    def __init__(self):
        print("Entity initialized")
    def take_damage(self, amount: int) -> None:
        pass

class Physical(Entity):
    def __init__(self):
        super().__init__()
    def take_damage(self, amount: int) -> None:
        print("Physical Armor blocks 5")
        super().take_damage(amount - 5)

class Magical(Entity):
    def __init__(self):
        super().__init__()
    def take_damage(self, amount: int) -> None:
        print("Magical Wards block 10")
        super().take_damage(amount - 10)

class Paladin(Physical, Magical):
    weapon: Weapon  # Fat Struct polymorphic slot!

    def __init__(self, weapon: Weapon):
        super().__init__()
        self.weapon = weapon
    
    def take_damage(self, amount: int) -> None:
        print("Paladin braces for impact!")
        super().take_damage(amount)

    def strike(self):
        # Executes V-Table dispatcher with expanded argument!
        dmg = self.weapon.attack("Dragon", bonus_damage=20)
        print('total damage', dmg)