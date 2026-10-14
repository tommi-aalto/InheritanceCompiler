from numba.experimental import jitclass
from numba import njit
TYPE_WEAPON = 0
TYPE_SWORD = 1
TYPE_BOW = 2
TYPE_ENTITY = 3
TYPE_PHYSICAL = 4
TYPE_MAGICAL = 5
TYPE_PALADIN = 6
ANCESTRY_TABLE = ((True, False, False, False, False, False, False), (True, True, False, False, False, False, False), (True, False, True, False, False, False, False), (False, False, False, True, False, False, False), (False, False, False, True, True, False, False), (False, False, False, True, False, True, False), (False, False, False, True, True, True, True))

@njit
def is_instance(obj, target_id: int) -> bool:
    return ANCESTRY_TABLE[obj.type_id][target_id]

@njit
def is_subclass(cls_id: int, target_id: int) -> bool:
    return ANCESTRY_TABLE[cls_id][target_id]

@jitclass()
class Weapon:
    type_id: int
    sharpness: int

    def _Weapon___init___for_Weapon(self, message: str):
        print(message)

    def _Sword___init___for_Sword(self, message: str, sharpness: int):
        self._Weapon___init___for_Sword(message)
        self.sharpness = sharpness

    def _Weapon___init___for_Sword(self, message: str):
        print(message)

    def _Weapon___init___for_Bow(self, message: str):
        print(message)

    def __init__(self, type_id: int, message: str, sharpness: int=None):
        self.type_id = type_id
        self.sharpness = 0
        if self.type_id == TYPE_WEAPON:
            return self._Weapon___init___for_Weapon(message if message is not None else '')
        elif self.type_id == TYPE_SWORD:
            return self._Sword___init___for_Sword(message if message is not None else '', sharpness if sharpness is not None else 0)
        elif self.type_id == TYPE_BOW:
            return self._Weapon___init___for_Bow(message if message is not None else '')
        else:
            return None

    def _Weapon_attack_for_Weapon(self, target: str) -> int:
        print(f'Base weapon strikes {target}!')
        return 5

    def _Sword_attack_for_Sword(self, target: str, bonus_damage: int=0) -> int:
        print('Sword slashes !')
        base_dmg = self._Weapon_attack_for_Sword(target)
        return base_dmg + self.sharpness + bonus_damage

    def _Weapon_attack_for_Sword(self, target: str) -> int:
        print(f'Base weapon strikes {target}!')
        return 5

    def _Bow_attack_for_Bow(self, target: str) -> int:
        print('Bow fires at ', target, '!')
        return 15

    def _Weapon_attack_for_Bow(self, target: str) -> int:
        print(f'Base weapon strikes {target}!')
        return 5

    def attack(self, target: str, bonus_damage: int=0):
        if self.type_id == TYPE_WEAPON:
            return self._Weapon_attack_for_Weapon(target if target is not None else '')
        elif self.type_id == TYPE_SWORD:
            return self._Sword_attack_for_Sword(target if target is not None else '', bonus_damage if bonus_damage is not None else 0)
        elif self.type_id == TYPE_BOW:
            return self._Bow_attack_for_Bow(target if target is not None else '')
        else:
            return None

def Sword(message: str, sharpness: int):
    return Weapon(TYPE_SWORD, message, sharpness)

def Bow(message: str):
    return Weapon(TYPE_BOW, message, sharpness=None)

@jitclass()
class Entity:
    type_id: int
    weapon: Weapon | None

    def _Entity___init___for_Entity(self):
        print('Entity initialized')

    def _Physical___init___for_Physical(self):
        self._Entity___init___for_Physical()

    def _Entity___init___for_Physical(self):
        print('Entity initialized')

    def _Entity___init___for_Magical(self):
        print('Entity initialized')

    def _Paladin___init___for_Paladin(self, weapon: Weapon):
        self._Physical___init___for_Paladin()
        self.weapon = weapon

    def _Physical___init___for_Paladin(self):
        self._Entity___init___for_Paladin()

    def _Entity___init___for_Paladin(self):
        print('Entity initialized')

    def __init__(self, type_id: int, weapon: Weapon=None):
        self.type_id = type_id
        self.weapon = None
        if self.type_id == TYPE_ENTITY:
            return self._Entity___init___for_Entity()
        elif self.type_id == TYPE_PHYSICAL:
            return self._Physical___init___for_Physical()
        elif self.type_id == TYPE_MAGICAL:
            return self._Entity___init___for_Magical()
        elif self.type_id == TYPE_PALADIN:
            assert weapon is not None
            return self._Paladin___init___for_Paladin(weapon)
        else:
            return None

    def _Entity_take_damage_for_Entity(self, amount: int) -> None:
        pass

    def _Physical_take_damage_for_Physical(self, amount: int) -> None:
        print('Physical Armor blocks 5')
        self._Entity_take_damage_for_Physical(amount - 5)

    def _Entity_take_damage_for_Physical(self, amount: int) -> None:
        pass

    def _Magical_take_damage_for_Magical(self, amount: int) -> None:
        print('Magical Wards block 10')
        self._Entity_take_damage_for_Magical(amount - 10)

    def _Entity_take_damage_for_Magical(self, amount: int) -> None:
        pass

    def _Paladin_take_damage_for_Paladin(self, amount: int) -> None:
        print('Paladin braces for impact!')
        self._Physical_take_damage_for_Paladin(amount)

    def _Physical_take_damage_for_Paladin(self, amount: int) -> None:
        print('Physical Armor blocks 5')
        self._Magical_take_damage_for_Paladin(amount - 5)

    def _Magical_take_damage_for_Paladin(self, amount: int) -> None:
        print('Magical Wards block 10')
        self._Entity_take_damage_for_Paladin(amount - 10)

    def _Entity_take_damage_for_Paladin(self, amount: int) -> None:
        pass

    def take_damage(self, amount: int):
        if self.type_id == TYPE_ENTITY:
            return self._Entity_take_damage_for_Entity(amount if amount is not None else 0)
        elif self.type_id == TYPE_PHYSICAL:
            return self._Physical_take_damage_for_Physical(amount if amount is not None else 0)
        elif self.type_id == TYPE_MAGICAL:
            return self._Magical_take_damage_for_Magical(amount if amount is not None else 0)
        elif self.type_id == TYPE_PALADIN:
            return self._Paladin_take_damage_for_Paladin(amount if amount is not None else 0)
        else:
            return None

    def _Paladin_strike_for_Paladin(self):
        assert self.weapon is not None
        dmg = self.weapon.attack('Dragon', bonus_damage=20)
        print('total damage', dmg)

    def strike(self):
        if self.type_id == TYPE_PALADIN:
            return self._Paladin_strike_for_Paladin()
        else:
            return None

def Physical():
    return Entity(TYPE_PHYSICAL, weapon=None)

def Magical():
    return Entity(TYPE_MAGICAL, weapon=None)

def Paladin(weapon: Weapon):
    return Entity(TYPE_PALADIN, weapon)