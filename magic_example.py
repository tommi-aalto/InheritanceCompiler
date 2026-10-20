from example_modules import Paladin, Sword, Magical

target = Magical()
weapon = Sword(sharpness=10)
paladin = Paladin(weapon)
paladin.strike(target)
print(target.health)