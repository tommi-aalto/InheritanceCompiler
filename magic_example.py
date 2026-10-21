from example_modules import Paladin, Bow, Magical

target = Magical()
weapon = Bow(arrow_sizes=[13,14,15,16,17])
paladin = Paladin(weapon)
paladin.strike(target)
print(target.health)