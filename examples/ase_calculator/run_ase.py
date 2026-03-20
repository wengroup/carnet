from ase.io import read

from carnet.ext.ase.calculator import CarnetCalculator

# Modify `model_path` to point to a trained model checkpoint on your system
model_path = "<path_to_trained_carnet_model.ckpt>"

calc = CarnetCalculator(model_path, device="cpu", need_stress=False)

filename = "liquid-64.xyz"
atoms = read(filename)
atoms.calc = calc

pe = atoms.get_potential_energy()
print("Potential energy:", pe)

forces = atoms.get_forces()
print("Forces:")
for i, f in enumerate(forces):
    print(i, f)
