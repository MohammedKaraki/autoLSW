import numpy as np
from fractions import Fraction
from crystal import Crystal
from coords import Coords
from group import Group, GroupElement
from sublats import Sublat, Sublats, MagSublat, MagSublats, MagCrystal


# Specify space group by its generators
group = Group.from_gen_strs(
        "y,-x-z,z",
        "x+z,y+z,-z",
        "-x-z,y,z",
        "-y-z,-x-z,z"
    )



# Specify magnetic atoms in primitive cell
sublats = Sublats([
    Sublat("A", Coords(Fraction(-1, 4), Fraction(1, 4), Fraction(1, 2)), "S"),
    Sublat("B", Coords(Fraction(1, 4), Fraction(3, 4), Fraction(1, 2)), "S"),
    ])


# Specify orthonormal x, y & z axes, in terms of primitive Bravais translations.
# (Used for spin & E-field & strain directions)
xyz_axes = [
    r"{1, 0, 0}",
    r"{0, 1, 0}",
    r"{-1, -1, 2}"
    ]

crystal = Crystal(group, sublats, xyz_axes)




# Specify interactions here.
# Use `H_rep="vector"` for electric field-induced interactions,
# and `H_rep="tensor"` for strain-induced interactions.

crystal.add_bond(sublats["A"].coords + Coords(0, 0, 0),
                 sublats["A"].coords + Coords(0, 0, 0),
                 label="0",
                 H_rep="trivial"
                 )

crystal.add_bond(sublats["A"].coords + Coords(0, 0, 0),
                 sublats["B"].coords + Coords(0, 0, 0),
                 label="1",
                 H_rep="trivial"
                 )

crystal.add_bond(sublats["A"].coords + Coords(0, 0, 0),
                 sublats["A"].coords + Coords(1, 0, 0),
                 label="2",
                 H_rep="trivial"
                 )


crystal.add_bond(sublats["A"].coords + Coords(0, 0, 0),
                 sublats["B"].coords + Coords(-1, -1, 1),
                 label="5",
                 H_rep="trivial"
                 )



# Generate k-space representation of symmetry element
# crystal.add_rep_of_group_element(
#     GroupElement.from_str(r"y+z,-x,-z"),
#     label="S4")


# Generate k-space spin Hamiltonian
crystal.gen_mathematica_code()

# M transforms crystalline unit cell to magnetic unit cell
M = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]], dtype=int)

# Label atoms in magnetic unit cell and specify their moment directions
magsublats = MagSublats([
    MagSublat("A", "A1", sublats["A"].coords, "{0,0,-1}"),
    MagSublat("B", "B1", sublats["B"].coords, "{0,0,1}"),
    ])

magcrystal = MagCrystal(
    M,
    sublats,
    magsublats
    )

# Generate q-space spin-wave Hamiltonian
magcrystal.make_tildeJq()
