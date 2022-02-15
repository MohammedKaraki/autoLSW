import numpy as np
from fractions import Fraction
from crystal import Crystal
from coords import Coords
from group import Group, GroupElement
from sublats import Sublat, Sublats, MagSublat, MagSublats, MagCrystal


group = Group.from_gen_strs(
    "-y+z, x-y+z, z",
    "-x, -y, -z",
)


z = Fraction(1, 99)

sublats = Sublats([
    Sublat("A", Coords(1*(-z), 2*(-z), 3*(-z)) + Coords(1, 2, 2), "1"),
    Sublat("B", Coords(1*z, 2*z, 3*z), "1"),
    ])


conv_axes = [
    r"{1, 0, 0}",
    r"{1/Sqrt[3], 2/Sqrt[3], 0}",
    r"{1, 2, 3}"
    ]

crystal = Crystal(group, sublats, conv_axes)

crystal.add_bond(sublats["A"].coords + Coords(0, 0, 0),
                 sublats["A"].coords + Coords(0, 0, 0),
                 label="0",
                 H_rep="trivial",
                 )

crystal.add_bond(sublats["A"].coords + Coords(0, 0, 0),
                 sublats["B"].coords + Coords(0, 0, 0),
                 label="1",
                 H_rep="trivial",
                 )

crystal.add_bond(sublats["A"].coords + Coords(0, 0, 0),
                 sublats["A"].coords + Coords(1, 0, 0),
                 label="3",
                 H_rep="trivial",
                 )


crystal.add_bond(sublats["A"].coords + Coords(0, 0, 0),
                 sublats["A"].coords + Coords(0, 0, 1),
                 label="4A",
                 H_rep="trivial",
                 )

crystal.add_bond(sublats["A"].coords + Coords(0, 0, 0),
                 sublats["B"].coords + Coords(0, 0, 1),
                 label="4B",
                 H_rep="trivial",
                 )




# crystal.add_rep_of_group_element(
#     GroupElement.from_str(r"-x,-y,-z"),
#     label="Inv")

crystal.gen_mathematica_code(suggested_Efield="{Ex,0,0}")

M = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 2]], dtype=int)

magsublats = MagSublats([
    MagSublat("A", "A1", sublats["A"].coords, "{1,0,0}"),
    MagSublat("B", "B1", sublats["B"].coords, "{1,0,0}"),
    MagSublat("A", "A2", sublats["A"].coords+Coords(0,0,1), "{-1,0,0}"),
    MagSublat("B", "B2", sublats["B"].coords+Coords(0,0,1), "{-1,0,0}"),
    ])

magcrystal = MagCrystal(
    M,
    sublats,
    magsublats
    )

magcrystal.make_tildeJq()
