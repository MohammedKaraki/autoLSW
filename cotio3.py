import numpy as np
from fractions import Fraction
from crystal import Crystal
from coords import Coords
from group import Group, GroupElement
from sublats import Sublat, Sublats, MagSublat, MagSublats, MagCrystal


def main():
    group = Group.from_gen_strs(
            "x,y,z",
            "-y+z,x-y+z,z",
            "-x+y,-x+z,z",
            "-x,-y,-z",
            "y-z,-x+y-z,-z",
            "x-y,x-z,-z",
            "x+1,y+1,z+1",
            "-y+z+1,x-y+z+1,z+1",
            "-x+y+1,-x+z+1,z+1",
            "-x+1,-y+1,-z+1",
            "y-z+1,-x+y-z+1,-z+1",
            "x-y+1,x-z+1,-z+1",
            "x+1,y+2,z+2",
            "-y+z+1,x-y+z+2,z+2",
            "-x+y+1,-x+z+2,z+2",
            "-x+1,-y+2,-z+2",
            "y-z+1,-x+y-z+2,-z+2",
            "x-y+1,x-z+2,-z+2",
            )


    z = Fraction(3562, 10000)
    sublats = Sublats([
        Sublat("A", Coords(z, 2*z, -1+3*z), "S"),
        Sublat("B", Coords(1-z, 1-2*z, 1-3*z), "S")
        ])

    conv_axes = [
        r"{50/253, 0, 0}",
        r"{50/(253 Sqrt[3]), 100/(253 Sqrt[3]), 0}",
        r"{10/139, 20/139, 30/139}",
        ]

    crystal = Crystal(group, sublats, conv_axes)

    crystal.add_bond(sublats["A"].coords + Coords(0, 0, 0),
            sublats["B"].coords + Coords(0, 0, 0),
            label="1",
            H_rep="trivial",
            )

    crystal.add_bond(sublats["A"].coords + Coords(0, 0, 0),
            sublats["B"].coords + Coords(0, 1, 1),
            label="2",
            H_rep="trivial",
            )

    crystal.add_bond(sublats["A"].coords + Coords(0, 0, 0),
            sublats["A"].coords + Coords(0, 1, 0),
            label="3",
            H_rep="trivial",
            )

    crystal.add_bond(sublats["A"].coords + Coords(0, 0, 0),
            sublats["A"].coords + Coords(0, 0, 1),
            label="4",
            H_rep="trivial",
            )

    crystal.add_bond(sublats["A"].coords + Coords(0, 0, 0),
            sublats["B"].coords + Coords(1, 1, 0),
            label="5",
            H_rep="trivial",
            )

    crystal.add_bond(sublats["A"].coords + Coords(0, 0, 0),
            sublats["B"].coords + Coords(0, 0, -1),
            label="6",
            H_rep="trivial",
            )


    crystal.add_bond(sublats["A"].coords + Coords(0, 0, 0),
            sublats["A"].coords + Coords(0, 1, 0),
            label="3",
            H_rep="vector",
            )



    crystal.gen_mathematica_code(additional_gstrs=[
        "x,y,z+1"
        ])

    M = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 2]], dtype=int)

    magsublats = MagSublats([
        MagSublat("A", "A1", sublats["A"].coords,
            "{1, 0, 0}"),
        MagSublat("B", "B1", sublats["B"].coords,
            "{1, 0, 0}"),
        MagSublat("A", "A2", sublats["A"].coords + Coords(0, 0, 1),
            "{-1, 0, 0}"),
        MagSublat("B", "B2", sublats["B"].coords + Coords(0, 0, 1),
            "{-1, 0, 0}"),
        ])

    magcrystal = MagCrystal(
            M,
            sublats,
            magsublats
            )

    magcrystal.make_tildeJq()


if __name__ == "__main__":
    main()
