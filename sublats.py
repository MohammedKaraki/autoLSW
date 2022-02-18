from fractions import Fraction
import numpy as np
from sympy import Matrix
from itertools import product

from crystal import Crystal


def inv(m):
    assert isinstance(m, np.ndarray)
    assert m.dtype in (int, Fraction)

    inv = np.array(Matrix(m).inv(), dtype=Fraction)
    assert np.array_equal(inv@m, np.identity(m.shape[0]))
    return inv


def find_fourier_ns(M):
    MinvT = inv(M).T

    def in_unit_square(n):
        assert n.dtype in (int, Fraction)
        return all(n >= 0) and all(n < 1)

    # This restricts the search region to [-10, 10]^3. Ad-hoc choice, but
    # simplifies the implementation. If this cube is not large enough,
    # this will be caught with the assertion that nothing at the boundary
    # is valid n.
    # Alternatively, we can check that len(result) is equal to the determinant
    # of M.
    # Both assertion are used in this function.

    N = 5
    result = []
    for x, y, z in product(range(-N, N+1),
                           range(-N, N+1),
                           range(-N, N+1)):
        n = np.array([x, y, z], dtype=int)
        if in_unit_square(MinvT @ n):
            assert all(abs(x) != N for x in n)
            result.append(n)

    det = int(round(np.linalg.det(M.astype(int))))
    assert len(result) == det
    return result


def matrix_dump(matrix):
    return "{" + ",".join("{" + ",".join(str(cell) for cell in row) + "}"
            for row in matrix) + "}"

class Sublat:
    def __init__(self, label, coords, spinvar_name):
        self.label = label
        self.coords = coords
        self.spinvar = spinvar_name


class Sublats:
    def __init__(self, sublats_list):
        self.sublats_list = sublats_list

    def __getitem__(self, label):
        index = self.label_to_index(label)
        return self.sublats_list[index]

    def count(self):
        return len(self.sublats_list)

    def label_to_index(self, label):
        result = None
        for index, sublat in enumerate(self.sublats_list):
            if sublat.label == label:
                assert result is None
                result = index

        assert result is not None
        return result


    def sublat_translation_decomp(self, site):
        sublat = None
        for cand_sublat in self.sublats_list:
            if (site - cand_sublat.coords).all_ints():
                assert(sublat is None)
                sublat = cand_sublat
        assert(sublat is not None)

        translation = site - sublat.coords
        assert translation.all_ints();
        return sublat, translation



class MagSublat:
    def __init__(self, parent_label, label, coords, direction_xyz):
        self.parent_label = parent_label
        self.label = label
        self.coords = coords
        self.direction_xyz = direction_xyz

    def get_spinvar(self, sublats):
        return sublats.sublats_list[
            sublats.label_to_index(self.parent_label)].spinvar

class MagSublats:
    def __init__(self, magsublats_list):
        self.magsublats_list = magsublats_list

    def __getitem__(self, label):
        index = self.label_to_index(label)
        return self.magsublats_list[index]

    def label_to_index(self, label):
        result = None
        for index, magsublat in enumerate(self.magsublats_list):
            if magsublat.label == label:
                assert result is None
                result = index

        assert result is not None
        return result


class MagCrystal:
    def __init__(self, M, sublats, magsublats):
        self.sublats = sublats
        self.magsublats = magsublats

        for magsublat in self.magsublats.magsublats_list:
            assert (magsublat.coords -
                self.sublats.sublats_list[self.sublats.label_to_index(
                    magsublat.parent_label)].coords).all_ints()

        assert M.dtype == int
        factor = np.linalg.det(M)
        assert factor.is_integer()
        int_factor = int(factor)
        assert int_factor == factor
        factor = int_factor

        assert isinstance(factor, int), factor
        assert factor >= 1

        assert len(magsublats.magsublats_list) == factor*sublats.count()

        self.magcell_enlargement_factor = factor
        self.M = M


        self.fourier_ns = find_fourier_ns(M)

    def make_tildeJq(self):
        num_magsublats = len(self.magsublats.magsublats_list)

        code = []
        code.append("Minv=Inverse[" + matrix_dump(self.M) + "];")
        code.append("Clear[tmpJ]")
        code.append("getBlock[orig_,sublatidx1_,sublatidx2_,kk_]:="
                "orig[[1+3*(sublatidx1);;"
                "3+3*(sublatidx1),1+3*(sublatidx2);;3+3*(sublatidx2)]]"
                "/.{k1->kk[[1]],k2->kk[[2]],k3->kk[[3]]};")

        code.append("fourierns={" + ",".join("{{{},{},{}}}".format(
            *[str(x) for x in n]) for n in self.fourier_ns) + "}"
            + ";")

        for magidx1 in range(num_magsublats):
            for magidx2 in range(num_magsublats):
                sublat1, dn1 = self.sublats.sublat_translation_decomp(
                    self.magsublats.magsublats_list[magidx1].coords)
                sublat2, dn2 = self.sublats.sublat_translation_decomp(
                    self.magsublats.magsublats_list[magidx2].coords)


                code.append("tildeJBlock[orig_,{},{},q_]:="
                    "(1/{})Total[(Exp[-I (q+2\[Pi]#).Minv.{{{},{},{}}}]"
                    "getBlock[orig,{},{},Transpose[Minv]."
                    "(q+2\[Pi]#)])&/@fourierns]".format(
                        magidx1, magidx2,
                        self.magcell_enlargement_factor,
                        *(dn1-dn2).coords_arr,
                        self.sublats.label_to_index(sublat1.label),
                        self.sublats.label_to_index(sublat2.label)
                        )
                    )

                code.append("tildeRhoBlock[orig_,{},{},q_,Rg_]:="
                    "(1/{})Total[(Exp[-I (q+2\[Pi]#).Minv.(-{{{},{},{}}}"
                    "+Rg.{{{},{},{}}})]"
                    "getBlock[orig,{},{},Transpose[Minv]."
                    "(q+2\[Pi]#)])&/@fourierns]".format(
                        magidx1, magidx2,
                        self.magcell_enlargement_factor,
                        *dn1.coords_arr, *dn2.coords_arr,
                        self.sublats.label_to_index(sublat1.label),
                        self.sublats.label_to_index(sublat2.label)
                        )
                    )

        code.append("(tildefyJ[Mat_]:=ArrayFlatten@Table["
            "tildeJBlock[Mat,row,col,{{q1,q2,q3}}],"
            "{{row,0,{0}}},{{col,0,{0}}}]"
            ");".format(num_magsublats-1)
            )

        code.append("(tildefyRho[Mat_, Rg_]:=ArrayFlatten@Table["
            "tildeRhoBlock[Mat,row,col,{{q1,q2,q3}}, Rg],"
            "{{row,0,{0}}},{{col,0,{0}}}]"
            ");".format(num_magsublats-1)
            )

        code.append(r"tildeJq[qq1_,qq2_,qq3_]="
            r"Identity[tildefyJ[totalJk]]/.{q1->qq1,q2->qq2,q3->qq3};")

        code.append(r"myRotationMatrix[n_]:=If[n=={0,0,Norm[n]},IdentityMatrix[3],"
            r"If[n=={0,0,-Norm[n]},RotationMatrix[\[Pi],{1,0,0}],"
            r"RotationMatrix[{n,{0,0,1}}]]]")

        code.append(r"DirectSum[list_] := "
            " ArrayFlatten@ReleaseHold@DiagonalMatrix[Hold /@ list]")
        code.append(r"make\[CapitalOmega][directions_]:="
            r"DirectSum[(myRotationMatrix[#]) & /@ directions]")

        spinvars = "{" + ",".join(magsublat.get_spinvar(self.sublats)
            for magsublat in self.magsublats.magsublats_list) + "}"
        code.append(r"normalizer=KroneckerProduct[DiagonalMatrix[(Sqrt[#])&/@"
            + spinvars + r"], IdentityMatrix[3]];")

        dirs = "{" + ",".join(magsublat.direction_xyz for magsublat in
            self.magsublats.magsublats_list) + "}"
        code.append(r"dirs=" + dirs + ";")
        code.append(r"omega=make\[CapitalOmega][dirs];")

        code.append(r"toLocal[mat_]:=omega.mat.Transpose[omega];")
        code.append(r"toReduced[mat_]:=With[{"
            r"relidxs=Select[Range[Length[mat]],Mod[#,3]!=0&]"
            r"}, mat[[relidxs,relidxs]]]")

        code.append(r"localtildeJ[q1_,q2_,q3_]="
            r"toLocal[tildeJq[q1,q2,q3]]//Chop//Identity;")
        code.append(r"normalizedlocaltildeJq[q1_,q2_,q3_]="
            r"normalizer.localtildeJ[q1,q2,q3].normalizer//Identity;")

        # TODO: verify (normalizer[[#2, #2]]/normalizer[[#1, #1]]) factor below
        code.append(r"""
            szIdxs=Range[3,Length[normalizedlocaltildeJq[0,0,0]],3];
            chempot=DirectSum@((#*DiagonalMatrix[{-1,-1,0}])&/@(
            Total/@Outer[(normalizer[[#2, #2]]/normalizer[[#1, #1]])normalizedlocaltildeJq[0,0,0][[
            #1,#2]]&,szIdxs,szIdxs]));
            Ridxs=Flatten[({-2,-1}+3*#)&/@Range[Length[szIdxs]]];

            R[q1_,q2_,q3_]=2(normalizedlocaltildeJq[q1,q2,q3]
                +chempot)[[Ridxs,Ridxs]];
            metric=KroneckerProduct[IdentityMatrix[Length[R[0,0,0]]/2], 
               PauliMatrix[2]];
            evals[q1_,q2_,q3_]:=NumericalSort@Chop@Eigenvalues[metric.R[q1,q2,q3]];
            """)

        code.append(r'Print["R[q1,q2,q3]:"]')
        code.append(r"MatrixForm@Identity@R[q1,q2,q3]")

        print("\n".join(code))
