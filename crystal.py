import numpy as np
from fractions import Fraction

import itertools
from group import Group, GroupElement
from coords import Coords



class Crystal:

    def __init__(self, group, sublats, conv_axes):
        self.group = group
        self.sublats = sublats
        self.conv_axes = conv_axes

        self.combined_code = []


    def _find_bond_little_group(self, site1, site2):
        littlegroup_noswap = []
        littlegroup_swap = []

        for g in self.group.elems:
            site1prime = g @ site1
            site2prime = g @ site2


            cond_noswap = (site1prime-site1) == (site2prime-site2)
            cond_swap = (site1prime-site2) == (site2prime-site1)

            assert(not cond_noswap
                   or not cond_swap
                   or site1==site2)

            if cond_noswap and (site1prime-site1).all_ints():
                assert((site2prime-site2).all_ints())
                littlegroup_noswap.append(g)
            if cond_swap and (site1prime-site2).all_ints():
                assert((site2prime-site1).all_ints())
                littlegroup_swap.append(g)

        return {"noswap": littlegroup_noswap,
                "swap": littlegroup_swap}

    @staticmethod
    def _make_matrix_defs(label, H_rep, only_heisenberg=False,
            suggested_J=None, suggested_P=None):
        code = []
        template = r"M={{Mxx,Mxy,Mxz},{Myx,Myy,Myz},{Mzx,Mzy,Mzz}}"
        templateHeisenberg = r"M={{Mj,0,0},{0,Mj,0},{0,0,Mj}}"

        if H_rep=="trivial":
            if not suggested_J:
                if only_heisenberg:
                    code.append(templateHeisenberg.replace(
                        "M", "J"+label) + ";")
                else:
                    code.append(template.replace("M", "J"+label) + ";")
                code.append("J=" + "J" + label + ";")
            else:
                code.append("J=" + suggested_J + ";")

        elif H_rep=="vector":
            axes = ["x", "y", "z"]
            if not suggested_P:
                for rowlabel in axes:
                    if only_heisenberg:
                        code.append(templateHeisenberg.replace(
                            "M", "P"+rowlabel+label) + ";")
                    else:
                        code.append(template.replace(
                            "M", "P"+rowlabel+label) + ";")
                code.append(
                    "P="
                    + "{"
                    + ",".join("P"+rowlabel+label for rowlabel in axes)
                    + "};"
                    )
            else:
                code.append("P=" + suggested_P + ";")

        elif H_rep=="tensor":
            axes = ["x", "y", "z"]

            for rowlabel in axes:
                for collabel in axes:
                    if only_heisenberg:
                        code.append(templateHeisenberg.replace(
                            "M",
                            r"S"+label+rowlabel+collabel) + ";")
                    else:
                        code.append(template.replace(
                            "M",
                            r"S"+label+rowlabel+collabel) + ";")
            code.append(
                r"\[CapitalSigma]="
                + "{"
                + ",".join("{"
                           + ",".join(
                               r"S"+label+rowlabel+collabel
                               for collabel in axes
                               )
                           + "}" for rowlabel in axes)
                + "};"
                )
        else:
            raise ValueError("Invalid H_rep value")

        return code

    @staticmethod
    def _to_mathematica(R):
        return ("{"
                + ",".join("{"
                           + ",".join(str(cell) for cell in row)
                           + "}"
                           for row in R)
                + "}")

    @staticmethod
    def _make_Rg_Og_Ogpolar_defs(R):
        return " ".join(["Rg=" + Crystal._to_mathematica(R) + ";",
                        "Ogpolar=RotAxesInv.Rg.RotAxes;",
                        "Og=Det[Rg]Ogpolar;"]
                       )


    @staticmethod
    def _make_transformed_matrix(H_rep):
        if H_rep == "trivial":
            return "(Transpose[Og].J.Og)"
        elif H_rep == "vector":
            return "(Inverse[Ogpolar]." \
                   + "Map[(Transpose[Og].#.Og)&, P, {1}])"
        elif H_rep == "tensor":
            return "ReleaseHold[Inverse[Ogpolar]." \
                   + "Map[Hold[(Transpose[Og].#.Og)]&, \[CapitalSigma]," \
                   + "{2}].Transpose[Inverse[Ogpolar]]]"
        else:
            raise ValueError("Invalid H_rep")

    @staticmethod
    def _make_constraints(label, H_rep, littlegroup):
        code = []

        def make_rhs(H_rep, swap):
            assert isinstance(swap, bool)

            if H_rep == "trivial":
                if swap:
                    return "Transpose[J]"
                else:
                    return "J"
            elif H_rep == "vector":
                if swap:
                    return "Map[Transpose[#]&, P, {1}]"
                else:
                    return "P"
            elif H_rep == "tensor":
                if swap:
                    return "Map[Transpose[#]&, \[CapitalSigma], {2}]"
                else:
                    return "\[CapitalSigma]"
            else:
                raise ValueError("Invalid H_rep")


        g_swap_pairs = (
            [(g, False) for g in littlegroup["noswap"]]
            + [(g, True) for g in littlegroup["swap"]]
            )

        code.append("constraints={")
        for g, swap in g_swap_pairs:
            transformed_matrix = Crystal._make_transformed_matrix(H_rep)

            code.append("(")
            code.append(Crystal._make_Rg_Og_Ogpolar_defs(g.R))
            code.append(transformed_matrix + "==" + make_rhs(H_rep, swap))
            code.append("),")
        code.append("True};")

        return code


    # def _inv(intmatrix):
    @staticmethod
    def _inv(fracmatrix):
        def make_array_fractional(arr):
            return np.array([[Fraction.from_float(x).limit_denominator() for x in row]
                for row in arr])


        def fractional_inv(arr):
            float_arr = arr.astype(float)
            float_inv = np.linalg.inv(float_arr)
            frac_inv = make_array_fractional(float_inv)
            assert np.all(frac_inv@arr == np.identity(arr.shape[0]))

            return frac_inv
        return fractional_inv(fracmatrix)
        # result = np.linalg.inv(intmatrix.astype(int)).astype(int)
        # assert np.array_equal(result@intmatrix, intmatrix@result)
        # assert np.array_equal(result@intmatrix, np.identity(3, dtype=Fraction))

        return result

    def add_bond(self, site1, site2, label, H_rep,
                 mask=None,
                 only_heisenberg=False,
                 suggested_J=None,
                 suggested_P=None):
        assert H_rep=="trivial" or H_rep=="vector" or H_rep=="tensor", \
               "Incorrect choice of bond Hamiltonian representation"

        code = [""] * 2

        littlegroup = self._find_bond_little_group(site1, site2)

        code.extend(Crystal._make_matrix_defs(label,
                                              H_rep,
                                              only_heisenberg=only_heisenberg,
                                              suggested_J=suggested_J,
                                              suggested_P=suggested_P))
        code.extend(Crystal._make_constraints(label,
                                              H_rep,
                                              littlegroup))
        code.append("rules=ToRules[Reduce[constraints,"
            + "Sort@DeleteDuplicates@Cases[constraints,"
            + "_Symbol, \[Infinity]]]];")

        if H_rep == "trivial":
            code.append(r"J=(J/.rules);")
            code.append(r"MatrixForm[J]")
        elif H_rep == "vector":
            code.append(r"P=(P/.rules);")
            code.append(r"MatrixForm[MatrixForm/@P]")
        elif H_rep == "tensor":
            code.append(r"\[CapitalSigma]=(\[CapitalSigma]/.rules);")
            code.append(r"MatrixForm[\[CapitalSigma]]")
        else:
            raise ValueError("Invalid H_rep")

        considered_ginvs = []
        considered_bonds = []
        for ginv in self.group.elems:
            site1prime = ginv @ site1
            site2prime = ginv @ site2

            unique_bond = True
            for consideredbond_site1, consideredbond_site2 in considered_bonds:
                d11 = site1prime - consideredbond_site1
                d12 = site1prime - consideredbond_site2
                d21 = site2prime - consideredbond_site1
                d22 = site2prime - consideredbond_site2
                if ((d11==d22 and d11.all_ints())
                    or (d12==d21 and d12.all_ints())):
                    unique_bond = False
                    break

            if unique_bond:
                considered_bonds.append((site1prime, site2prime))
                considered_ginvs.append(ginv)

        del considered_bonds

        code.append(r"Clear[curJk];")
        code.append(r"curJk[row_,col_]=ConstantArray[0,{3,3}];")

        for ginv in considered_ginvs:
            site1prime = ginv @ site1
            site2prime = ginv @ site2

            sublat1, unitcell1 = self.sublats.sublat_translation_decomp(
                site1prime)
            sublat2, unitcell2 = self.sublats.sublat_translation_decomp(
                site2prime)

            Rg = Crystal._inv(ginv.R)

            row = self.sublats.label_to_index(sublat1.label)
            col = self.sublats.label_to_index(sublat2.label)

            code.append(Crystal._make_Rg_Og_Ogpolar_defs(Rg))

            transformed_matrix = Crystal._make_transformed_matrix(H_rep)
            if H_rep == "trivial":
                code.append(r"curJ={};".format(transformed_matrix))
            elif H_rep == "vector":
                code.append(
                    r"curJ=Total[Flatten[Evector*{},{{1}}]];".format(
                        transformed_matrix))
            elif H_rep == "tensor":
                code.append(r"curJ=Total[Flatten[\[Sigma]*{},{{1,2}}]];".format(
                    transformed_matrix))
            else:
                raise ValueError("Invalid H_rep")

            exp1 = r"Exp[I{{k1,k2,k3}}.{{{},{},{}}}]".format(
                *unitcell1.coords_arr)
            exp2 = r"Exp[-I{{k1,k2,k3}}.{{{},{},{}}}]".format(
                *unitcell2.coords_arr)

            curJk_submatrix = r"curJk[{},{}]".format(row, col)
            code.append(curJk_submatrix + "=" + curJk_submatrix + "+"
                  + r"({}*{}*curJ)//Simplify;".format(exp1, exp2))

        num_sublats = self.sublats.count()
        code.append(
            r"""(totalJk=totalJk+Identity[TrigToExp[ComplexExpand[ArrayFlatten[
            ((1/2)*Table[{1}
            (curJk[row,col]+ConjugateTranspose[curJk[col,row]]),
            {{row,0,{0}}},
            {{col,0,{0}}}]
            )]]]])//MatrixForm;""".format(
                num_sublats-1, mask+"*" if mask else ""))

        self.combined_code.extend(code)

    @staticmethod
    def _is_fraction_or_int_array(arr):
        return arr.dtype == int or arr.dtype == Fraction

    def add_rep_of_group_element(self, g, label):
        Rinv = Crystal._inv(g.R)
        tinv = -Rinv@g.t
        ginv = GroupElement(Rinv, tinv)

        assert (g@ginv).to_str() == "+x,+y,+z"

        self.add_rep_of_group_element_incorrect(g, str(label) + "incorrect")
        self.add_rep_of_group_element_incorrect(ginv, str(label))


    def add_rep_of_group_element_incorrect(self, g, label):
        Rinv = Crystal._inv(g.R)

        code = []

        code.append("Clear[rep]")
        code.append("rep[row_,col_]=ConstantArray[0,{3,3}];")
        code.append(self._make_Rg_Og_Ogpolar_defs(g.R))
        for sublati in self.sublats.sublats_list:
            sitei = sublati.coords

            assert Crystal._is_fraction_or_int_array(g.t)
            assert Crystal._is_fraction_or_int_array(sitei.coords_arr)
            d_arr = sitei.coords_arr - g.t
            assert Crystal._is_fraction_or_int_array(d_arr)
            sitej_coords = Rinv @ (d_arr)
            assert Crystal._is_fraction_or_int_array(sitej_coords)
            sitej = Coords.from_array(sitej_coords)

            sublatj, delta_n = self.sublats.sublat_translation_decomp(sitej)

            code.append("deltan={{{},{},{}}};".format(
                *delta_n.coords_arr))

            row = self.sublats.label_to_index(sublati.label)
            col = self.sublats.label_to_index(sublatj.label)
            code.append(
                "rep[{},{}]"
                "=Exp[I{{k1,k2,k3}}.(-Rg.deltan)]Og;".format(
                    row, col))
        num_sublats = self.sublats.count()


        krep_function_name = r"\[Rho]k{}".format(label)
        code.append(
            r"""({1}[k1_,k2_,k3_]=Table[rep[row,col],
            {{row,0,{0}}},
            {{col,0,{0}}}]
            //ArrayFlatten);""".format(
                num_sublats-1, krep_function_name))

        qrep_function_name = r"\[Rho]q{}".format(label)
        code.append(
                r"""{0}[qq1_,qq2_,qq3_]:=ReplaceAll[tildefyRho[
                {1}[k1,k2,k3], {2}
                ],{{q1->qq1,q2->qq2,q3->qq3}}]//Identity;""".format(
                qrep_function_name,
                krep_function_name,
                self._to_mathematica(g.R)))


        self.combined_code.extend(code)



    def gen_mathematica_code(self, 
            additional_gstrs=None,
            suggested_Efield=None, suggested_strain=None):
        code = []
        # code.append(r'Clear["Global`*"]')


        if not suggested_Efield:
            code.append(r"Evector={Ex,Ey,Ez};")
        else:
            code.append(r"Evector={};".format(suggested_Efield))

        if not suggested_strain:
            code.append(r"M={{Mxx,Mxy,Mxz},"
                        "{Mxy,Myy,Myz},{Mxz,Myz,Mzz}};".replace(
                            "M", r"\[Sigma]"))
        else:
            code.append(r"\[Sigma]={};".format(suggested_strain))



        code.append("RotAxes=Transpose[{"
                    + ",".join(self.conv_axes)
                    + "}];")
        code.append("RotAxesInv=Inverse[RotAxes];")

        def group_to_mathematica(additional_gstrs):
            gStr_code = []
            gStr_code.append("gStrs={};")
            gStr_code.append("Rgs={};")

            gStr_code.append("primitiveGroupElemCount={};".format(
                len(self.group.elems)));
            gstrs = [h.to_str() for h in self.group.elems]
            if additional_gstrs is not None:
                gstrs.extend(additional_gstrs)

            gs = [GroupElement.from_str(x) for x in gstrs]
            Rgs = [ '{' + ','.join('{' + ','.join(str(x) for x in row) + '}'
                            for row in g.R) + '}'
                            for g in gs
                            ]

            for i, g in enumerate(gs):
                self.add_rep_of_group_element(g, i+1)
            gStr_code.append('gStrs={' + ','.join('"'+x+'"' for x in gstrs) + '};')
            gStr_code.append('Rgs={' + ','.join(Rgs) + '};')

            # i = 0
            # for h in self.group.elems:
            #     h_str = h.to_str()
            #     for dx, dy, dz in itertools.product([0], [0], [0]):
            #         i += 1
            #         g = GroupElement.from_str(h_str)
            #         g.t[0] += dx
            #         g.t[1] += dy
            #         g.t[2] += dz
            #
            #         g_str = g.to_str()
            #         self.add_rep_of_group_element(
            #             GroupElement.from_str(g_str),
            #             str(i))
            #
            #         gStr_code.append("gStrs=gStrs~Join~{{\"{}\"}};".format(g_str))
            #         gStr_code.append("Rgs=Rgs~Join~{{{}}};".format(
            #             "{" + ",".join("{" + ",".join(str(x) for x in row) + "}"
            #                 for row in g.R) + "}"))
            return "\n".join(gStr_code)

        code.append(group_to_mathematica(additional_gstrs))



        code.append("totalJk=ConstantArray[0,{{{0},{0}}}];".format(
            3 * self.sublats.count()))
        code.extend(self.combined_code)

        code.append(r"valVecPairs[mat_]:="
            r"(sys=Eigensystem[mat];"
            r"ord=Ordering[sys[[1]]];"
            r"Transpose@{sys[[1,ord]],sys[[2,ord]]})")
        code.append(r"vecSym[vec_,sym_]:="
            r"Conjugate[vec].sym.vec/(Conjugate[vec].vec)//Rationalize")
        code.append(r'\[Rho]str[s_] :='
            r'ToExpression["\[Rho]q" <> ToString[FirstPosition[gStrs, s][[1]]]]')
        code.append(r'\[Rho]strincorrect[s_] :='
            r'ToExpression["\[Rho]q" <> ToString[FirstPosition[gStrs, s][[1]]] <> "incorrect"]')
        print("\n".join(code))
