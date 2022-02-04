import re
from fractions import Fraction
import numpy as np
from math import floor

from coords import Coords


class GroupElement:
    """A group element in a space group can be represented
    by a pair: O(3) rotation "r" and a fractional translation "t".
    """

    def __init__(self, R, t):
        self.R = R
        self.t = t

    @classmethod
    def from_str(cls, s):
        result = np.array([GroupElement._parse_expr(row)
            for row in s.split(r",")], dtype=Fraction)
        R = result[:, :3]
        t = result[:, 3]

        return cls(R, t)

    def __matmul__(self, rhs):
        if isinstance(rhs, self.__class__):
            R = self.R @ rhs.R
            t = self.t + self.R@rhs.t
            return GroupElement(R, t)

        if isinstance(rhs, Coords):
            result = self.t + self.R@rhs.coords_arr
            return Coords.from_array(result)

        raise TypeError("Invalid type of rhs")

    def __eq__(self, rhs):
        return (np.all(self.R == rhs.R)
                and np.all(self.t == rhs.t))

    def __str__(self):
        r_lines = ["".join(fr"{x:^ {4}}" for x in row) \
                for row in self.R]
        t_lines = [fr"{str(x):^{8}}" for x in self.t]

        lines = GroupElement._boundary_format(
            [a + "|" + b for a, b in zip(r_lines, t_lines)])

        return "\n".join(lines)

    def to_str(self):
        def row_to_str(row):
            assert len(row) == 4
            assert all(x in (-1, 0, 1) for x in row[:3])

            result = ""
            for i, v in enumerate(('x', 'y', 'z')):
                if row[i] == 1:
                    result += "+" + v
                elif row[i] == -1:
                    result += "-" + v

            translation = row[-1]
            if translation != 0:
                result += "+" + str(translation)

            return result
        result = ",".join(row_to_str(list(self.R[i]) + [self.t[i]])
            for i in range(3))

        assert(GroupElement.from_str(result) == self)
        return result

    def _parse_term(term):
        """term := ("x"|"y"|"z"|int|int/int)

        Output is a 4x1 array.
        """

        result = None

        if term == "x":
           result = np.array([1, 0, 0, 0])
        elif term == "y":
           result = np.array([0, 1, 0, 0])
        elif term == "z":
           result = np.array([0, 0, 1, 0])
        elif re.match(r"^\d+$", term):
           num = int(term)
           result = np.array([0, 0, 0, num])
        elif re.match(r"^\d+/\d+$", term):
           numer, denom = term.split(r"/")
           numer, denom = int(numer), int(denom)
           result = np.array([0, 0, 0, Fraction(numer, denom)])
        else:
            assert(False, "Invalid term")

        return result


    def _parse_expr(expr):
        """expr := term (+|-) term ..."""

        expr = expr.replace(" ", "")

        def collapse_signs(signs):
            """If "signs" is a literal of signs, collapse it.

            Otherwise, return the input as is.
            """
            if not re.match(r"(\+|-)+", signs):
                return signs
            if signs.count('-')%2 == 1:
                return '-'
            return '+'

        ops_and_terms = (
            [collapse_signs(term) for term in re.split(r"((?:\+|-)+)", '+' + expr)]
            )

        result = np.array([0, 0, 0, 0], dtype=Fraction)

        for op, term in zip(ops_and_terms[1::2], ops_and_terms[2::2]):
            assert(op in ['-', '+'])
            sign = (-1 if op=='-' else +1)
            result += sign * GroupElement._parse_term(term)

        return result


    def _boundary_format(lines):
        """Decorate text with matrix boundaries."""
        line_count = len(lines)
        assert(line_count >= 2)


        def left_decoration(line_idx):
            if line_idx == 0:
                return "┌"
            if line_idx == line_count-1:
                return "└"
            return "│"

        def right_decoration(line_idx):
            if line_idx == 0:
                return "┐"
            if line_idx == line_count-1:
                return "┘"
            return "│"

        formatted_lines = [
            left_decoration(idx) + line + right_decoration(idx)
            for idx, line in enumerate(lines)
            ]

        return formatted_lines


class Group:
    def __init__(self, *gens):
        self.elems = Group._make_group_elems_from_gens(gens)

    @classmethod
    def from_gen_strs(cls, *gen_strs):
        gens = [GroupElement.from_str(gen_str)
                for gen_str in gen_strs]

        return cls(*gens)

    def _make_group_elems_from_gens(gens):
        """Generate group elements from list of generators."""
        from itertools import product
        from functools import reduce

        result = []
        powers_list = []

        for gs in product(*[Group._all_powers(gen) for gen in gens]):
            g = reduce(lambda g1, g2: g1 @ g2, gs)
            g.t -= np.floor(g.t)

            if g not in result:
                result.append(g)

        return result


    def _all_powers(g):
        """Given a group element, return list of all integer
        powers, with integer translation parts modded out."""
        result = []

        x = g

        while True:
            x = g @ x
            x.t -= np.floor(x.t)
            if x in result:
                break

            result.append(x)

        return result

    def __str__(self):
        return "\n".join(
            ["Group elements:"]
            + [str(g) for g in self.elems]
            + ["-----------------------------"]
            )
