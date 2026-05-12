#!/usr/bin/env pytest

import itertools
import random
import unittest
import logging

import pytest

from fandango import Fandango
from fandango.api import _split_combinations
from .utils import DOCS_ROOT, RESOURCES_ROOT


class APITest(unittest.TestCase):
    SPEC_abc = r"""
    <start> ::= ('a' | 'b' | 'c')+
    where str(<start>) != 'd'
    """

    SPEC_abcd = r"""
    <start> ::= ('a' | 'b' | 'c')+ 'd'
    where str(<start>) != 'd'
    """

    def test_fuzz(self):
        with open(DOCS_ROOT / "persons-faker.fan") as persons:
            fan = Fandango(persons)

        random.seed(0)
        for tree in itertools.islice(fan.generate_solutions(), 10):
            print(str(tree))

    def test_fuzz_from_string(self):
        fan = Fandango(self.SPEC_abc, logging_level=logging.INFO)
        random.seed(0)
        for tree in itertools.islice(fan.generate_solutions(), 10):
            print(str(tree))

    def test_parse(self):
        fan = Fandango(self.SPEC_abc)
        word = "abc"

        for tree in fan.parse(word):
            assert tree is not None
            print(f"tree = {repr(str(tree))}")
            print(tree.to_grammar())

    def test_incomplete_parse(self):
        fan = Fandango(self.SPEC_abcd)
        word = "ab"

        for tree in fan.parse(word, prefix=True):
            assert tree is not None
            print(f"tree = {repr(str(tree))}")
            print(tree.to_grammar())

    def test_failing_incomplete_parse(self):
        fan = Fandango(self.SPEC_abcd)
        invalid_word = "ab"

        assert len(list(fan.parse(invalid_word))) == 0

    def test_failing_parse(self):
        fan = Fandango(self.SPEC_abcd)
        invalid_word = "abcdef"

        assert len(list(fan.parse(invalid_word))) == 0

    def ensure_capped_generation(self):
        fan = Fandango(self.SPEC_abcd, logging_level=logging.INFO)
        solutions = fan.fuzz()
        self.assertLess(
            100,
            len(solutions),
            f"Expected more than 100 trees, only received {len(solutions)}",
        )


@pytest.mark.parametrize("even_number", ["0", "1", "10", "11", "12", "123", "1234"])
def test_even_number(even_number):
    with open(RESOURCES_ROOT / "even_numbers.fan", "r") as file:
        fan = Fandango(file)

    parses = list(fan.parse(even_number))

    is_even = int(even_number) % 2 == 0
    successful_parse = len(parses) > 0

    assert (
        successful_parse == is_even
    ), f"parsed for {even_number} the following: {parses}"

    assert all(
        all(c.check(p) for c in fan.constraints) for p in parses
    ), f"some parses did not match the constraints for {even_number}"


def test_split_combinations_depth_0():
    items = [1, 2, 3, 4, 5]
    combinations: list[tuple[list[int], list[int]]] = [
        ([], [1, 2, 3, 4, 5]),
    ]
    assert list(_split_combinations(items, 0)) == combinations


def test_split_combinations_depth_1():
    items = [1, 2, 3, 4, 5]
    combinations = [
        ([1], [2, 3, 4, 5]),
        ([2], [1, 3, 4, 5]),
        ([3], [1, 2, 4, 5]),
        ([4], [1, 2, 3, 5]),
        ([5], [1, 2, 3, 4]),
    ]
    assert list(_split_combinations(items, 1)) == combinations


def test_split_combinations_depth_2():
    items = [1, 2, 3, 4, 5]
    combinations = [
        ([1, 2], [3, 4, 5]),
        ([1, 3], [2, 4, 5]),
        ([1, 4], [2, 3, 5]),
        ([1, 5], [2, 3, 4]),
        ([2, 3], [1, 4, 5]),
        ([2, 4], [1, 3, 5]),
        ([2, 5], [1, 3, 4]),
        ([3, 4], [1, 2, 5]),
        ([3, 5], [1, 2, 4]),
        ([4, 5], [1, 2, 3]),
    ]
    assert list(_split_combinations(items, 2)) == combinations


def test_split_combinations_depth_full():
    items = [1, 2, 3, 4, 5]
    combinations: list[tuple[list[int], list[int]]] = [
        ([1, 2, 3, 4, 5], []),
    ]
    assert list(_split_combinations(items, 5)) == combinations


def test_split_combinations_depth_full_plus_1():
    items = [1, 2, 3, 4, 5]
    combinations: list[tuple[list[int], list[int]]] = []
    assert list(_split_combinations(items, 6)) == combinations


def test_split_combinations_empty_list():
    items: list[int] = []
    assert list(_split_combinations(items, 0)) == [([], [])]
    assert list(_split_combinations(items, 1)) == []


class TestOOL(unittest.TestCase):
    def test_invert_constraints_depth_1_single_constraint(self):
        with open(RESOURCES_ROOT / "even_numbers.fan", "r") as file:
            original = Fandango(file)
            assert len(original.constraints) == 1
            original_constraint = original.constraints[0]

        inverted_list = list(original.invert_constraints(depth=1))
        assert len(inverted_list) == 1
        inverted = inverted_list[0]
        assert len(inverted.constraints) == 1
        inverted_constraint = inverted.constraints[0]

        original_solutions = original.fuzz(desired_solutions=10)
        inverted_solutions = inverted.fuzz(desired_solutions=10)

        assert len(original_solutions) == len(inverted_solutions) == 10
        # check that own constraints are satisfied
        assert all(original_constraint.check(s) for s in original_solutions)
        assert all(inverted_constraint.check(s) for s in inverted_solutions)

        # check that inverted constraints are not satisfied
        assert all(
            not original_constraint.check(s) for s in inverted_solutions
        ), "constraint {} satisfied for {}".format(
            original_constraint.format_as_spec(),
            "\n".join(str(s) for s in inverted_solutions),
        )

        assert all(
            not inverted_constraint.check(s) for s in original_solutions
        ), "constraint {} satisfied for {}".format(
            inverted_constraint.format_as_spec(),
            "\n".join(str(s) for s in original_solutions),
        )

    def test_invert_constraints_depth_1_multiple_constraints(self):
        with open(RESOURCES_ROOT / "divisible_numbers.fan", "r") as file:
            original = Fandango(file)
            assert len(original.constraints) == 3
            original_constraints = original.constraints
            original_solutions = original.fuzz(desired_solutions=10)
            assert all(
                original_constraint.check(s)
                for s in original_solutions
                for original_constraint in original_constraints
            )

        inverted_list = list(original.invert_constraints(depth=1))
        assert len(inverted_list) == 3
        assert all(len(inverted.constraints) == 3 for inverted in inverted_list)

        # original solutions don't satisfy inverted constraints
        for inverted in inverted_list:
            assert all(
                any(
                    not inverted_constraint.check(s)
                    for inverted_constraint in inverted.constraints
                )
                for s in original_solutions
            )

        for i in range(3):
            inverted = inverted_list[i]
            all_wrong_fandangos = (
                inverted_list[:i] + inverted_list[i + 1 :] + [original]
            )
            inverted_solutions = inverted.fuzz(desired_solutions=10)

            # inverted solutions satisfy their own inverted constraints
            assert all(
                inverted_constraint.check(s)
                for s in inverted_solutions
                for inverted_constraint in inverted.constraints
            )

            # inverted solutions don't satisfy original and other inverted constraints
            assert all(
                any(
                    not wrong_constraint.check(s)
                    for wrong_constraint in wrong_fandango.constraints
                )
                for wrong_fandango in all_wrong_fandangos
                for s in inverted_solutions
            )

    def test_invert_constraints_depth_2_multiple_constraints(self):
        with open(RESOURCES_ROOT / "divisible_numbers.fan", "r") as file:
            original = Fandango(file)
            assert len(original.constraints) == 3

        inverted_list = list(original.invert_constraints(depth=2))
        assert len(inverted_list) == 3

        for inverted in inverted_list:
            assert len(inverted.constraints) == 3
            # solutions divisible by exactly one of 2, 3, 5
            assert all(
                1 == sum(int(str(solution)) % mod == 0 for mod in [2, 3, 5])
                for solution in inverted.fuzz(desired_solutions=10)
            )


if __name__ == "__main__":
    unittest.main()
