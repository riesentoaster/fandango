#!/usr/bin/env pytest

import os
import re
import shlex
import shutil
import subprocess
import unittest
import time

from fandango.cli import get_parser


class test_cli(unittest.TestCase):
    def run_command(self, command):
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = proc.communicate()
        return out.decode(), err.decode(), proc.returncode

    def test_help(self):
        command = shlex.split("fandango --help")
        out, err, code = self.run_command(command)
        parser = get_parser(True)
        self.assertEqual(0, code)
        self.assertEqual(out, parser.format_help())
        self.assertEqual(err, "")

    def test_fuzz_basic(self):
        command = shlex.split(
            "fandango fuzz -f tests/resources/digit.fan -n 10 --random-seed 426912 --no-cache"
        )
        expected = """35716
4
9768
30
5658
5
9
649
20
41"""
        out, err, code = self.run_command(command)
        self.assertEqual(0, code)
        self.assertEqual(expected, out.strip())
        self.assertEqual("", err)

    def test_output_to_file(self):
        command = shlex.split(
            "fandango fuzz -f tests/resources/digit.fan -n 10 --random-seed 426912 -o tests/resources/test.txt -s ; --no-cache"
        )
        expected = "35716;4;9768;30;5658;5;9;649;20;41"
        out, err, code = self.run_command(command)
        self.assertEqual(0, code)
        self.assertEqual("", out)
        self.assertEqual("", err)
        with open("tests/resources/test.txt", "r") as fd:
            actual = fd.read()
        os.remove("tests/resources/test.txt")
        self.assertEqual(expected, actual)

    def test_output_multiple_files(self):
        command = shlex.split(
            "fandango fuzz -f tests/resources/digit.fan -n 10 --random-seed 426912 -d tests/resources/test --no-cache"
        )
        expected = ["35716", "4", "9768", "30", "5658", "5", "9", "649", "20", "41"]
        (
            out,
            err,
            code,
        ) = self.run_command(command)
        self.assertEqual(0, code)
        self.assertEqual("", out)
        self.assertEqual("", err)
        for i, expected_value in enumerate(expected):
            filename = f"tests/resources/test/fandango-{i:04d}.txt"
            with open(filename, "r") as fd:
                actual = fd.read()
            self.assertEqual(expected_value, actual)
            os.remove(filename)
        shutil.rmtree("tests/resources/test")

    def test_output_with_libfuzzer_harness(self):
        compile = shlex.split(
            "clang -g -O2 -fPIC -shared -o tests/resources/test_libfuzzer_interface tests/resources/test_libfuzzer_interface.c"
        )
        out, err, code = self.run_command(compile)
        self.assertEqual("", out)
        self.assertEqual("", err)
        self.assertEqual(0, code)

        command = shlex.split(
            "fandango fuzz -f tests/resources/digit.fan -n 10 --random-seed 426912 --file-mode binary --no-cache --input-method libfuzzer tests/resources/test_libfuzzer_interface"
        )
        expected = ["35716", "4", "9768", "30", "5658", "5", "9", "649", "20", "41"]
        expected_output = "\n".join([f"data: {value}" for value in expected]) + "\n"
        out, err, code = self.run_command(command)
        self.assertEqual("", err)
        self.assertEqual(expected_output, out)
        self.assertEqual(0, code)

    def test_infinite_mode(self):
        command = shlex.split(
            "fandango fuzz -f tests/resources/digit.fan --infinite --no-cache"
        )
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(20)
        self.assertIsNone(proc.poll(), "Process terminated before 20 seconds")
        proc.terminate()
        out, _ = proc.communicate()
        printed_lines = out.splitlines()
        self.assertGreater(
            len(printed_lines),
            100,
            f"Not enough output lines: {len(printed_lines)}",
        )

    def test_unsat(self):
        command = shlex.split(
            "fandango fuzz -f tests/resources/digit.fan -n 10 --random-seed 426912 -c False --max-generations 50"
        )
        expected = """fandango:ERROR: Population did not converge to a perfect population
fandango:ERROR: Only found 0 perfect solutions, instead of the required 10
"""
        out, err, code = self.run_command(command)
        self.assertEqual(0, code)
        self.assertEqual("", out)
        self.assertEqual(expected, err)

    def test_binfinity(self):
        command = shlex.split(
            "fandango fuzz -f docs/binfinity.fan -n 1 --format=none --validate --random-seed 426912"
        )
        out, err, code = self.run_command(command)
        self.assertEqual("", err)
        self.assertEqual("", out)
        self.assertEqual(0, code)

    def test_infinity(self):
        # docs/infinity.fan can only generate a limited number of individuals,
        # so we decrease the population size
        command = shlex.split(
            "fandango fuzz -f docs/infinity.fan -n 1 --format=none --validate --random-seed 426912 --population-size 10"
        )
        out, err, code = self.run_command(command)
        self.assertEqual("", err)
        self.assertEqual("", out)
        self.assertEqual(0, code)

    def test_max_repetition(self):
        command = shlex.split(
            "fandango fuzz -f tests/resources/digit.fan -n 10 --max-generations 50 --max-repetitions 10 --no-cache -c 'len(str(<start>)) > 10'"
        )
        expected = """fandango:ERROR: Population did not converge to a perfect population
fandango:ERROR: Only found 0 perfect solutions, instead of the required 10
"""
        out, err, code = self.run_command(command)
        self.assertEqual(0, code)
        self.assertEqual("", out)
        self.assertEqual(expected, err)

    def test_max_nodes_unsat(self):
        command = shlex.split(
            "fandango fuzz -f tests/resources/gen_number.fan  -n 10 --population-size 10 --max-generations 30 --no-cache -c 'len(str(<start>)) > 60' --max-nodes 30"
        )
        err_pattern = """fandango:ERROR: Population did not converge to a perfect population
fandango:ERROR: Only found (\d) perfect solutions, instead of the required 10"""
        out_pattern = """(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\.\d+\n)*"""
        out, err, code = self.run_command(command)
        self.assertRegex(out, out_pattern)
        self.assertRegex(err, err_pattern)
        self.assertEqual(0, code)

        num_from_error_message = int(re.findall(err_pattern, err)[0])
        self.assertEqual(num_from_error_message, len(out.split("\n")) - 1)

    def test_unparse_grammar(self):
        # We unparse the standard library as well as docs/persons.fan
        command = shlex.split(
            """
                              sh -c 'printf "set -f docs/persons.fan\nset" | fandango shell'
                              """
        )
        out, err, code = self.run_command(command)
        self.assertEqual(0, code)
        self.assertEqual("", err)
        self.assertTrue(out.startswith("<_char> ::= r'(.|\\n)'\n"))
        self.assertTrue(out.endswith("<age> ::= <digit>+\n"))
