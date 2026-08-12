import os
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
C_ROOT = ROOT / "c"
UNITS = C_ROOT / "Makefile.deepseek-v4.units"
MANIFEST = ROOT / "plans" / "v4-dependency-closure.txt"


def target_units():
    source = UNITS.read_text(encoding="utf-8")
    block = source.split("V4_TARGET_UNITS =", 1)[1].split(
        "V4_TEST_UNITS =", 1
    )[0]
    return re.findall(r"COLI_V4_UNIT_[A-Z0-9_]+", block)


def declared_headers():
    return {
        line
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.startswith("c/") and line.endswith(".h")
    }


class V4DependencyClosureTest(unittest.TestCase):
    def test_manifest_matches_every_v4_production_unit(self):
        compiler = os.environ.get("CC", "gcc")
        actual = set()
        for unit in target_units():
            result = subprocess.run(
                [compiler, "-MM", "-I", str(C_ROOT), f"-D{unit}",
                 str(C_ROOT / "deepseek_v4.c")],
                check=True,
                text=True,
                capture_output=True,
            )
            actual.update(re.findall(r"c/[A-Za-z0-9_.-]+\.h", result.stdout))
        self.assertEqual(actual, declared_headers())


if __name__ == "__main__":
    unittest.main()
