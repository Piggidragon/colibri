import os
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
C_ROOT = ROOT / "c"
UNITS = C_ROOT / "Makefile.deepseek-v4.units"
MANIFEST = ROOT / "plans" / "v4-dependency-closure.txt"

# backend_cuda.cu/backend_cuda_v4.cu are compiled by nvcc, not gcc, and are
# outside deepseek_v4.c's own #include graph -- gcc -MM on deepseek_v4.c with
# -DCOLI_V4_CUDA only reaches backend_cuda.h/backend_cuda_v4.h (the headers
# deepseek_v4.c itself includes), never backend_gpu_compat.h (included only
# from inside the .cu files). Scanned separately below with -x c++, close
# enough to CUDA syntax for header-dependency purposes without needing nvcc.
CUDA_SOURCES = ["backend_cuda.cu", "backend_cuda_v4.cu"]


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
            for cuda_flag in ([], ["-DCOLI_V4_CUDA"]):
                result = subprocess.run(
                    [compiler, "-MM", "-I", str(C_ROOT), f"-D{unit}",
                     *cuda_flag, str(C_ROOT / "deepseek_v4.c")],
                    check=True,
                    text=True,
                    capture_output=True,
                )
                actual.update(re.findall(r"c/[A-Za-z0-9_.-]+\.h", result.stdout))
        for cuda_source in CUDA_SOURCES:
            result = subprocess.run(
                [compiler, "-MM", "-I", str(C_ROOT), "-x", "c++",
                 "-DCOLI_V4_CUDA", str(C_ROOT / cuda_source)],
                check=True,
                text=True,
                capture_output=True,
            )
            actual.update(re.findall(r"c/[A-Za-z0-9_.-]+\.h", result.stdout))
        self.assertEqual(actual, declared_headers())


if __name__ == "__main__":
    unittest.main()
