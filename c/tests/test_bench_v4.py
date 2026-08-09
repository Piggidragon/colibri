from __future__ import annotations

import sys
import subprocess
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import bench_v4


class BenchV4Test(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "num_hidden_layers": 3,
            "head_dim": 8,
            "index_head_dim": 4,
            "sliding_window": 2,
            "max_position_embeddings": 32,
            "compress_ratios": [0, 4, 8],
        }

    def test_kv_bytes_matches_planner_formula_and_clamps_context(self) -> None:
        # window: 3*2*8*4=192; ratio4: 8*(8+4)*4=384;
        # ratio8: 4*8*4=128.
        self.assertEqual(bench_v4.kv_bytes(self.config, 99), 704)

    def test_kv_bytes_ignores_trailing_ratios_like_engine(self) -> None:
        self.config["compress_ratios"] = [0, 4, 8, 128]
        self.assertEqual(bench_v4.kv_bytes(self.config, 99), 704)

    def test_kv_bytes_rejects_missing_ratios(self) -> None:
        self.config["compress_ratios"] = [0, 4]
        with self.assertRaises(ValueError):
            bench_v4.kv_bytes(self.config, 32)

    def test_parse_engine_output(self) -> None:
        stderr = """
ram_tiers available=28.00GiB dense=resident(6.27GiB) target_slots=27 target_cache=15.40GiB head=resident-bf16 projected=27.90GiB
v4_tokens prompt=500 generated=5 total=505 expert_requests=1000 hits=183 misses=817 hit_rate=18.300 bytes=1200000000 target_only=1
timing time_to_first_token=10.000s after_first=4.000s
"""
        result = bench_v4.parse_engine_output(stderr, self.config, 32)
        self.assertEqual(result["prefill_tok_s"], 50.0)
        self.assertEqual(result["decode_tok_s"], 1.0)
        self.assertEqual(result["target_slots"], 27)
        self.assertAlmostEqual(result["expert_hit_pct"], 18.3)
        self.assertAlmostEqual(result["disk_gb_per_token"], 1.2e9 / 504 / 1e9)
        self.assertIsNone(result["dspark"])

    def test_parse_engine_output_accepts_current_vram_report(self) -> None:
        stderr = """
ram_tiers available=28.00GiB vram=11.46GiB(reserve=1.32GiB) dense=vram(host=0.81GiB device=5.46GiB) target_slots=40 target_cache=21.42GiB head=vram-bf16 dspark=vram-lazy projected=27.95GiB
v4_dspark attempts=2 drafted=4 accepted=1 acceptance=25.0% adaptive_disabled=1
v4_tokens prompt=1 generated=24 total=25 expert_requests=7740 hits=5450 misses=2290 hit_rate=70.413 bytes=30615797760 target_only=0
timing time_to_first_token=3.205s after_first=46.833s
"""
        result = bench_v4.parse_engine_output(stderr, self.config, 32)
        self.assertEqual(result["dense_state"], "vram")
        self.assertAlmostEqual(result["dense_gib"], 0.81)
        self.assertFalse(result["target_only"])
        self.assertEqual(result["dspark"]["accepted"], 1)

    def test_dspark_flag_enables_full_mtp_defaults(self) -> None:
        environment = {"V4_MTP": "0", "V4_DRAFT": "0"}
        bench_v4.configure_dspark_environment(environment, True, [])
        self.assertEqual(environment["V4_MTP"], "1")
        self.assertEqual(environment["V4_DRAFT"], "3")

    def test_explicit_dspark_overrides_win_but_inert_run_is_rejected(self) -> None:
        environment = {}
        bench_v4.configure_dspark_environment(
            environment, True, [("V4_DRAFT", "0")]
        )
        self.assertEqual(environment["V4_DRAFT"], "0")
        with self.assertRaisesRegex(ValueError, "no speculative attempts"):
            bench_v4.validate_run_mode(
                {"target_only": True, "dspark": None}, True
            )

    def test_run_mode_validation_accepts_real_dspark_metrics(self) -> None:
        bench_v4.validate_run_mode(
            {"target_only": False, "dspark": {"attempts": 4}}, True
        )
        bench_v4.validate_run_mode(
            {"target_only": True, "dspark": None}, False
        )

    @mock.patch("bench_v4.subprocess.run")
    def test_git_state_wraps_non_git_failure(self, run: mock.Mock) -> None:
        run.side_effect = subprocess.CalledProcessError(128, ["git", "rev-parse"])
        with self.assertRaisesRegex(RuntimeError, "from a Git work tree"):
            bench_v4.git_state()

    def test_materialized_profiles_are_fixed_and_ordered(self) -> None:
        short = bench_v4.prompt_bytes("short")
        medium = bench_v4.prompt_bytes("medium")
        long = bench_v4.prompt_bytes("long")
        self.assertGreater(len(short), 1000)
        self.assertGreater(len(medium), len(short) * 8)
        self.assertGreater(len(long), len(medium) * 7)
        self.assertEqual(long, bench_v4.prompt_bytes("long"))

    def test_assignment_validation(self) -> None:
        self.assertEqual(bench_v4.parse_assignment("V4_DRAFT=3"), ("V4_DRAFT", "3"))
        with self.assertRaises(Exception):
            bench_v4.parse_assignment("not-an-assignment")
        with self.assertRaises(Exception):
            bench_v4.parse_assignment("CTX=4096")
        with self.assertRaises(Exception):
            bench_v4.parse_assignment("V4_SCRATCH_MB=64")


if __name__ == "__main__":
    unittest.main()
