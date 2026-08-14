import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent.parent
CLI = HERE / "coli"


class CliOutputLanguageTest(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(CLI), *args],
            cwd=HERE,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
            timeout=10,
        )

    def test_help_is_english(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run DeepSeek V4 Flash locally", result.stdout)
        self.assertIn("automatically apply the RAM/VRAM plan", result.stdout)
        self.assertNotIn("modello", result.stdout.lower())
        self.assertNotIn("motore", result.stdout.lower())

    def test_serve_help_includes_allowed_host(self):
        result = self.run_cli("serve", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--allowed-host", result.stdout)

    def test_tune_help_describes_measured_safe_profile(self):
        result = self.run_cli("tune", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fastest quality-preserving execution profile", result.stdout)
        self.assertIn("--min-gain", result.stdout)

    def test_info_status_is_english(self):
        with tempfile.TemporaryDirectory() as model:
            result = self.run_cli("info", "--model", model)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("config.json is missing", result.stdout)
        self.assertIn("disk", result.stdout)
        self.assertIn("engine", result.stdout)

    def test_missing_model_error_is_english(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_model = str(Path(directory) / "missing-model")
            result = self.run_cli("run", "--model", missing_model, "hello")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model not found", result.stderr)
        self.assertIn("set COLI_MODEL or use --model", result.stderr)


class ChatCapForwardingTest(unittest.TestCase):
    """`coli chat` always starts the V4 server and forwards an explicit cap."""

    @classmethod
    def setUpClass(cls):
        loader = SourceFileLoader("coli_cli_under_test", str(CLI))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        cls.coli = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.coli)

    def _chat_server_cmd(self, cap):
        coli = self.coli
        captured = {}

        class FakeProc:
            def __init__(self, cmd, **_kw):
                captured["cmd"] = cmd
            def poll(self):
                return None
            def terminate(self):
                pass
            def wait(self, timeout=None):
                return 0
            def kill(self):
                pass

        class FakeSpinner:
            def __init__(self, *_a, **_k):
                pass
            def start(self):
                pass
            def stop(self):
                pass

        model = tempfile.mkdtemp()
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", model], check=False))
        (Path(model) / "config.json").write_text(json.dumps({"model_type": "deepseek_v4"}))
        args = types.SimpleNamespace(model=model, cap=cap, ngen=256, api_key=None,
                                     no_attach=True, attach=None)
        with mock.patch.object(coli, "need_model"), \
             mock.patch.object(coli, "banner"), \
             mock.patch.object(coli, "engine_for", return_value="/stub/deepseek_v4"), \
             mock.patch.object(coli, "env_for_engine", return_value={}), \
             mock.patch.object(coli, "server_probe", return_value="deepseek-v4-flash-0731"), \
             mock.patch.object(coli, "chat_attached"), \
             mock.patch.object(coli, "Spinner", FakeSpinner), \
             mock.patch("subprocess.Popen", FakeProc):
            coli.cmd_chat(args)
        return captured["cmd"]

    def test_explicit_cap_rides_along(self):
        cmd = self._chat_server_cmd(cap=32)
        self.assertIn("--cap", cmd)
        self.assertEqual(cmd[cmd.index("--cap") + 1], "32")
        self.assertEqual(cmd[cmd.index("--arch") + 1], "deepseek_v4")

    def test_explicit_cap_zero_rides_along(self):
        cmd = self._chat_server_cmd(cap=0)
        self.assertEqual(cmd[cmd.index("--cap") + 1], "0")

    def test_absent_cap_stays_absent(self):
        cmd = self._chat_server_cmd(cap=None)
        self.assertNotIn("--cap", cmd)


class BannerModelLineTest(unittest.TestCase):
    """The banner names V4 and makes unsupported checkpoints explicit."""

    @classmethod
    def setUpClass(cls):
        loader = SourceFileLoader("coli_banner_under_test", str(CLI))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        cls.coli = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.coli)

    def make_model(self, config, shard_bytes=0):
        directory = Path(tempfile.mkdtemp(prefix="coli-banner-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(directory, ignore_errors=True))
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        if shard_bytes:
            # An empty file: the size is reported by the getsize patch in
            # line(). truncate() to the real size would be sparse on ext4 and
            # APFS but NOT on NTFS, where it allocates -- the first revision of
            # this test asked a Windows runner for 372 GB and got
            # "OSError: [Errno 28] No space left on device".
            (directory / "model-00001.safetensors").write_bytes(b"")
        return directory

    def line(self, config, shard_bytes=0):
        directory = self.make_model(config, shard_bytes)
        if not shard_bytes:
            return self.coli.model_banner_line(str(directory))
        real_getsize = os.path.getsize

        def fake_getsize(path):
            return shard_bytes if str(path).endswith(".safetensors") else real_getsize(path)

        with mock.patch.object(self.coli.os.path, "getsize", fake_getsize):
            return self.coli.model_banner_line(str(directory))

    def test_v4_names_itself(self):
        line = self.line({"model_type": "deepseek_v4", "n_routed_experts": 256})
        self.assertTrue(line.startswith("DeepSeek V4 Flash"), line)

    def test_deepseek_v4_is_not_read_as_glm(self):
        """The regression this exists for: a non-GLM checkpoint said GLM-5.2."""
        line = self.line({"model_type": "deepseek_v4", "n_routed_experts": 256})
        self.assertNotIn("GLM", line)
        self.assertNotIn("744B", line)

    def test_unknown_model_is_marked_unsupported(self):
        line = self.line({"model_type": "qwen3_moe", "num_hidden_layers": 48,
                          "n_routed_experts": 128})
        self.assertIn("qwen3_moe", line)
        self.assertIn("unsupported", line)

    def test_missing_model_type_does_not_invent_one(self):
        line = self.line({"num_hidden_layers": 32})
        self.assertIn("unsupported", line)

    def test_no_model_keeps_the_generic_tagline(self):
        self.assertIn("DeepSeek V4 Flash", self.coli.model_banner_line(None))

    def test_unreadable_model_falls_back_instead_of_raising(self):
        """`coli info` banners before validating the path; it must not crash."""
        self.assertIn("DeepSeek V4 Flash", self.coli.model_banner_line("/nonexistent/xyz"))

    def test_size_is_reported_without_rounding_to_zero(self):
        small = self.line({"model_type": "deepseek_v4"}, shard_bytes=4_200_000_000)
        self.assertIn("4.2 GB on disk", small)
        large = self.line({"model_type": "deepseek_v4"}, shard_bytes=372_000_000_000)
        self.assertIn("372 GB on disk", large)
        tiny = self.line({"model_type": "deepseek_v4"}, shard_bytes=3_000_000)
        self.assertIn("MB on disk", tiny)

    def test_model_is_keyword_only(self):
        """banner(sub, x) must not read x as a path: other PRs add arguments."""
        with self.assertRaises(TypeError):
            self.coli.banner("run", True)


class V4OmpEnvironmentTest(unittest.TestCase):
    """V4 retains control of its hybrid-core team inside the binary."""

    @classmethod
    def setUpClass(cls):
        loader = SourceFileLoader("coli_omp_under_test", str(CLI))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        cls.coli = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.coli)

    def args(self):
        return types.SimpleNamespace(model="/x", ram=None, ctx=None, ngen=None,
                                     temp=None, cap=None)

    def test_v4_leaves_thread_count_to_the_binary(self):
        with mock.patch.object(self.coli.sys, "platform", "linux"), \
             mock.patch("resource_plan.physical_cpu_count", return_value=6), \
             mock.patch.dict(os.environ, {}, clear=True):
            env = self.coli.env_for_engine(self.args(), "deepseek_v4")
        # Thread count stays in the binary's hands (V4_OMP_CORES / logical
        # default), and spin-wait tuning stays withheld from this disk-bound
        # engine (docs/deepseek-v4-tuning-32gb.md). OMP_PROC_BIND/OMP_PLACES
        # ARE set -- see the next test -- since neither controls thread count.
        for name in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "OMP_DYNAMIC"):
            self.assertNotIn(name, env)

    def test_v4_pins_placement_so_a_reduced_team_stays_off_e_cores(self):
        with mock.patch.object(self.coli.sys, "platform", "linux"), \
             mock.patch("resource_plan.physical_cpu_count", return_value=6), \
             mock.patch.dict(os.environ, {}, clear=True):
            env = self.coli.env_for_engine(self.args(), "deepseek_v4")
        self.assertEqual(env.get("OMP_PROC_BIND"), "close")
        self.assertEqual(env.get("OMP_PLACES"), "cores")

    def test_explicit_setting_still_wins(self):
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "3"}), \
             mock.patch("resource_plan.physical_cpu_count", return_value=6):
            env = self.coli.env_for_engine(self.args(), "deepseek_v4")
        self.assertEqual(env["OMP_NUM_THREADS"], "3")

    def test_kill_switch_is_honoured(self):
        with mock.patch.dict(os.environ, {"COLI_NO_OMP_TUNE": "1"}, clear=False):
            os.environ.pop("OMP_NUM_THREADS", None)
            os.environ.pop("OMP_PROC_BIND", None)
            os.environ.pop("OMP_PLACES", None)
            env = self.coli.env_for_engine(self.args(), "deepseek_v4")
        self.assertNotIn("OMP_NUM_THREADS", env)
        self.assertNotIn("OMP_PROC_BIND", env)
        self.assertNotIn("OMP_PLACES", env)


if __name__ == "__main__":
    unittest.main()
