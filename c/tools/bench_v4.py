#!/usr/bin/env python3
"""Reproducible one-shot benchmark harness for the DeepSeek V4 engine."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
C_DIR = TOOLS_DIR.parent
REPO_ROOT = C_DIR.parent
PROMPT_DIR = TOOLS_DIR / "bench_v4_prompts"
PROFILE_SPECS = {
    "short": ("short.txt", 1),
    "medium": ("context.txt", 8),
    "long": ("context.txt", 64),
}

RAM_RE = re.compile(
    r"ram_tiers available=(?P<available>[\d.]+)GiB "
    r"dense=(?P<dense_state>[^ (]+)\((?P<dense>[\d.]+)GiB\) "
    r"target_slots=(?P<slots>\d+) target_cache=(?P<cache>[\d.]+)GiB "
    r"head=(?P<head>\S+) projected=(?P<projected>[\d.]+)GiB"
)
TOKENS_RE = re.compile(
    r"v4_tokens prompt=(?P<prompt>\d+) generated=(?P<generated>\d+) "
    r"total=(?P<total>\d+) expert_requests=(?P<requests>\d+) "
    r"hits=(?P<hits>\d+) misses=(?P<misses>\d+) "
    r"hit_rate=(?P<hit>[\d.]+) bytes=(?P<bytes>\d+) "
    r"target_only=(?P<target_only>[01])"
)
TIMING_RE = re.compile(
    r"timing time_to_first_token=(?P<ttft>[\d.]+)s "
    r"after_first=(?P<decode>[\d.]+)s"
)
DSPARK_RE = re.compile(
    r"v4_dspark attempts=(?P<attempts>\d+) drafted=(?P<drafted>\d+) "
    r"accepted=(?P<accepted>\d+) acceptance=(?P<acceptance>[\d.]+)%"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prompt_bytes(profile: str, prompt_dir: Path = PROMPT_DIR) -> bytes:
    name, repeats = PROFILE_SPECS[profile]
    source = (prompt_dir / name).read_bytes().rstrip() + b"\n"
    if repeats == 1:
        return source
    chunks = []
    for index in range(repeats):
        chunks.append(f"\n[Context pass {index + 1:03d}/{repeats:03d}]\n".encode())
        chunks.append(source)
    return b"".join(chunks)


def kv_bytes(config: dict[str, object], context: int) -> int:
    layers = int(config["num_hidden_layers"])
    head_dim = int(config["head_dim"])
    index_dim = int(config["index_head_dim"])
    window = int(config["sliding_window"])
    maximum = int(config.get("max_position_embeddings", context))
    context = min(context, maximum)
    ratios = list(config["compress_ratios"])
    if len(ratios) < layers:
        raise ValueError("compress_ratios has fewer entries than num_hidden_layers")
    total = layers * window * head_dim * 4
    for raw_ratio in ratios[:layers]:
        ratio = int(raw_ratio)
        if ratio <= 0:
            continue
        compressed = (context + ratio - 1) // ratio
        total += compressed * head_dim * 4
        if ratio == 4:
            total += compressed * index_dim * 4
    return total


def required_match(pattern: re.Pattern[str], text: str, label: str) -> re.Match[str]:
    match = pattern.search(text)
    if not match:
        raise ValueError(f"engine output is missing {label}")
    return match


def parse_engine_output(stderr: str, config: dict[str, object], context: int) -> dict[str, object]:
    ram = required_match(RAM_RE, stderr, "ram_tiers")
    tokens = required_match(TOKENS_RE, stderr, "v4_tokens")
    timing = required_match(TIMING_RE, stderr, "timing")
    prompt = int(tokens["prompt"])
    generated = int(tokens["generated"])
    ttft = float(timing["ttft"])
    decode_seconds = float(timing["decode"])
    decode_tokens = max(0, generated - 1)
    forward_tokens = max(1, prompt + generated - 1)
    result: dict[str, object] = {
        "prompt_tokens": prompt,
        "generated_tokens": generated,
        "prefill_tok_s": prompt / ttft if ttft > 0 else None,
        "decode_tok_s": decode_tokens / decode_seconds
        if decode_tokens and decode_seconds > 0 else None,
        "ttft_s": ttft,
        "expert_requests": int(tokens["requests"]),
        "expert_hits": int(tokens["hits"]),
        "expert_misses": int(tokens["misses"]),
        "expert_hit_pct": float(tokens["hit"]),
        "disk_bytes": int(tokens["bytes"]),
        "disk_gb_per_token": int(tokens["bytes"]) / forward_tokens / 1e9,
        "ram_available_gib": float(ram["available"]),
        "dense_state": ram["dense_state"],
        "dense_gib": float(ram["dense"]),
        "target_slots": int(ram["slots"]),
        "ram_cache_gib": float(ram["cache"]),
        "head_state": ram["head"],
        "projected_gib": float(ram["projected"]),
        "kv_gib": kv_bytes(config, context) / (1 << 30),
        "target_only": bool(int(tokens["target_only"])),
    }
    dspark = DSPARK_RE.search(stderr)
    result["dspark"] = ({
        "attempts": int(dspark["attempts"]),
        "drafted": int(dspark["drafted"]),
        "accepted": int(dspark["accepted"]),
        "acceptance_pct": float(dspark["acceptance"]),
    } if dspark else None)
    return result


def git_state() -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO_ROOT,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
        ).stdout)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "cannot determine source revision; run the benchmark from a Git work tree"
        ) from error
    return revision, dirty


def configure_dspark_environment(
    environment: dict[str, str], enabled: bool,
    overrides: list[tuple[str, str]],
) -> None:
    if enabled:
        # Full MTP is the documented DSpark path. The engine defaults both
        # gates off, so merely omitting --no-dspark would still run target-only.
        environment.update({"V4_MTP": "1", "V4_DRAFT": "3"})
    environment.update(dict(overrides))


def validate_run_mode(metrics: dict[str, object], dspark_enabled: bool) -> None:
    dspark = metrics.get("dspark")
    target_only = bool(metrics.get("target_only"))
    if dspark_enabled:
        attempts = dspark.get("attempts", 0) if isinstance(dspark, dict) else 0
        if target_only or int(attempts) < 1:
            raise ValueError(
                "--dspark produced no speculative attempts; check V4_MTP, "
                "V4_DRAFT, and the checkpoint's mtp.* tensors"
            )
    elif dspark is not None or not target_only:
        raise ValueError("target-only benchmark unexpectedly used speculative decoding")


def drop_linux_page_cache() -> None:
    if sys.platform != "linux":
        raise RuntimeError("cold cache runs require Linux /proc/sys/vm/drop_caches")
    os.sync()
    try:
        Path("/proc/sys/vm/drop_caches").write_text("3\n", encoding="ascii")
        return
    except PermissionError:
        pass
    result = subprocess.run(
        ["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"],
        input="3\n", text=True, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(
            "cannot drop Linux page cache; run `sudo -v` first or execute the "
            "benchmark from a shell allowed to write /proc/sys/vm/drop_caches"
        )


def run_engine(command: list[str], environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command, env=environment, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise RuntimeError(
            f"engine exited with {result.returncode}\nstdout:\n{result.stdout}"
            f"\nstderr:\n{result.stderr}"
        )
    return result


def environment_snapshot(environment: dict[str, str]) -> dict[str, str]:
    exact = {"CTX", "OMP_NUM_THREADS", "SEED", "COLI_TEMP"}
    return dict(sorted(
        (key, value) for key, value in environment.items()
        if key in exact or key.startswith("V4_") or key.startswith("COLI_V4_")
    ))


def compact_line(record: dict[str, object]) -> str:
    metrics = record["metrics"]
    assert isinstance(metrics, dict)
    def number(key: str, suffix: str = "", digits: int = 2) -> str:
        value = metrics.get(key)
        return "-" if value is None else f"{float(value):.{digits}f}{suffix}"
    revision = str(record["revision"]) + ("+dirty" if record["dirty"] else "")
    return (
        f"bench_v4 rev={revision} profile={record['profile']} "
        f"cache={record['cache_state']} ctx={record['context']} "
        f"prefill={number('prefill_tok_s', 'tok/s')} "
        f"decode={number('decode_tok_s', 'tok/s')} ttft={number('ttft_s', 's')} "
        f"expert_hit={number('expert_hit_pct', '%', 1)} "
        f"disk={number('disk_gb_per_token', 'GB/token')} "
        f"ram_cache={number('ram_cache_gib', 'GiB')} slots={metrics['target_slots']} "
        f"kv={number('kv_gib', 'GiB')} vram=- omp_threads={record['omp_threads']}"
    )


def parse_assignment(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("environment override must be NAME=VALUE")
    name, value = text.split("=", 1)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise argparse.ArgumentTypeError(f"invalid environment name: {name}")
    owned = {"V4_SCRATCH_MB", "COLI_V4_SAVE_USAGE"}
    if name in owned or not (name.startswith("V4_") or name.startswith("COLI_V4_")):
        raise argparse.ArgumentTypeError(
            f"--env accepts additional V4 knobs only; use the dedicated option for {name}"
        )
    return name, value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--binary", type=Path, default=C_DIR / "deepseek_v4")
    parser.add_argument("--profile", required=True, choices=PROFILE_SPECS)
    parser.add_argument("--cache-state", required=True, choices=("cold", "warm"))
    parser.add_argument("--ctx", type=int, default=32768)
    parser.add_argument("--ram-gb", type=float, default=28.0)
    parser.add_argument("--scratch-mb", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--omp-threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--dspark", action="store_true")
    parser.add_argument("--env", action="append", type=parse_assignment, default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.ctx < 2 or args.ram_gb <= 0 or args.scratch_mb < 1 or \
       args.max_tokens < 1 or args.warmup_tokens < 1 or args.omp_threads < 1:
        parser.error("numeric benchmark arguments must be positive (CTX >= 2)")
    model = args.model.resolve()
    binary = args.binary.resolve()
    config_path = model / "config.json"
    if not binary.is_file():
        parser.error(f"engine binary does not exist: {binary}")
    if not config_path.is_file():
        parser.error(f"model config does not exist: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # Validate planner geometry before a cold run drops caches and starts the
    # engine; a malformed config must not discard an expensive measurement.
    kv_bytes(config, args.ctx)
    materialized = prompt_bytes(args.profile)
    revision, dirty = git_state()
    environment = dict(os.environ)
    environment.pop("RAM_GB", None)  # one-shot V4 uses --memory-gb exclusively
    environment.update({
        "CTX": str(args.ctx),
        "V4_SCRATCH_MB": str(args.scratch_mb),
        "OMP_NUM_THREADS": str(args.omp_threads),
        "SEED": "1",
        "COLI_TEMP": "0",
        "COLI_V4_SAVE_USAGE": "0",
    })
    configure_dspark_environment(environment, args.dspark, args.env)
    with tempfile.NamedTemporaryFile(prefix=f"bench-v4-{args.profile}-", suffix=".txt") as prompt:
        prompt.write(materialized)
        prompt.flush()
        command = [
            str(binary), str(model), "--prompt-file", prompt.name,
            "--max-tokens", str(args.max_tokens),
            "--memory-gb", f"{args.ram_gb:g}",
        ]
        if not args.dspark:
            command.append("--no-dspark")
        if args.cache_state == "cold":
            drop_linux_page_cache()
        else:
            warmup = command.copy()
            warmup[warmup.index("--max-tokens") + 1] = str(args.warmup_tokens)
            run_engine(warmup, environment)
        completed = run_engine(command, environment)
    metrics = parse_engine_output(completed.stderr, config, args.ctx)
    validate_run_mode(metrics, args.dspark)
    now = dt.datetime.now().astimezone()
    record: dict[str, object] = {
        "schema": 1,
        "timestamp": now.isoformat(timespec="seconds"),
        "revision": revision,
        "dirty": dirty,
        "profile": args.profile,
        "cache_state": args.cache_state,
        "context": args.ctx,
        "ram_gb": args.ram_gb,
        "scratch_mb": args.scratch_mb,
        "max_tokens": args.max_tokens,
        "warmup_tokens": args.warmup_tokens if args.cache_state == "warm" else 0,
        "dspark_enabled": args.dspark,
        "omp_threads": args.omp_threads,
        "model": str(model),
        "binary": str(binary),
        "prompt_sha256": sha256(materialized),
        "environment": environment_snapshot(environment),
        "metrics": metrics,
    }
    output = args.output or REPO_ROOT / "bench" / f"{now.date()}-{revision}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    print(compact_line(record))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"bench_v4: {error}", file=sys.stderr)
        raise SystemExit(1)
