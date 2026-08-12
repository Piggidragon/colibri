# Tools

These scripts support DeepSeek-V4-Flash engineering. They are not runtime
dependencies of the C engine.

- `bench_v4.py`: reproducible short, medium, and long V4 benchmark profiles.
- `make_deepseek_v4_tiny.py`: generate the ignored deterministic Tiny fixture.
- `make_deepseek_v4_oracle.py`: record and verify full-checkpoint target oracles.
- `repack_fp8_passthrough.py`: byte-preserving FP8 repack utility.
- `clean.py`: remove generated V4 build artifacts.

Run tools from `c/`. The full checkpoint remains the only valid DSpark fixture:

```sh
python3 tools/bench_v4.py --model /path/to/deepseek-v4-flash --profile short
make deepseek-v4-tiny-check
```
