# DeepSeek V4 Flash is the sole supported runtime in this Linux x86-64 fork.
# Keep the public targets here; the amalgamation's object-unit graph remains in
# Makefile.deepseek-v4, where each target unit can be built independently.

PYTHON ?= python3
ARCH ?= native
PREFIX ?= /usr/local
BINDIR ?= $(PREFIX)/bin
LIBEXECDIR ?= $(PREFIX)/libexec/colibri
INSTALL ?= install

# Same non-CUDA CFLAGS as Makefile.deepseek-v4, duplicated here (not included)
# because these test binaries link directly against deepseek_v4.c object units
# built with test-only -D flags that must not leak into the real V4_OBJS build.
CC = gcc
CFLAGS = -D_GNU_SOURCE -O3 -march=$(ARCH) -fopenmp -pthread -include pthread.h \
	-Wall -Wextra -Wno-unused-parameter -Wno-misleading-indentation \
	-Wno-unused-function
LDFLAGS = -lm -fopenmp -pthread

.PHONY: all deepseek-v4 deepseek-v4-clean deepseek-v4-tiny-generate \
	deepseek-v4-tiny-check deepseek-v4-oracle test test-c test-python \
	check clean install uninstall rans

all: deepseek-v4

deepseek-v4:
	$(MAKE) -f Makefile.deepseek-v4 ARCH=$(ARCH) deepseek-v4

deepseek-v4-clean:
	$(MAKE) -f Makefile.deepseek-v4 deepseek-v4-clean

deepseek-v4-tiny-generate:
	$(PYTHON) tools/make_deepseek_v4_tiny.py --output ./deepseek_v4_tiny --force

deepseek-v4-tiny-check: deepseek-v4-tiny-generate
	$(MAKE) -f Makefile.deepseek-v4 deepseek-v4-clean
	$(MAKE) -f Makefile.deepseek-v4 ARCH=$(ARCH) deepseek-v4
	$(PYTHON) tests/test_deepseek_v4_tiny.py --binary ./deepseek_v4 --fixture ./deepseek_v4_tiny
	$(PYTHON) tests/test_deepseek_v4_prefix.py --binary $(CURDIR)/deepseek_v4 --fixture $(CURDIR)/deepseek_v4_tiny

deepseek-v4-oracle: deepseek-v4
	@test -n "$(MODEL)" || { echo "usage: make deepseek-v4-oracle MODEL=/path/to/checkpoint" >&2; exit 2; }
	$(PYTHON) tools/make_deepseek_v4_oracle.py --model "$(MODEL)" --binary ./deepseek_v4 \
		--output tests/deepseek_v4_oracle.json \
		--memory-gb $(or $(MEMORY_GB),32) \
		--prompt "$(or $(ORACLE_PROMPT),The capital of France is)" \
		--validate \
		--teacher-forcing $(or $(ORACLE_TEACHER_FORCING),32) \
		--greedy $(or $(ORACLE_GREEDY),20)

# --- C test gates ---
# A rule below of the form `tests/test_x: tests/test_x.c ...` is what makes
# test_x a gate -- TEST_RULES derives the list from the rules themselves, so
# adding a gate means adding the test's rule here, nothing else. Tests that
# depended on the deleted multi-engine colibri.c amalgam (GLM/Inkling/Kimi
# K3/OLMoE, dropped in 878e08d/1953aa8) have no rule and are therefore not
# gates; their .c files are dead until a future PR removes them outright.
TEST_RULES := $(shell sed -n 's|^tests/\(test_[a-z0-9_]*\):.*|\1|p' $(firstword $(MAKEFILE_LIST)))
TEST_BINS  := $(addprefix tests/,$(TEST_RULES))

tests/test_serve_sentinel: tests/test_serve_sentinel.c compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_ue8m0: tests/test_ue8m0.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_json: tests/test_json.c json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# test_tok and test_tok_kimi take a tokenizer.json fixture and a cases file as
# argv, not env/hardcoded paths, so run_tests.py invoking them with no args
# would just print usage and "pass" -- deliberately no gate rule for either;
# build+run them by hand: ./tests/test_tok tokenizer.json < cases
tests/test_tok_o200k: tests/test_tok_o200k.c tok.h tok_unicode.h tok_unicode_o200k.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_tok_kimi_tiny: tests/test_tok_kimi_tiny.c tok.h tok_unicode.h tok_unicode_o200k.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_fse: tests/test_fse.c fse_coli.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st_pread: tests/test_st_pread.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st_shape: tests/test_st_shape.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st: tests/test_st.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_st_mirror: tests/test_st_mirror.c st.h json.h compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_tier: tests/test_tier.c tier.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_grammar: tests/test_grammar.c grammar.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_ablate: tests/test_ablate.c abl.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_schema_gbnf: tests/test_schema_gbnf.c schema_gbnf.h grammar.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_decode_batch: tests/test_decode_batch.c decode_batch.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_e8_kernel: tests/test_e8_kernel.c quant.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_rans: tests/test_rans.c rans.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_fp8_passthrough: tests/test_fp8_passthrough.c quant.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_i4_acc512: tests/test_i4_acc512.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_compat_direct: tests/test_compat_direct.c compat.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_expert_store_ops: tests/test_expert_store_ops.c expert_store.h tensor.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_route_trace: tests/test_route_trace.c route_trace.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_kv_prefix: tests/test_kv_prefix.c kv_prefix.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_native_quant: tests/test_native_quant.c deepseek_v4.c native_quant.h tensor.h quant.h
	$(CC) $(CFLAGS) -DCOLI_V4_UNIT_NATIVE_QUANT deepseek_v4.c $< -o $@ $(LDFLAGS)

tests/test_v4_scratch_env: tests/test_v4_scratch_env.c deepseek_v4.c deepseek_v4_internal.h
	$(CC) $(CFLAGS) -DCOLI_V4_UNIT_RESOURCE_PLAN deepseek_v4.c $< -o $@ $(LDFLAGS)

tests/test_v4_prefill_chunk: tests/test_v4_prefill_chunk.c deepseek_v4.c deepseek_v4_internal.h
	$(CC) $(CFLAGS) -DCOLI_V4_UNIT_RESOURCE_PLAN deepseek_v4.c $< -o $@ $(LDFLAGS)

tests/test_v4_pin_env: tests/test_v4_pin_env.c deepseek_v4.c deepseek_v4_internal.h
	$(CC) $(CFLAGS) -DCOLI_V4_UNIT_RESOURCE_PLAN \
		-DCOLI_V4_MAX_PIN_SLOTS_PER_LAYER=16 -DCOLI_V4_PIN_RAMP_REQUESTS=24 \
		deepseek_v4.c $< -o $@ $(LDFLAGS)

tests/test_v4_omp_tune: tests/test_v4_omp_tune.c omp_tune.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_v4_mirror: tests/test_v4_mirror.c v4_mirror.h st.h compat.h json.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# coli_v4_indexer_select is a static inline in deepseek_v4_internal.h (see
# plans/12-expert-cache-policy.md Commit 2), so this test needs no unit
# define and no deepseek_v4.c -- just the header.
tests/test_v4_indexer_select: tests/test_v4_indexer_select.c deepseek_v4_internal.h
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

tests/test_v4_flash_attention: tests/test_v4_flash_attention.c deepseek_v4.c deepseek_v4_internal.h v4_kv_codec.h native_quant.h turbo_quant.h
	$(CC) $(CFLAGS) -DCOLI_V4_UNIT_SPARSE_ATTENTION -DCOLI_V4_UNIT_NATIVE_QUANT \
		-DCOLI_V4_UNIT_KV_CODEC \
		deepseek_v4.c $< -o $@ $(LDFLAGS)

tests/test_v4_kv_codec: tests/test_v4_kv_codec.c deepseek_v4.c v4_kv_codec.h turbo_quant.h
	$(CC) $(CFLAGS) -DCOLI_V4_UNIT_KV_CODEC -DCOLI_V4_UNIT_NATIVE_QUANT \
		deepseek_v4.c $< -o $@ $(LDFLAGS)

tests/test_v4_context_bytes: tests/test_v4_context_bytes.c deepseek_v4.c v4_kv_codec.h turbo_quant.h
	$(CC) $(CFLAGS) -DCOLI_V4_UNIT_KV_CODEC -DCOLI_V4_UNIT_NATIVE_QUANT \
		deepseek_v4.c $< -o $@ $(LDFLAGS)

tests/test_v4_turbo_quant: tests/test_v4_turbo_quant.c deepseek_v4.c v4_kv_codec.h turbo_quant.h
	$(CC) $(CFLAGS) -DCOLI_V4_UNIT_KV_CODEC -DCOLI_V4_UNIT_NATIVE_QUANT \
		deepseek_v4.c $< -o $@ $(LDFLAGS)

V4_KV_SNAPSHOT_DIR = build/kv_snapshot
# attention.o contains the full attention transaction unit.  This isolated
# snapshot test intentionally relies on function/data sections plus the linker
# GC flag below to discard unrelated engine references; keep both in sync.
V4_KV_SNAPSHOT_CFLAGS = $(CFLAGS) -ffunction-sections -fdata-sections \
	-DCOLI_V4_TEST_HOOKS
V4_KV_SNAPSHOT_GC_LDFLAG = -Wl,--gc-sections

$(V4_KV_SNAPSHOT_DIR):
	@mkdir -p $@

$(V4_KV_SNAPSHOT_DIR)/attention.o: deepseek_v4.c deepseek_v4_internal.h v4_kv_codec.h | $(V4_KV_SNAPSHOT_DIR)
	$(CC) $(V4_KV_SNAPSHOT_CFLAGS) -DCOLI_V4_UNIT_ATTENTION_TRANSACTION \
		-c deepseek_v4.c -o $@

$(V4_KV_SNAPSHOT_DIR)/indexer_snapshot.o: deepseek_v4.c deepseek_v4_internal.h v4_kv_codec.h | $(V4_KV_SNAPSHOT_DIR)
	$(CC) $(V4_KV_SNAPSHOT_CFLAGS) -DCOLI_V4_UNIT_INDEXER_SNAPSHOT \
		-c deepseek_v4.c -o $@

$(V4_KV_SNAPSHOT_DIR)/codec.o: deepseek_v4.c v4_kv_codec.h native_quant.h turbo_quant.h | $(V4_KV_SNAPSHOT_DIR)
	$(CC) $(V4_KV_SNAPSHOT_CFLAGS) -DCOLI_V4_UNIT_KV_CODEC \
		-c deepseek_v4.c -o $@

$(V4_KV_SNAPSHOT_DIR)/quant.o: deepseek_v4.c native_quant.h | $(V4_KV_SNAPSHOT_DIR)
	$(CC) $(V4_KV_SNAPSHOT_CFLAGS) -DCOLI_V4_UNIT_NATIVE_QUANT \
		-c deepseek_v4.c -o $@

tests/test_v4_kv_snapshot: tests/test_v4_kv_snapshot.c \
	$(V4_KV_SNAPSHOT_DIR)/attention.o $(V4_KV_SNAPSHOT_DIR)/codec.o \
	$(V4_KV_SNAPSHOT_DIR)/indexer_snapshot.o \
	$(V4_KV_SNAPSHOT_DIR)/quant.o
	$(CC) $(V4_KV_SNAPSHOT_CFLAGS) $^ -o $@ $(LDFLAGS) \
		$(V4_KV_SNAPSHOT_GC_LDFLAG)

include Makefile.deepseek-v4.units
V4_TEST_LINK_OBJS = \
	COLI_V4_UNIT_ST.o \
	COLI_V4_UNIT_CONFIG.o \
	COLI_V4_UNIT_MATH.o \
	COLI_V4_UNIT_SPARSE_ATTENTION.o \
	COLI_V4_UNIT_RESOURCE_PLAN.o \
	COLI_V4_UNIT_PROMPT.o \
	$(addsuffix .o,$(V4_TEST_UNITS)) \
	COLI_V4_UNIT_KV_CODEC.o \
	COLI_V4_UNIT_NATIVE_QUANT.o

tests/test_deepseek_v4: tests/test_deepseek_v4.c deepseek_v4.c deepseek_v4.h \
		Makefile.deepseek-v4.units
	$(MAKE) -f Makefile.deepseek-v4 ARCH=$(ARCH) deepseek-v4-test-objs \
		COLI_V4_UNIT_ST.o COLI_V4_UNIT_CONFIG.o COLI_V4_UNIT_MATH.o COLI_V4_UNIT_SPARSE_ATTENTION.o \
		COLI_V4_UNIT_RESOURCE_PLAN.o COLI_V4_UNIT_PROMPT.o COLI_V4_UNIT_KV_CODEC.o \
		COLI_V4_UNIT_NATIVE_QUANT.o
	$(CC) $(CFLAGS) $< $(V4_TEST_LINK_OBJS) -o $@ -pthread $(LDFLAGS)

V4_OWN_DIR = build/ownership
V4_OWN_CFLAGS = $(CFLAGS) -DCOLI_V4_TEST_HOOKS
V4_OWNERSHIP_OBJS = \
	$(V4_OWN_DIR)/COLI_V4_UNIT_RUNTIME.o \
	$(V4_OWN_DIR)/COLI_V4_UNIT_CONFIG.o \
	$(V4_OWN_DIR)/COLI_V4_UNIT_ST.o \
	$(V4_OWN_DIR)/COLI_V4_UNIT_KV_CODEC.o \
	$(V4_OWN_DIR)/COLI_V4_UNIT_NATIVE_QUANT.o

$(V4_OWN_DIR):
	mkdir -p $(V4_OWN_DIR)

$(V4_OWN_DIR)/COLI_V4_UNIT_RUNTIME.o: deepseek_v4.c deepseek_v4.h deepseek_v4_internal.h | $(V4_OWN_DIR)
	$(CC) $(V4_OWN_CFLAGS) -DCOLI_V4_UNIT_RUNTIME -c deepseek_v4.c -o $@

$(V4_OWN_DIR)/COLI_V4_UNIT_CONFIG.o: deepseek_v4.c deepseek_v4.h deepseek_v4_internal.h | $(V4_OWN_DIR)
	$(CC) $(V4_OWN_CFLAGS) -DCOLI_V4_UNIT_CONFIG -c deepseek_v4.c -o $@

$(V4_OWN_DIR)/COLI_V4_UNIT_ST.o: deepseek_v4.c deepseek_v4.h deepseek_v4_internal.h st.h | $(V4_OWN_DIR)
	$(CC) $(V4_OWN_CFLAGS) -DCOLI_V4_UNIT_ST -c deepseek_v4.c -o $@

$(V4_OWN_DIR)/COLI_V4_UNIT_NATIVE_QUANT.o: deepseek_v4.c deepseek_v4.h deepseek_v4_internal.h quant.h | $(V4_OWN_DIR)
	$(CC) $(V4_OWN_CFLAGS) -DCOLI_V4_UNIT_NATIVE_QUANT -c deepseek_v4.c -o $@

$(V4_OWN_DIR)/COLI_V4_UNIT_KV_CODEC.o: deepseek_v4.c deepseek_v4.h deepseek_v4_internal.h v4_kv_codec.h native_quant.h turbo_quant.h | $(V4_OWN_DIR)
	$(CC) $(V4_OWN_CFLAGS) -DCOLI_V4_UNIT_KV_CODEC -c deepseek_v4.c -o $@

tests/test_v4_ownership: tests/test_v4_ownership.c \
		tests/v4_engine_fixture.h $(V4_OWNERSHIP_OBJS)
	$(CC) $(V4_OWN_CFLAGS) $< $(V4_OWNERSHIP_OBJS) -o $@ -pthread $(LDFLAGS)

tests/test_v4_dspark_reserve: tests/test_v4_dspark_reserve.c \
		tests/v4_engine_fixture.h $(V4_OWNERSHIP_OBJS)
	$(CC) $(V4_OWN_CFLAGS) $< $(V4_OWNERSHIP_OBJS) -o $@ -pthread $(LDFLAGS)

V4_HEAD_TIER_DIR = build/v4-head-tier

$(V4_HEAD_TIER_DIR):
	mkdir -p $(V4_HEAD_TIER_DIR)

$(V4_HEAD_TIER_DIR)/head_cache.o: deepseek_v4.c deepseek_v4.h \
		deepseek_v4_internal.h | $(V4_HEAD_TIER_DIR)
	$(CC) $(CFLAGS) -DCOLI_V4_UNIT_HEAD_CACHE -c deepseek_v4.c -o $@

# The head-cache unit reaches into the safetensors reader for its load path.
$(V4_HEAD_TIER_DIR)/st.o: deepseek_v4.c deepseek_v4.h deepseek_v4_internal.h \
		st.h | $(V4_HEAD_TIER_DIR)
	$(CC) $(CFLAGS) -DCOLI_V4_UNIT_ST -c deepseek_v4.c -o $@

# ... which in turn wants the BF16 decoder from the quant unit.
$(V4_HEAD_TIER_DIR)/native_quant.o: deepseek_v4.c deepseek_v4.h \
		deepseek_v4_internal.h quant.h | $(V4_HEAD_TIER_DIR)
	$(CC) $(CFLAGS) -DCOLI_V4_UNIT_NATIVE_QUANT -c deepseek_v4.c -o $@

tests/test_v4_head_tier: tests/test_v4_head_tier.c \
		$(V4_HEAD_TIER_DIR)/head_cache.o $(V4_HEAD_TIER_DIR)/st.o \
		$(V4_HEAD_TIER_DIR)/native_quant.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

V4_ROWS8_DIR = build/v4-rows8
V4_ROWS8_CFLAGS = $(CFLAGS) -DCOLI_V4_TEST_HOOKS

$(V4_ROWS8_DIR):
	mkdir -p $(V4_ROWS8_DIR)

$(V4_ROWS8_DIR)/layer_resident.o: deepseek_v4.c deepseek_v4.h \
		deepseek_v4_internal.h st.h | $(V4_ROWS8_DIR)
	$(CC) $(V4_ROWS8_CFLAGS) -DCOLI_V4_UNIT_LAYER_RESIDENT \
		-c deepseek_v4.c -o $@

$(V4_ROWS8_DIR)/st.o: deepseek_v4.c deepseek_v4.h \
		deepseek_v4_internal.h st.h | $(V4_ROWS8_DIR)
	$(CC) $(V4_ROWS8_CFLAGS) -DCOLI_V4_UNIT_ST -c deepseek_v4.c -o $@

$(V4_ROWS8_DIR)/native_quant.o: deepseek_v4.c deepseek_v4.h \
		deepseek_v4_internal.h native_quant.h quant.h | $(V4_ROWS8_DIR)
	$(CC) $(V4_ROWS8_CFLAGS) -DCOLI_V4_UNIT_NATIVE_QUANT \
		-c deepseek_v4.c -o $@

tests/test_v4_rows8_skip: tests/test_v4_rows8_skip.c \
		$(V4_ROWS8_DIR)/layer_resident.o $(V4_ROWS8_DIR)/st.o \
		$(V4_ROWS8_DIR)/native_quant.o
	$(CC) $(V4_ROWS8_CFLAGS) $^ -o $@ -pthread $(LDFLAGS)

# A private object dir, like V4_ROWS8_DIR: sub-making COLI_V4_UNIT_RESOURCE_PLAN.o
# in the shared directory races tests/test_deepseek_v4's sub-make under make -j,
# and both would rewrite .deepseek-v4-build-config underneath each other.
V4_DENSE_TIER_DIR = build/v4-dense-tier

$(V4_DENSE_TIER_DIR):
	mkdir -p $(V4_DENSE_TIER_DIR)

$(V4_DENSE_TIER_DIR)/resource_plan.o: deepseek_v4.c deepseek_v4.h \
		deepseek_v4_internal.h | $(V4_DENSE_TIER_DIR)
	$(CC) $(CFLAGS) -DCOLI_V4_UNIT_RESOURCE_PLAN -c deepseek_v4.c -o $@

tests/test_v4_dense_tier: tests/test_v4_dense_tier.c \
		$(V4_DENSE_TIER_DIR)/resource_plan.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

tests/test_v4_vram_tier: tests/test_v4_vram_tier.c \
		$(V4_DENSE_TIER_DIR)/resource_plan.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

tests/test_v4_tier_feedback: tests/test_v4_tier_feedback.c \
		$(V4_DENSE_TIER_DIR)/resource_plan.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

test-c: $(TEST_BINS)
	$(PYTHON) tools/run_tests.py $(TEST_BINS)

# int4-rans256-g0 codec bridge for tools/repack_rans.py + tools/rans_verify.py.
# OPTIONAL: both tools fall back to a pure-Python codec producing the same
# bytes far more slowly when the library is absent (fine for tests, not for a
# real repack).
RANSLIB = tools/librans_c.so
rans: $(RANSLIB)
$(RANSLIB): tools/rans_ctypes.c rans.h
	$(CC) $(CFLAGS) -fPIC -shared $< -o $@ $(LDFLAGS)

# test-python builds the rans ctypes bridge first: without it the repack e2e
# test's C-vs-Python byte-identity pin would silently compare Python against
# itself (the test ALSO skips loudly if the library is somehow still absent,
# e.g. when invoked via unittest directly -- but a green `make test` must mean
# the pin was actually checked).
test-python: $(RANSLIB)
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

test: test-c test-python

check: deepseek-v4 test

install: deepseek-v4
	$(INSTALL) -d $(DESTDIR)$(BINDIR) $(DESTDIR)$(LIBEXECDIR) $(DESTDIR)$(LIBEXECDIR)/tools
	$(INSTALL) -m 755 coli $(DESTDIR)$(BINDIR)/coli
	$(INSTALL) -m 755 deepseek_v4 $(DESTDIR)$(LIBEXECDIR)/deepseek_v4
	$(INSTALL) -m 644 resource_plan.py doctor.py autotune.py openai_server.py version.py $(DESTDIR)$(LIBEXECDIR)/
	$(INSTALL) -m 644 tools/*.py $(DESTDIR)$(LIBEXECDIR)/tools/
	@# The dashboard is an optional build artifact (cd web && npm run build), so install
	@# it only when it exists. It goes NEXT TO openai_server.py, which probes ./web/dist.
	@if [ -f ../web/dist/index.html ]; then \
	  $(INSTALL) -d $(DESTDIR)$(LIBEXECDIR)/web/dist; \
	  cp -R ../web/dist/. $(DESTDIR)$(LIBEXECDIR)/web/dist/; \
	  echo "installed the web dashboard"; \
	else \
	  echo "web/dist absent -- skipping the dashboard (build it with: cd web && npm run build)"; \
	fi

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/coli
	rm -rf $(DESTDIR)$(LIBEXECDIR)

clean: deepseek-v4-clean
	$(PYTHON) tools/clean.py
