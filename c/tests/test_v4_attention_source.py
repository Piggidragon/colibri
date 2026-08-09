import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE = (ROOT / "deepseek_v4.c").read_text(encoding="utf-8")


def definition_end(source, start):
    opening = source.index("{", start)
    depth = 0
    for position in range(opening, len(source)):
        if source[position] == "{":
            depth += 1
        elif source[position] == "}":
            depth -= 1
            if depth == 0:
                return position + 1
    raise AssertionError("unterminated definition")


def definitions(source, marker):
    result = []
    position = 0
    while True:
        start = source.find(marker, position)
        if start < 0:
            return result
        end = definition_end(source, start)
        result.append(source[start:end])
        position = end


def regions(source, start_marker, end_marker):
    starts = []
    position = 0
    while True:
        start = source.find(start_marker, position)
        if start < 0:
            break
        starts.append(start)
        position = start + len(start_marker)
    ends = definitions(source, end_marker)
    if len(starts) != len(ends):
        raise AssertionError(
            f"{start_marker!r}: {len(starts)} starts but {len(ends)} ends"
        )
    return [source[start : source.index(end, start) + len(end)]
            for start, end in zip(starts, ends)]


class DeepSeekV4AmalgamSourceTest(unittest.TestCase):
    def assert_copies(self, copies, expected, source):
        self.assertEqual(len(copies), expected, f"copy count for {source}")
        for copy in copies[1:]:
            self.assertEqual(copy, copies[0], f"diverged copy of {source}")

    def test_attention_copies_stay_identical(self):
        copies = regions(
            ENGINE,
            "struct ColiDeepSeekV4WindowAttentionState {",
            "int coli_v4_attention_window_token_ref(\n",
        )
        self.assert_copies(copies, 3, "deepseek_v4_attention.c")

    def test_all_attention_callers_keep_two_source_migration(self):
        token_callers = definitions(ENGINE, "static int attention_token_impl(")
        self.assertEqual(len(token_callers), 3, "attention_token_impl copy count")
        batch_callers = definitions(
            ENGINE, "int coli_v4_attention_window_batch_ref(\n"
        )
        self.assertEqual(len(batch_callers), 1, "standalone batch path count")

        for caller in token_callers + batch_callers:
            self.assertEqual(
                caller.count("coli_v4_attention_two_source_codec_ref("), 1
            )
            self.assertEqual(caller.count("v4_cuda_flash_attention("), 1)
            self.assertIn(
                "if (v4_flash_enabled() && state->kv_device &&", caller
            )
            self.assertIn(
                "heads, head_dim, state->rope_dim, state->row_bytes,", caller
            )
            self.assertEqual(caller.count("v4_attention_cuda_write("), 2)
            self.assertEqual(
                caller.count("window_indices[i] = i <= position ? i : -1;"), 1
            )
            self.assertEqual(
                caller.count("int oldest = (position + 1) % state->window_size;"),
                1,
            )
            self.assertEqual(
                caller.count(
                    "window_indices[i] = (oldest + i) % state->window_size;"
                ),
                1,
            )
            self.assertNotIn("all_kv", caller)
            self.assertNotIn("coli_v4_sparse_attention_ref(", caller)
            self.assertEqual(caller.count("encode_attention_kv_row("), 2)
            self.assertNotIn("coli_v4_kv_encode_row(", caller)

        batch = batch_callers[0]
        self.assertIn(
            "if (!result) compressed_counts[item] = state->compressed_count;",
            batch,
        )
        self.assertIn(
            "state->compressed, compressed_counts[item], window_indices,",
            batch,
        )
        self.assertIn(
            "item_compressed_indices, selected, state->codec, state->rope_dim,",
            batch,
        )

    def test_attention_snapshot_cuda_restore_is_bulk(self):
        restore = definitions(
            ENGINE, "int coli_v4_attention_snapshot_restore("
        )
        self.assertEqual(len(restore), 1)
        self.assertEqual(restore[0].count("v4_cuda_kv_write_row("), 2)
        self.assertNotIn("for (int slot", restore[0])
        self.assertIn(
            "(size_t)state->window_size * state->row_bytes", restore[0]
        )
        self.assertIn(
            "(size_t)state->compressed_count * state->row_bytes", restore[0]
        )

    def test_attention_encode_failures_set_an_error(self):
        helpers = definitions(ENGINE, "static int encode_attention_kv_row(")
        self.assert_copies(helpers, 3, "attention KV encode helper")
        self.assertIn('"cannot encode KV row"', helpers[0])

    def test_compressor_copies_stay_identical(self):
        copies = regions(
            ENGINE,
            "struct ColiDeepSeekV4CompressorState {",
            "int coli_v4_compressor_step(",
        )
        self.assert_copies(copies, 2, "deepseek_v4_compressor.c")

    def test_indexer_copies_stay_identical(self):
        copies = regions(
            ENGINE,
            "struct ColiDeepSeekV4Indexer {",
            "int coli_v4_indexer_compressed_count(",
        )
        self.assert_copies(copies, 2, "deepseek_v4_indexer.c")

    def test_common_layer_definitions_stay_identical(self):
        # The resident unit intentionally adds rows8 packing to layer_load.
        # The surrounding definitions are still duplicated source and must not
        # drift while that one explicitly different function remains exempt.
        self.assertIn("/* ---- begin include deepseek_v4_layer.c ---- */", ENGINE)
        self.assertIn("/* ---- end include deepseek_v4_layer.c ---- */", ENGINE)
        self.assertIn("/* ######## deepseek_v4_layer.c ######## */", ENGINE)
        self.assertIn("#endif /* COLI_V4_UNIT_LAYER */", ENGINE)
        resident_start = ENGINE.index("/* ---- begin include deepseek_v4_layer.c ---- */")
        resident_end = ENGINE.index("/* ---- end include deepseek_v4_layer.c ---- */",
                                    resident_start)
        resident = ENGINE[resident_start:resident_end]
        normal_start = ENGINE.index("/* ######## deepseek_v4_layer.c ######## */")
        normal_end = ENGINE.index("#endif /* COLI_V4_UNIT_LAYER */", normal_start)
        normal = ENGINE[normal_start:normal_end]
        for marker in (
            "int coli_v4_layer_plan(",
            "int coli_v4_layer_validate(",
            "void coli_v4_layer_free(",
            "const void *coli_v4_layer_data(",
        ):
            copies = definitions(resident, marker) + definitions(normal, marker)
            self.assert_copies(copies, 2, marker)


if __name__ == "__main__":
    unittest.main()
