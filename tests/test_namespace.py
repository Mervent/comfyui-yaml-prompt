"""Tests for yaml_include namespacing: Jinja function, multi-doc merge, scoped vars."""

import pytest
import yaml

from conftest import INCLUDES_DIR
from yaml_prompt.jinja_env import render_template
from yaml_prompt.parser import YAMLPromptTemplateParser
from yaml_prompt.pipeline import PipelineError, merge_documents


class TestMergeDocuments:
    def test_single_doc_no_namespace(self):
        docs = [{"meta": ["detailed"], "vars": {"x": "1"}}]

        merged, namespaces = merge_documents(docs)

        assert merged == {"meta": ["detailed"], "vars": {"x": "1"}}
        assert namespaces == frozenset()

    def test_none_docs_skipped(self):
        docs = [None, {"meta": ["a"]}, None]

        merged, _ = merge_documents(docs)

        assert merged == {"meta": ["a"]}

    def test_empty_list(self):
        merged, namespaces = merge_documents([])

        assert merged == {}
        assert namespaces == frozenset()

    def test_namespace_prefixes_keys(self):
        docs = [
            {"_namespace": "fx", "effects": ["sparkles"], "vars": {"w": "1"}},
            {"meta": ["detailed"]},
        ]

        merged, namespaces = merge_documents(docs)

        assert "fx.effects" in merged
        assert "fx.vars" in merged
        assert "meta" in merged
        assert "_namespace" not in merged
        assert namespaces == frozenset({"fx"})

    def test_multiple_namespaces(self):
        docs = [
            {"_namespace": "fx", "effects": ["glow"]},
            {"_namespace": "common", "style": ["photo"]},
            {"meta": ["detailed"]},
        ]

        merged, namespaces = merge_documents(docs)

        assert merged == {
            "fx.effects": ["glow"],
            "common.style": ["photo"],
            "meta": ["detailed"],
        }
        assert namespaces == frozenset({"fx", "common"})

    def test_duplicate_namespace_raises(self):
        docs = [
            {"_namespace": "fx", "a": 1},
            {"_namespace": "fx", "b": 2},
        ]

        with pytest.raises(PipelineError, match="Duplicate namespace"):
            merge_documents(docs)

    def test_root_docs_merge_last_wins(self):
        docs = [
            {"meta": ["first"]},
            {"meta": ["second"]},
        ]

        merged, _ = merge_documents(docs)

        assert merged["meta"] == ["second"]


class TestYamlIncludeFunction:
    def test_basic_include(self):
        raw = '{{ yaml_include("ns_common.yaml", "common") }}\nmeta:\n  - detailed\n'

        result = render_template(raw, search_paths=[INCLUDES_DIR], seed=42)

        assert "_namespace: common" in result
        assert "---" in result
        assert "photorealistic" in result
        assert "meta:" in result

    def test_auto_namespace_generates_hex(self):
        raw = '{{ yaml_include("ns_common.yaml") }}\nmeta:\n  - detailed\n'

        result = render_template(raw, search_paths=[INCLUDES_DIR], seed=42)

        lines = result.strip().split("\n")
        ns_line = next(l for l in lines if l.startswith("_namespace: "))
        auto_ns = ns_line.split(": ", 1)[1]
        assert len(auto_ns) == 6
        int(auto_ns, 16)

    def test_auto_namespace_deterministic(self):
        raw = '{{ yaml_include("ns_common.yaml") }}'

        r1 = render_template(raw, search_paths=[INCLUDES_DIR], seed=99)
        r2 = render_template(raw, search_paths=[INCLUDES_DIR], seed=99)

        assert r1 == r2

    def test_auto_namespace_varies_with_seed(self):
        raw = '{{ yaml_include("ns_common.yaml") }}'

        r1 = render_template(raw, search_paths=[INCLUDES_DIR], seed=1)
        r2 = render_template(raw, search_paths=[INCLUDES_DIR], seed=2)

        def extract_ns(text: str) -> str:
            return next(l for l in text.split("\n") if l.startswith("_namespace: ")).split(": ", 1)[1]

        assert extract_ns(r1) != extract_ns(r2)

    def test_included_content_rendered(self):
        raw = '{{ yaml_include("ns_common.yaml", "c") }}'

        result = render_template(raw, search_paths=[INCLUDES_DIR], seed=42)

        assert "photorealistic" in result
        assert "cinematic" in result

    def test_jinja_vars_inherited(self):
        raw = '{{ yaml_include("ns_common.yaml", "c") }}\ntheme: {{ mood }}\n'

        result = render_template(
            raw, search_paths=[INCLUDES_DIR], seed=42, jinja_vars={"mood": "dark"}
        )

        assert "theme: dark" in result

    def test_document_boundary_present(self):
        raw = '{{ yaml_include("ns_common.yaml", "c") }}\nmeta:\n  - detailed\n'

        result = render_template(raw, search_paths=[INCLUDES_DIR], seed=42)

        parts = result.split("---")
        assert len(parts) >= 2

    def test_nested_yaml_include(self):
        raw = '{{ yaml_include("ns_nested_outer.yaml", "outer") }}\nmeta:\n  - top\n'

        result = render_template(raw, search_paths=[INCLUDES_DIR], seed=42)

        assert "_namespace: inner" in result
        assert "_namespace: outer" in result
        assert "sharp focus" in result
        assert "dramatic shadows" in result
        assert "meta:" in result


class TestYamlIncludeRoundTrip:
    def _render_and_parse(self, raw, seed=42, jinja_vars=None):
        rendered = render_template(
            raw, search_paths=[INCLUDES_DIR], seed=seed, jinja_vars=jinja_vars,
        )
        docs = list(yaml.safe_load_all(rendered))
        merged, namespaces = merge_documents(docs)
        parser = YAMLPromptTemplateParser(seed=seed)
        return parser.parse_document(merged, namespaces=namespaces)

    def test_namespaced_sections_produce_blocks(self):
        raw = '{{ yaml_include("ns_common.yaml", "c") }}\nmeta:\n  - detailed\n'

        blocks = self._render_and_parse(raw)

        flat = [line for block in blocks for line in block]
        assert any("photorealistic" in l or "cinematic" in l for l in flat)
        assert any("detailed" in l for l in flat)

    def test_no_section_shadowing(self):
        raw = (
            '{{ yaml_include("ns_common.yaml", "c") }}\n'
            "style:\n"
            "  - impressionist\n"
        )

        blocks = self._render_and_parse(raw)

        flat = [line for block in blocks for line in block]
        assert any("photorealistic" in l for l in flat)
        assert any("impressionist" in l for l in flat)

    def test_scoped_vars_isolation(self):
        raw = (
            '{{ yaml_include("ns_with_vars.yaml", "themed") }}\n'
            "meta:\n"
            "  - $color background\n"
        )

        blocks = self._render_and_parse(raw)

        flat = [line for block in blocks for line in block]
        themed_lines = [l for l in flat if "red" in l or "dark" in l]
        assert len(themed_lines) >= 1
        meta_lines = [l for l in flat if "background" in l]
        assert meta_lines[0] == "$color background"

    def test_scoped_vars_inherit_global(self):
        raw = (
            "vars:\n"
            "  base: global_val\n"
            '{{ yaml_include("ns_with_vars.yaml", "themed") }}\n'
            "meta:\n"
            "  - $base check\n"
        )

        blocks = self._render_and_parse(raw)

        flat = [line for block in blocks for line in block]
        assert any("global_val" in l for l in flat)

    def test_backward_compat_no_namespace(self):
        raw = "meta:\n  - detailed\nstyle:\n  - photo\n"

        blocks = self._render_and_parse(raw)

        flat = [line for block in blocks for line in block]
        assert "detailed" in flat
        assert "photo" in flat

    def test_backward_compat_plain_include(self):
        raw = "{% include 'part.yaml' %}\nextra:\n  - bonus\n"

        blocks = self._render_and_parse(raw)

        flat = [line for block in blocks for line in block]
        assert any("included content" in l for l in flat)
        assert any("bonus" in l for l in flat)

    def test_multiple_namespaces_no_collision(self):
        raw = (
            '{{ yaml_include("ns_effects.yaml", "fx") }}\n'
            '{{ yaml_include("ns_with_vars.yaml", "themed") }}\n'
            "meta:\n"
            "  - base\n"
        )

        blocks = self._render_and_parse(raw)

        flat = [line for block in blocks for line in block]
        assert any("sparkles" in l or "glow" in l for l in flat)
        assert any("red" in l or "dark" in l for l in flat)
        assert any("base" in l for l in flat)

    def test_nested_yaml_include_round_trip(self):
        raw = '{{ yaml_include("ns_nested_outer.yaml", "outer") }}\nmeta:\n  - top\n'

        blocks = self._render_and_parse(raw)

        flat = [line for block in blocks for line in block]
        assert any("sharp focus" in l for l in flat)
        assert any("dramatic shadows" in l for l in flat)
        assert any("top" in l for l in flat)

    def test_namespaced_effects_with_block_template(self):
        raw = '{{ yaml_include("ns_effects.yaml", "fx") }}\nmeta:\n  - detailed\n'

        blocks = self._render_and_parse(raw)

        flat = [line for block in blocks for line in block]
        fx_block = [l for l in flat if "sparkles" in l or "glow" in l]
        assert len(fx_block) == 1
        assert fx_block[0].startswith("(")
        assert ":0.8)" in fx_block[0]

    def test_duplicate_namespace_in_pipeline_raises(self):
        raw = (
            '{{ yaml_include("ns_common.yaml", "dup") }}\n'
            '{{ yaml_include("ns_effects.yaml", "dup") }}\n'
            "meta:\n  - x\n"
        )

        rendered = render_template(raw, search_paths=[INCLUDES_DIR], seed=42)
        docs = list(yaml.safe_load_all(rendered))

        with pytest.raises(PipelineError, match="Duplicate namespace"):
            merge_documents(docs)


class TestScopedIncludeRNG:

    @staticmethod
    def _extract_child_content(rendered: str, namespace: str) -> dict | None:
        for doc in yaml.safe_load_all(rendered):
            if isinstance(doc, dict) and doc.get("_namespace") == namespace:
                return doc
        return None

    def test_same_child_from_different_parents(self):
        parent_a = (
            '{{ choice("x", "y", "z") }}\n'
            '{{ yaml_include("ns_random.yaml", "child") }}\n'
            "meta:\n  - a\n"
        )
        parent_b = (
            '{{ choice("1", "2") }}\n'
            '{{ choice("p", "q") }}\n'
            '{{ choice("j", "k", "l", "m") }}\n'
            '{{ yaml_include("ns_random.yaml", "child") }}\n'
            "meta:\n  - b\n"
        )

        rendered_a = render_template(parent_a, search_paths=[INCLUDES_DIR], seed=42)
        rendered_b = render_template(parent_b, search_paths=[INCLUDES_DIR], seed=42)

        child_a = self._extract_child_content(rendered_a, "child")
        child_b = self._extract_child_content(rendered_b, "child")

        assert child_a is not None
        assert child_a == child_b

    def test_duplicate_include_produces_identical_content(self):
        raw = (
            '{{ yaml_include("ns_random.yaml", "first") }}\n'
            '{{ choice("noise", "filler") }}\n'
            '{{ yaml_include("ns_random.yaml", "second") }}\n'
            "meta:\n  - x\n"
        )

        rendered = render_template(raw, search_paths=[INCLUDES_DIR], seed=42)

        first = self._extract_child_content(rendered, "first")
        second = self._extract_child_content(rendered, "second")

        assert first is not None
        assert second is not None
        first_content = {k: v for k, v in first.items() if k != "_namespace"}
        second_content = {k: v for k, v in second.items() if k != "_namespace"}
        assert first_content == second_content

    def test_nested_include_child_stable_across_parents(self):
        parent_a = (
            '{{ yaml_include("ns_random_outer.yaml", "outer") }}\n'
            "meta:\n  - a\n"
        )
        parent_b = (
            '{{ choice("a", "b", "c") }}\n'
            '{{ choice("d", "e", "f") }}\n'
            '{{ yaml_include("ns_random_outer.yaml", "outer") }}\n'
            "meta:\n  - b\n"
        )

        rendered_a = render_template(parent_a, search_paths=[INCLUDES_DIR], seed=42)
        rendered_b = render_template(parent_b, search_paths=[INCLUDES_DIR], seed=42)

        inner_a = self._extract_child_content(rendered_a, "rng_inner")
        inner_b = self._extract_child_content(rendered_b, "rng_inner")

        assert inner_a is not None
        assert inner_a == inner_b

    def test_no_seed_skips_scoping(self):
        raw = (
            '{{ yaml_include("ns_random.yaml", "child") }}\n'
            "meta:\n  - x\n"
        )

        results = set()
        for _ in range(30):
            rendered = render_template(raw, search_paths=[INCLUDES_DIR])
            child = self._extract_child_content(rendered, "child")
            results.add(str(child))

        assert len(results) > 1

    def test_different_seeds_produce_different_child(self):
        raw = '{{ yaml_include("ns_random.yaml", "child") }}\nmeta:\n  - x\n'

        results = set()
        for seed in range(20):
            rendered = render_template(raw, search_paths=[INCLUDES_DIR], seed=seed)
            child = self._extract_child_content(rendered, "child")
            results.add(str(child))

        assert len(results) > 1

    def test_different_files_get_different_scoped_rng(self):
        raw = (
            '{{ yaml_include("ns_random.yaml", "rng") }}\n'
            '{{ yaml_include("ns_common.yaml", "common") }}\n'
            "meta:\n  - x\n"
        )

        rendered = render_template(raw, search_paths=[INCLUDES_DIR], seed=42)

        rng_doc = self._extract_child_content(rendered, "rng")
        common_doc = self._extract_child_content(rendered, "common")

        assert rng_doc is not None
        assert common_doc is not None
        assert rng_doc != common_doc
