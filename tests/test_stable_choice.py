"""Tests for stable (SHA-256 keyed) choice selection across independent renders."""

from yaml_prompt.jinja_env import render_template
from yaml_prompt.parser import YAMLPromptTemplateParser

from conftest import WILDCARDS_DIR


SEED = 42


# --- Jinja choice() stable across independent renders ---


def test_jinja_choice_stable_across_renders():
    raw = "{{ choice('happy', 'sad', 'angry') }}"

    r1 = render_template(raw, seed=SEED)
    r2 = render_template(raw, seed=SEED)

    assert r1 == r2
    assert r1 in ("happy", "sad", "angry")


def test_jinja_choice_stable_despite_preceding_rng_calls():
    bare = "{{ choice('happy', 'sad', 'angry') }}"
    with_rand = "{{ rand(0, 1) }}\n{{ choice('happy', 'sad', 'angry') }}"

    result_bare = render_template(bare, seed=SEED)
    result_with_rand = render_template(with_rand, seed=SEED).split("\n")[1]

    assert result_bare == result_with_rand


def test_jinja_choice_different_seeds_can_differ():
    raw = "{{ choice('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j') }}"

    results = {render_template(raw, seed=s) for s in range(50)}

    assert len(results) > 1


def test_jinja_choice_no_seed_uses_rng():
    raw = "{{ choice('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j') }}"

    results = {render_template(raw) for _ in range(30)}

    assert len(results) > 1


# --- Jinja weighted_choice() stable ---


def test_jinja_weighted_choice_stable():
    raw = "{{ weighted_choice([['a', 10], ['b', 1]]) }}"

    r1 = render_template(raw, seed=SEED)
    r2 = render_template(raw, seed=SEED)

    assert r1 == r2


def test_jinja_weighted_choice_respects_weights():
    raw = "{{ weighted_choice([['heavy', 100], ['light', 1]]) }}"

    heavy_count = sum(
        1 for s in range(200)
        if render_template(raw, seed=s) == "heavy"
    )

    assert heavy_count > 160


# --- Parser brace {a|b|c} stable ---


def test_brace_choice_stable_across_parsers(make_parser):
    doc = {"s": {"values": ["{a|b|c}"]}}

    r1 = make_parser(seed=SEED).parse_document(doc)
    r2 = make_parser(seed=SEED).parse_document(doc)

    assert r1 == r2


def test_brace_choice_stable_despite_preceding_rng(make_parser):
    doc_bare = {"s": {"values": ["{happy|sad|angry}"]}}
    doc_with_chance = {
        "pre": {"chance": 0.99, "values": ["filler"]},
        "s": {"values": ["{happy|sad|angry}"]},
    }

    bare = make_parser(seed=SEED).parse_document(doc_bare)
    with_prefix = make_parser(seed=SEED).parse_document(doc_with_chance)

    bare_choice = bare[0][0]
    prefixed_choice = with_prefix[-1][0]

    assert bare_choice == prefixed_choice


def test_brace_weighted_stable(make_parser):
    doc = {"s": {"values": ["{0.9::heavy|0.1::light}"]}}

    r1 = make_parser(seed=SEED).parse_document(doc)
    r2 = make_parser(seed=SEED).parse_document(doc)

    assert r1 == r2


def test_brace_weighted_respects_weights(make_parser):
    heavy_count = sum(
        1 for s in range(200)
        if make_parser(seed=s).expand_string("{100::heavy|1::light}", {}) == "heavy"
    )

    assert heavy_count > 160


# --- Parser choice/oneOf blocks stable ---


def test_choice_block_stable_across_parsers(make_parser):
    doc = {"s": {"values": [{"choice": {"values": ["x", "y", "z"]}}]}}

    r1 = make_parser(seed=SEED).parse_document(doc)
    r2 = make_parser(seed=SEED).parse_document(doc)

    assert r1 == r2


def test_choice_block_stable_despite_preceding_rng(make_parser):
    block = {"choice": {"values": ["x", "y", "z"]}}
    doc_bare = {"s": {"values": [block]}}
    doc_with_prefix = {
        "filler": {"values": ["{a|b|c|d|e}"]},
        "s": {"values": [block]},
    }

    bare = make_parser(seed=SEED).parse_document(doc_bare)
    with_prefix = make_parser(seed=SEED).parse_document(doc_with_prefix)

    assert bare[0][0] == with_prefix[-1][0]


# --- Cross-layer agreement: Jinja choice() == parser {a|b|c} ---


def test_jinja_choice_agrees_with_parser_brace(make_parser):
    items = ("alpha", "beta", "gamma")

    jinja_result = render_template(
        "{{ choice('alpha', 'beta', 'gamma') }}", seed=SEED
    )

    parser_result = make_parser(seed=SEED).expand_string(
        "{alpha|beta|gamma}", {}
    )

    assert jinja_result == parser_result


def test_jinja_choice_agrees_with_parser_choice_block(make_parser):
    jinja_result = render_template(
        "{{ choice('x', 'y', 'z') }}", seed=SEED
    )

    parser = make_parser(seed=SEED)
    parser_result = parser._resolve_choice(
        {"values": ["x", "y", "z"]}, {}
    )

    assert jinja_result == parser_result


# --- Same choice in two "separate nodes" simulation ---


def test_same_choice_in_independent_templates():
    face_template = (
        "face:\n"
        "  - {{ choice('smiling', 'frowning', 'surprised') }} expression\n"
        "  - detailed portrait\n"
    )
    general_template = (
        "body:\n"
        "  - full body\n"
        "  - {{ choice('smiling', 'frowning', 'surprised') }} expression\n"
    )

    face_rendered = render_template(face_template, seed=SEED)
    general_rendered = render_template(general_template, seed=SEED)

    face_expr = [l for l in face_rendered.split("\n") if "expression" in l][0]
    general_expr = [l for l in general_rendered.split("\n") if "expression" in l][0]

    assert face_expr.strip().lstrip("- ") == general_expr.strip().lstrip("- ")


def test_same_brace_in_independent_parsers(make_parser):
    face_doc = {
        "face": ["detailed portrait"],
        "face_expr": ["{smiling|frowning|surprised} expression"],
    }
    general_doc = {
        "body": ["full body"],
        "expr": ["{smiling|frowning|surprised} expression"],
    }

    face_blocks = make_parser(seed=SEED).parse_document(face_doc)
    general_blocks = make_parser(seed=SEED).parse_document(general_doc)

    face_lines = [l for b in face_blocks for l in b if "expression" in l]
    general_lines = [l for b in general_blocks for l in b if "expression" in l]

    assert face_lines[0] == general_lines[0]
