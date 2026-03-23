"""Tests for seed reproducibility guarantees."""

from pathlib import Path

from yaml_prompt.jinja_env import render_template

WILDCARDS_DIR = Path(__file__).resolve().parent / "fixtures" / "wildcards"


def test_jinja_and_yaml_wildcard_agree(make_parser):
    seed = 42
    jinja_result = render_template(
        "{{ wildcard('colors') }}",
        seed=seed,
        wildcard_dir=WILDCARDS_DIR,
    )

    parser = make_parser(seed=seed)
    blocks = parser.parse_document({"s": ["__colors__"]})
    yaml_result = blocks[0][0]

    assert jinja_result == yaml_result


def test_same_seed_same_output(make_parser):
    doc = {
        "s": {
            "values": [
                "{a|b|c}",
                {"choice": {"values": ["x", "y", "z"]}},
            ]
        }
    }
    outputs = [make_parser(seed=42).parse_document(doc) for _ in range(10)]

    assert all(o == outputs[0] for o in outputs)


def test_different_seed_different_output(make_parser):
    doc = {"s": {"values": ["{a|b|c|d|e|f|g|h|i|j}"]}}

    results = {
        str(make_parser(seed=s).parse_document(doc)) for s in range(20)
    }

    assert len(results) > 1


def test_no_seed_varies(make_parser):
    doc = {"s": {"values": ["{a|b|c|d|e|f|g|h|i|j}"]}}

    results = {str(make_parser().parse_document(doc)) for _ in range(20)}

    assert len(results) > 1


def test_chance_1_no_rng_consumption(make_parser):
    """BUG-4 regression: chance=1 must not consume RNG state."""
    doc_with_chance = {"s": {"chance": 1, "values": ["{a|b|c}"]}}
    doc_without_chance = {"s": {"values": ["{a|b|c}"]}}

    result_with = make_parser(seed=42).parse_document(doc_with_chance)
    result_without = make_parser(seed=42).parse_document(doc_without_chance)

    assert result_with == result_without


def test_wildcard_stable_across_rng_changes(make_parser):
    doc_simple = {"s": ["__colors__"]}
    doc_with_choice = {
        "pre": {"values": [{"choice": {"values": ["x", "y", "z"]}}]},
        "s": ["__colors__"],
    }

    blocks_simple = make_parser(seed=42).parse_document(doc_simple)
    blocks_with_choice = make_parser(seed=42).parse_document(doc_with_choice)

    assert blocks_simple[0][0] == blocks_with_choice[1][0]
