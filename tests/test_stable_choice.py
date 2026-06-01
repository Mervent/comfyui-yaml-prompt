import random

from yaml_prompt.jinja_env import render_template
from yaml_prompt.pipeline import process_file


SEED = 42


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


def test_jinja_weighted_choice_stable():
    raw = "{{ weighted_choice([['a', 10], ['b', 1]]) }}"

    r1 = render_template(raw, seed=SEED)
    r2 = render_template(raw, seed=SEED)

    assert r1 == r2


def test_jinja_weighted_choice_respects_weights():
    raw = "{{ weighted_choice([['heavy', 100], ['light', 1]]) }}"

    heavy_count = sum(1 for s in range(200) if render_template(raw, seed=s) == "heavy")

    assert heavy_count > 160


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
        1
        for s in range(200)
        if make_parser(seed=s).expand_string("{100::heavy|1::light}", {}) == "heavy"
    )

    assert heavy_count > 160


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


def test_jinja_choice_agrees_with_parser_brace(make_parser):
    items = ("alpha", "beta", "gamma")

    jinja_result = render_template("{{ choice('alpha', 'beta', 'gamma') }}", seed=SEED)

    parser_result = make_parser(seed=SEED).expand_string("{alpha|beta|gamma}", {})

    assert jinja_result == parser_result


def test_jinja_choice_agrees_with_parser_choice_block(make_parser):
    jinja_result = render_template("{{ choice('x', 'y', 'z') }}", seed=SEED)

    parser = make_parser(seed=SEED)
    parser_result = parser._choices.resolve({"values": ["x", "y", "z"]}, {})

    assert jinja_result == parser_result


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


def test_section_chance_stable_across_parsers(make_parser):
    section = {"chance": 0.5, "values": ["blood and gore"]}
    doc_bare = {"violence": section}
    doc_with_prefix = {
        "meta": ["ultra-detailed", "masterpiece"],
        "violence": section,
    }

    bare = make_parser(seed=SEED).parse_document(doc_bare)
    with_prefix = make_parser(seed=SEED).parse_document(doc_with_prefix)

    bare_has = len(bare) > 0 and any("blood" in l for b in bare for l in b)
    prefix_has = any("blood" in l for b in with_prefix for l in b)

    assert bare_has == prefix_has


def test_named_item_chance_stable_across_parsers(make_parser):
    item = {"name": "dramatic volumetric lighting", "chance": 0.5}
    doc_bare = {"effects": {"values": [item]}}
    doc_with_prefix = {
        "meta": ["ultra-detailed"],
        "filler": {"values": ["{a|b|c|d|e}"]},
        "effects": {"values": [item]},
    }

    bare = make_parser(seed=SEED).parse_document(doc_bare)
    with_prefix = make_parser(seed=SEED).parse_document(doc_with_prefix)

    bare_has = any("dramatic" in l for b in bare for l in b)
    prefix_has = any("dramatic" in l for b in with_prefix for l in b)

    assert bare_has == prefix_has


def test_choice_block_chance_stable_across_parsers(make_parser):
    choice_block = {"choice": {"chance": 0.5, "values": ["depth-of-field", "bokeh"]}}
    doc_bare = {"effects": {"values": [choice_block]}}
    doc_with_prefix = {
        "meta": ["ultra-detailed"],
        "effects": {"values": [choice_block]},
    }

    bare = make_parser(seed=SEED).parse_document(doc_bare)
    with_prefix = make_parser(seed=SEED).parse_document(doc_with_prefix)

    bare_has = any("depth" in l or "bokeh" in l for b in bare for l in b)
    prefix_has = any("depth" in l or "bokeh" in l for b in with_prefix for l in b)

    assert bare_has == prefix_has


def test_option_chance_stable_across_parsers(make_parser):
    choice_block = {
        "choice": {
            "values": [
                {"name": "blood", "chance": 0.5},
                "wounded",
            ]
        }
    }
    doc_bare = {"violence": {"values": [choice_block]}}
    doc_with_prefix = {
        "meta": ["filler"],
        "violence": {"values": [choice_block]},
    }

    for s in range(50):
        bare = make_parser(seed=s).parse_document(doc_bare)
        with_prefix = make_parser(seed=s).parse_document(doc_with_prefix)

        bare_violence = bare[-1]
        prefix_violence = with_prefix[-1]
        assert bare_violence == prefix_violence, f"seed={s}"


def test_chance_distribution_holds(make_parser):
    doc = {"s": {"chance": 0.25, "values": ["content"]}}

    pass_count = sum(1 for s in range(500) if make_parser(seed=s).parse_document(doc))

    assert 75 < pass_count < 175


def test_two_yaml_files_same_choices_and_chances(tmp_path):
    face_yaml = tmp_path / "face.yaml"
    face_yaml.write_text(
        "face:\n"
        "  - detailed portrait\n"
        "expression:\n"
        "  - '{smiling|frowning|surprised} expression'\n"
        "effects:\n"
        "  chance: 0.5\n"
        "  values:\n"
        "    - dramatic lighting\n"
    )

    general_yaml = tmp_path / "general.yaml"
    general_yaml.write_text(
        "body:\n"
        "  - full body shot\n"
        "  - cinematic\n"
        "expression:\n"
        "  - '{smiling|frowning|surprised} expression'\n"
        "effects:\n"
        "  chance: 0.5\n"
        "  values:\n"
        "    - dramatic lighting\n"
    )

    face_result = process_file(face_yaml, seed=SEED)
    general_result = process_file(general_yaml, seed=SEED)

    face_blocks = [l for b in face_result.blocks for l in b]
    general_blocks = [l for b in general_result.blocks for l in b]

    face_expr = [l for l in face_blocks if "expression" in l][0]
    general_expr = [l for l in general_blocks if "expression" in l][0]
    assert face_expr == general_expr

    face_has_effects = any("dramatic" in l for l in face_blocks)
    general_has_effects = any("dramatic" in l for l in general_blocks)
    assert face_has_effects == general_has_effects


def test_dict_chance_value_only_matches_float_section(make_parser):
    doc_float = {"s": {"chance": 0.5, "values": ["x"]}}
    doc_dict = {"s": {"chance": {"value": 0.5}, "values": ["x"]}}

    for s in range(50):
        float_result = make_parser(seed=s).parse_document(doc_float)
        dict_result = make_parser(seed=s).parse_document(doc_dict)
        assert float_result == dict_result, f"seed={s}"


def test_dict_chance_value_only_matches_float_item(make_parser):
    doc_float = {"s": {"values": [{"name": "x", "chance": 0.5}]}}
    doc_dict = {"s": {"values": [{"name": "x", "chance": {"value": 0.5}}]}}

    for s in range(50):
        float_result = make_parser(seed=s).parse_document(doc_float)
        dict_result = make_parser(seed=s).parse_document(doc_dict)
        assert float_result == dict_result, f"seed={s}"


def test_dict_chance_value_only_matches_float_choice(make_parser):
    doc_float = {"s": {"values": [{"choice": {"chance": 0.5, "values": ["a", "b"]}}]}}
    doc_dict = {
        "s": {"values": [{"choice": {"chance": {"value": 0.5}, "values": ["a", "b"]}}]}
    }

    for s in range(50):
        float_result = make_parser(seed=s).parse_document(doc_float)
        dict_result = make_parser(seed=s).parse_document(doc_dict)
        assert float_result == dict_result, f"seed={s}"


def test_dict_chance_value_only_matches_float_option(make_parser):
    doc_float = {
        "s": {"values": [{"choice": {"values": [{"name": "a", "chance": 0.5}, "b"]}}]}
    }
    doc_dict = {
        "s": {
            "values": [
                {"choice": {"values": [{"name": "a", "chance": {"value": 0.5}}, "b"]}}
            ]
        }
    }

    for s in range(50):
        float_result = make_parser(seed=s).parse_document(doc_float)
        dict_result = make_parser(seed=s).parse_document(doc_dict)
        assert float_result == dict_result, f"seed={s}"


def test_custom_key_links_different_content(make_parser):
    doc_a = {
        "s": {
            "chance": {"value": 0.5, "key": "lighting"},
            "values": ["rim light on face"],
        }
    }
    doc_b = {
        "s": {
            "chance": {"value": 0.5, "key": "lighting"},
            "values": ["volumetric full-scene light"],
        }
    }

    for s in range(50):
        a = make_parser(seed=s).parse_document(doc_a)
        b = make_parser(seed=s).parse_document(doc_b)
        a_has = len(a) > 0 and any(b2 for b2 in a)
        b_has = len(b) > 0 and any(b2 for b2 in b)
        assert a_has == b_has, f"seed={s}"


def test_custom_key_stable_across_parsers(make_parser):
    section = {
        "chance": {"value": 0.5, "key": "fx_gate"},
        "values": ["depth-of-field"],
    }
    doc_bare = {"effects": section}
    doc_with_prefix = {
        "meta": ["ultra-detailed", "masterpiece"],
        "effects": section,
    }

    for s in range(50):
        bare = make_parser(seed=s).parse_document(doc_bare)
        prefix = make_parser(seed=s).parse_document(doc_with_prefix)
        bare_has = any("depth" in ln for b in bare for ln in b)
        prefix_has = any("depth" in ln for b in prefix for ln in b)
        assert bare_has == prefix_has, f"seed={s}"


def test_custom_key_different_keys_can_differ(make_parser):
    doc_a = {"s": {"chance": {"value": 0.5, "key": "key_alpha"}, "values": ["x"]}}
    doc_b = {"s": {"chance": {"value": 0.5, "key": "key_beta"}, "values": ["x"]}}

    differ_count = sum(
        1
        for s in range(200)
        if bool(make_parser(seed=s).parse_document(doc_a))
        != bool(make_parser(seed=s).parse_document(doc_b))
    )

    assert differ_count > 20


def test_unstable_chance_varies_across_runs(make_parser):
    doc = {"s": {"chance": {"value": 0.5, "stable": False}, "values": ["x"]}}

    results = set()
    for _ in range(50):
        blocks = make_parser(seed=42).parse_document(doc)
        has_x = len(blocks) > 0 and any("x" in ln for b in blocks for ln in b)
        results.add(has_x)

    assert len(results) == 2


def test_unstable_chance_distribution(make_parser):
    doc = {"s": {"chance": {"value": 0.25, "stable": False}, "values": ["x"]}}

    pass_count = sum(1 for _ in range(500) if make_parser(seed=42).parse_document(doc))

    assert 75 < pass_count < 175


def test_unstable_chance_1_no_side_effects(make_parser):
    doc_with = {"s": {"chance": {"value": 1, "stable": False}, "values": ["{a|b|c}"]}}
    doc_without = {"s": {"values": ["{a|b|c}"]}}

    result_with = make_parser(seed=42).parse_document(doc_with)
    result_without = make_parser(seed=42).parse_document(doc_without)

    assert result_with == result_without


def test_unstable_chance_key_ignored(make_parser):
    doc = {
        "s": {
            "chance": {"value": 0.5, "stable": False, "key": "should_be_ignored"},
            "values": ["x"],
        }
    }

    results = set()
    for _ in range(50):
        blocks = make_parser(seed=42).parse_document(doc)
        has_x = len(blocks) > 0 and any("x" in ln for b in blocks for ln in b)
        results.add(has_x)

    assert len(results) == 2


def test_dict_chance_zero_always_skips(make_parser):
    doc = {"s": {"chance": {"value": 0}, "values": ["x"]}}

    for s in range(20):
        assert make_parser(seed=s).parse_document(doc) == []


def test_dict_chance_one_always_passes(make_parser):
    doc = {"s": {"chance": {"value": 1}, "values": ["x"]}}

    for s in range(20):
        blocks = make_parser(seed=s).parse_document(doc)
        assert any("x" in ln for b in blocks for ln in b)


def test_choice_key_links_across_sections(make_parser):
    """Two choice blocks with same key and same-length values pick same index."""
    doc = {
        "weapon": {
            "values": [
                {"choice": {"key": "loadout", "values": ["sword", "axe", "spear"]}}
            ]
        },
        "armor": {
            "values": [
                {"choice": {"key": "loadout", "values": ["light", "medium", "heavy"]}}
            ]
        },
    }
    mapping = {"sword": "light", "axe": "medium", "spear": "heavy"}

    for s in range(50):
        blocks = make_parser(seed=s).parse_document(doc)
        weapon = blocks[0][0]
        armor = blocks[1][0]
        assert armor == mapping[weapon], f"seed={s}: {weapon} → {armor}"


def test_choice_key_stable_across_parsers(make_parser):
    block = {"choice": {"key": "fx", "values": ["bloom", "blur", "glow"]}}
    doc_bare = {"fx": {"values": [block]}}
    doc_with_prefix = {"meta": ["filler"], "fx": {"values": [block]}}

    for s in range(50):
        bare = make_parser(seed=s).parse_document(doc_bare)
        prefixed = make_parser(seed=s).parse_document(doc_with_prefix)
        assert bare[-1] == prefixed[-1], f"seed={s}"


def test_choice_key_different_keys_can_differ(make_parser):
    doc_a = {
        "s": {
            "values": [
                {"choice": {"key": "alpha", "values": ["a", "b", "c", "d", "e"]}}
            ]
        }
    }
    doc_b = {
        "s": {
            "values": [{"choice": {"key": "beta", "values": ["a", "b", "c", "d", "e"]}}]
        }
    }

    differ = sum(
        1
        for s in range(200)
        if make_parser(seed=s).parse_document(doc_a)
        != make_parser(seed=s).parse_document(doc_b)
    )

    assert differ > 20


def test_choice_unstable_varies_across_runs(make_parser):
    doc = {
        "s": {
            "values": [
                {"choice": {"stable": False, "values": ["a", "b", "c", "d", "e"]}}
            ]
        }
    }

    results = set()
    for _ in range(50):
        blocks = make_parser(seed=42).parse_document(doc)
        results.add(blocks[0][0])

    assert len(results) > 1


def test_choice_unstable_no_side_effects(make_parser):
    """stable: true explicitly should behave identically to no key at all."""
    doc_with = {"s": {"values": [{"choice": {"stable": True, "values": ["{a|b|c}"]}}]}}
    doc_without = {"s": {"values": [{"choice": {"values": ["{a|b|c}"]}}]}}

    for s in range(20):
        assert make_parser(seed=s).parse_document(doc_with) == make_parser(
            seed=s
        ).parse_document(doc_without), f"seed={s}"


def test_choice_unstable_ignores_global_random_seed(make_parser):
    doc = {
        "s": {
            "values": [
                {"choice": {"stable": False, "values": ["a", "b", "c", "d", "e"]}}
            ]
        }
    }

    results = set()
    for _ in range(50):
        random.seed(42)
        blocks = make_parser(seed=42).parse_document(doc)
        results.add(blocks[0][0])

    assert len(results) > 1


def test_chance_unstable_ignores_global_random_seed(make_parser):
    doc = {"s": {"chance": {"value": 0.5, "stable": False}, "values": ["x"]}}

    results = set()
    for _ in range(50):
        random.seed(42)
        blocks = make_parser(seed=42).parse_document(doc)
        has_x = len(blocks) > 0 and any("x" in ln for b in blocks for ln in b)
        results.add(has_x)

    assert len(results) == 2
