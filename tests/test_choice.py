import pytest

from yaml_prompt.pipeline import process_file


def _write_files(directory, contents):
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in contents.items():
        (directory / name).write_text(text, encoding="utf-8")


def test_choice_basic(parser):
    result = parser._resolve_item({"choice": {"values": ["a", "b"]}}, {})

    assert result in ("a", "b")


def test_one_of_alias(parser):
    result = parser._resolve_item({"oneOf": {"values": ["x", "y"]}}, {})

    assert result in ("x", "y")


def test_choice_weighted(make_parser):
    heavy_count = sum(
        1
        for seed in range(200)
        if make_parser(seed=seed)._resolve_item(
            {
                "choice": {
                    "values": [
                        {"name": "a", "weight": 10},
                        {"name": "b", "weight": 1},
                    ]
                }
            },
            {},
        )
        == "a"
    )

    assert heavy_count > 150


def test_choice_with_chance_zero(parser):
    result = parser._resolve_item({"choice": {"chance": 0, "values": ["a"]}}, {})

    assert result is None


def test_choice_chance_1(parser):
    result = parser._resolve_item({"choice": {"chance": 1, "values": ["a"]}}, {})

    assert result == "a"


def test_choice_with_template(parser):
    result = parser._resolve_item(
        {"choice": {"template": "($value:1.2)", "values": ["fire"]}}, {}
    )

    assert result == "(fire:1.2)"


def test_choice_option_with_chance_zero(parser):
    result = parser._resolve_item(
        {"choice": {"values": [{"name": "a", "chance": 0}]}}, {}
    )

    assert result is None


def test_choice_missing_values_key(parser):
    with pytest.raises(ValueError, match="choice/oneOf requires 'values'"):
        parser._resolve_item({"choice": {}}, {})


def test_choice_wrapper_shorthand(parser):
    result = parser._resolve_item({"choice": ["a", "b", "c"]}, {})

    assert result in ("a", "b", "c")


def test_choice_variables_in_options(parser):
    result = parser._resolve_item(
        {"choice": {"values": ["$color ball"]}}, {"color": "red"}
    )

    assert result == "red ball"


def test_choice_multi_key_dict(parser):
    result = parser._resolve_item(
        {"choice": True, "values": ["sword"], "template": "($value:1.2)"}, {}
    )

    assert result == "(sword:1.2)"


def test_named_item_with_chance_skip(parser):
    result = parser._resolve_item({"name": "hello", "chance": 0}, {})

    assert result is None


def test_named_item_without_chance(parser):
    result = parser._resolve_item({"name": "hello"}, {})

    assert result == "hello"


def test_nested_choice_basic(parser):
    result = parser._resolve_item(
        {
            "choice": {
                "values": [
                    {"choice": {"values": ["a", "b"]}},
                    {"choice": {"values": ["c", "d"]}},
                ]
            }
        },
        {},
    )

    assert result in ("a", "b", "c", "d")


def test_nested_oneof_alias(parser):
    result = parser._resolve_item(
        {
            "oneOf": {
                "values": [
                    {"oneOf": {"values": ["x", "y"]}},
                    {"oneOf": {"values": ["z"]}},
                ]
            }
        },
        {},
    )

    assert result in ("x", "y", "z")


def test_nested_choice_three_levels(parser):
    result = parser._resolve_item(
        {
            "choice": {
                "values": [
                    {
                        "choice": {
                            "values": [
                                {"choice": {"values": ["deep_a", "deep_b"]}},
                            ]
                        }
                    },
                ]
            }
        },
        {},
    )

    assert result in ("deep_a", "deep_b")


def test_nested_choice_with_weight(make_parser):
    heavy_count = sum(
        1
        for seed in range(200)
        if make_parser(seed=seed)._resolve_item(
            {
                "choice": {
                    "values": [
                        {"choice": {"values": ["a"]}, "weight": 10},
                        {"choice": {"values": ["b"]}, "weight": 1},
                    ]
                }
            },
            {},
        )
        == "a"
    )

    assert heavy_count > 150


def test_nested_choice_inner_chance(parser):
    result = parser._resolve_item(
        {
            "choice": {
                "values": [
                    {"choice": {"chance": 0, "values": ["never"]}},
                    "fallback",
                ]
            }
        },
        {},
    )

    assert result in (None, "fallback")


def test_nested_choice_inner_template(parser):
    result = parser._resolve_item(
        {
            "choice": {
                "values": [
                    {"choice": {"template": "($value:1.2)", "values": ["fire"]}},
                ]
            }
        },
        {},
    )

    assert result == "(fire:1.2)"


def test_nested_choice_all_skipped(parser):
    result = parser._resolve_item(
        {
            "choice": {
                "values": [
                    {"choice": {"chance": 0, "values": ["a"]}},
                    {"choice": {"chance": 0, "values": ["b"]}},
                ]
            }
        },
        {},
    )

    assert result is None


def test_nested_choice_depth_exceeded(parser):
    block = {"values": ["leaf"]}
    for _ in range(20):
        block = {"values": [{"choice": block}]}

    with pytest.raises(ValueError, match="Nested choice depth exceeded"):
        parser._resolve_item({"choice": block}, {})


def test_nested_choice_deterministic(make_parser):
    item = {
        "choice": {
            "values": [
                {"choice": {"values": ["a", "b", "c"]}},
                {"choice": {"values": ["x", "y", "z"]}},
            ]
        }
    }

    results = [make_parser(seed=42)._resolve_item(item, {}) for _ in range(10)]

    assert all(r == results[0] for r in results)


def test_nested_choice_with_variables(parser):
    result = parser._resolve_item(
        {
            "choice": {
                "values": [
                    {"choice": {"values": ["$color ball"]}},
                ]
            }
        },
        {"color": "red"},
    )

    assert result == "red ball"


def test_nested_choice_in_section(parser):
    section = {
        "values": [
            "intro",
            {
                "choice": {
                    "values": [
                        {"choice": {"values": ["inner_a", "inner_b"]}},
                        "flat_option",
                    ]
                }
            },
        ],
    }

    result = parser._parse_section(section, {})

    assert len(result) >= 1
    joined = " ".join(result)
    assert "intro" in joined
    assert any(v in joined for v in ("inner_a", "inner_b", "flat_option"))


def test_choice_files_reads_file_relative_to_base_dir(make_parser, tmp_path):
    _write_files(tmp_path / "poses", {"sitting.txt": "  sitting on a chair\n"})
    parser = make_parser(seed=42, base_dir=tmp_path)

    result = parser._resolve_item(
        {"choice": {"files": "poses/*.txt", "template": "pose: $value"}}, {}
    )

    assert result == "pose: sitting on a chair"


def test_choice_files_picks_one_of_matching_files(make_parser, tmp_path):
    _write_files(tmp_path, {"a.txt": "alpha", "b.txt": "beta", "c.md": "gamma"})

    results = {
        make_parser(seed=seed, base_dir=tmp_path)._resolve_item(
            {"choice": {"files": "*.txt"}}, {}
        )
        for seed in range(50)
    }

    assert results == {"alpha", "beta"}


def test_choice_files_same_seed_same_file_in_any_location(make_parser, tmp_path):
    """Selection depends on paths relative to base_dir, not absolute ones."""
    contents = {f"{name}.txt": name for name in ("one", "two", "three", "four")}
    _write_files(tmp_path / "home_a" / "parts", contents)
    _write_files(tmp_path / "elsewhere" / "home_b" / "parts", contents)
    item = {"choice": {"files": "parts/*.txt"}}

    picks_a = [
        make_parser(seed=seed, base_dir=tmp_path / "home_a")._resolve_item(item, {})
        for seed in range(30)
    ]
    picks_b = [
        make_parser(
            seed=seed, base_dir=tmp_path / "elsewhere" / "home_b"
        )._resolve_item(item, {})
        for seed in range(30)
    ]

    assert picks_a == picks_b


def test_choice_files_no_match_returns_none(make_parser, tmp_path):
    parser = make_parser(seed=42, base_dir=tmp_path)

    result = parser._resolve_item({"choice": {"files": "missing/*.txt"}}, {})

    assert result is None


def test_choice_files_with_values_raises(make_parser, tmp_path):
    parser = make_parser(seed=42, base_dir=tmp_path)

    with pytest.raises(ValueError, match="not both"):
        parser._resolve_item({"choice": {"files": "*.txt", "values": ["a"]}}, {})


def test_choice_files_non_string_raises(make_parser, tmp_path):
    parser = make_parser(seed=42, base_dir=tmp_path)

    with pytest.raises(TypeError, match="must be a glob string"):
        parser._resolve_item({"choice": {"files": ["a/*.txt", "b/*.txt"]}}, {})


def test_process_file_resolves_files_next_to_yaml(tmp_path, monkeypatch):
    project = tmp_path / "project"
    _write_files(project / "poses", {"standing.txt": "standing tall"})
    (project / "template.yaml").write_text(
        "pose:\n  values:\n    - choice:\n        files: poses/*.txt\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = process_file(project / "template.yaml", seed=42)

    assert result.prompt == "standing tall"
