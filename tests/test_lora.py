from yaml_prompt.lora import extract_lora_tags, strip_lora_tags


def test_extract_no_tags():
    result = extract_lora_tags("a beautiful landscape")

    assert result == []


def test_extract_single_name_only():
    result = extract_lora_tags("text <lora:detail_v2>")

    assert result == [("detail_v2", 1.0, 1.0)]


def test_extract_single_one_weight():
    result = extract_lora_tags("text <lora:detail_v2:0.8>")

    assert result == [("detail_v2", 0.8, 0.8)]


def test_extract_single_two_weights():
    result = extract_lora_tags("text <lora:detail_v2:0.8:0.5>")

    assert result == [("detail_v2", 0.8, 0.5)]


def test_extract_multiple():
    result = extract_lora_tags("<lora:a:0.5> text <lora:b:0.7:0.3>")

    assert result == [("a", 0.5, 0.5), ("b", 0.7, 0.3)]


def test_extract_duplicate_keeps_first():
    result = extract_lora_tags("<lora:x:0.5> <lora:x:0.9>")

    assert result == [("x", 0.5, 0.5)]


def test_extract_path_in_name():
    result = extract_lora_tags("<lora:styles/anime_v2:0.8>")

    assert result == [("styles/anime_v2", 0.8, 0.8)]


def test_extract_invalid_weight_defaults():
    result = extract_lora_tags("<lora:x:abc>")

    assert result == [("x", 1.0, 1.0)]


def test_extract_empty_name_skipped():
    result = extract_lora_tags("<lora::0.5>")

    assert result == []


def test_extract_invalid_clip_weight_defaults():
    result = extract_lora_tags("<lora:x:0.5:bad>")

    assert result == [("x", 0.5, 1.0)]


def test_extract_extra_segments_ignored():
    result = extract_lora_tags("<lora:x:0.5:0.3:extra:stuff>")

    assert result == [("x", 0.5, 0.3)]


def test_strip_removes_tags():
    result = strip_lora_tags("hello <lora:x:0.5> world")

    assert result == "hello world"


def test_strip_multiple_tags():
    result = strip_lora_tags("a <lora:x:1> b <lora:y:2> c")

    assert result == "a b c"


def test_strip_no_tags():
    result = strip_lora_tags("no tags here")

    assert result == "no tags here"


def test_strip_tag_at_edges():
    result = strip_lora_tags("<lora:x:1>text<lora:y:1>")

    assert result == "text"


def test_strip_collapses_whitespace():
    result = strip_lora_tags("a  <lora:x:1>  b")

    assert result == "a b"


def test_extract_inf_weight_defaults():
    result = extract_lora_tags("<lora:x:inf>")

    assert result == [("x", 1.0, 1.0)]


def test_extract_nan_weight_defaults():
    result = extract_lora_tags("<lora:x:0.5:nan>")

    assert result == [("x", 0.5, 1.0)]
