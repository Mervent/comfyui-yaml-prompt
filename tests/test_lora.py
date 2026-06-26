from yaml_prompt.lora import LoraEntry, extract_lora_tags, extract_lora_tags_lbw, strip_lora_tags


def test_extract_no_tags():
    result = extract_lora_tags("a beautiful landscape")

    assert result == []


def test_extract_single_name_only():
    result = extract_lora_tags("text <lora:detail_v2>")

    assert result == [("detail_v2.safetensors", 1.0, 1.0)]


def test_extract_single_one_weight():
    result = extract_lora_tags("text <lora:detail_v2:0.8>")

    assert result == [("detail_v2.safetensors", 0.8, 0.8)]


def test_extract_single_two_weights():
    result = extract_lora_tags("text <lora:detail_v2:0.8:0.5>")

    assert result == [("detail_v2.safetensors", 0.8, 0.5)]


def test_extract_multiple():
    result = extract_lora_tags("<lora:a:0.5> text <lora:b:0.7:0.3>")

    assert result == [("a.safetensors", 0.5, 0.5), ("b.safetensors", 0.7, 0.3)]


def test_extract_duplicate_keeps_first():
    result = extract_lora_tags("<lora:x:0.5> <lora:x:0.9>")

    assert result == [("x.safetensors", 0.5, 0.5)]


def test_extract_path_in_name():
    result = extract_lora_tags("<lora:styles/anime_v2:0.8>")

    assert result == [("styles/anime_v2.safetensors", 0.8, 0.8)]


def test_extract_invalid_weight_defaults():
    result = extract_lora_tags("<lora:x:abc>")

    assert result == [("x.safetensors", 1.0, 1.0)]


def test_extract_empty_name_skipped():
    result = extract_lora_tags("<lora::0.5>")

    assert result == []


def test_extract_invalid_clip_weight_defaults():
    result = extract_lora_tags("<lora:x:0.5:bad>")

    assert result == [("x.safetensors", 0.5, 1.0)]


def test_extract_extra_segments_ignored():
    result = extract_lora_tags("<lora:x:0.5:0.3:extra:stuff>")

    assert result == [("x.safetensors", 0.5, 0.3)]


def test_extract_preserves_existing_extension():
    result = extract_lora_tags("<lora:my_model.safetensors:0.8>")

    assert result == [("my_model.safetensors", 0.8, 0.8)]


def test_extract_preserves_ckpt_extension():
    result = extract_lora_tags("<lora:old_model.ckpt:0.5>")

    assert result == [("old_model.ckpt", 0.5, 0.5)]


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

    assert result == [("x.safetensors", 1.0, 1.0)]


def test_extract_nan_weight_defaults():
    result = extract_lora_tags("<lora:x:0.5:nan>")

    assert result == [("x.safetensors", 0.5, 1.0)]


def test_lbw_basic():
    result = extract_lora_tags_lbw("text <lora:detail:0.8:LBW=SD-ALL>")

    assert result == [LoraEntry("detail.safetensors", 0.8, 0.8, lbw="SD-ALL")]


def test_lbw_with_a_b():
    result = extract_lora_tags_lbw("<lora:sle:0.7:LBW=SD-MIDD:A=0.5:B=0.3>")

    assert result == [
        LoraEntry("sle.safetensors", 0.7, 0.7, lbw="SD-MIDD", lbw_a=0.5, lbw_b=0.3)
    ]


def test_lbw_two_weights_with_lbw():
    result = extract_lora_tags_lbw("<lora:x:0.8:0.5:LBW=SD-ALL>")

    assert result == [
        LoraEntry("x.safetensors", 0.8, 0.5, lbw="SD-ALL")
    ]


def test_lbw_no_lbw_fields():
    result = extract_lora_tags_lbw("text <lora:detail:0.8>")

    assert result == [LoraEntry("detail.safetensors", 0.8, 0.8)]


def test_lbw_no_tags():
    result = extract_lora_tags_lbw("just text")

    assert result == []


def test_lbw_empty_name_skipped():
    result = extract_lora_tags_lbw("<lora::0.5:LBW=SD-ALL>")

    assert result == []


def test_lbw_duplicate_keeps_first():
    result = extract_lora_tags_lbw("<lora:x:0.5:LBW=SD-ALL> <lora:x:0.9>")

    assert result == [
        LoraEntry("x.safetensors", 0.5, 0.5, lbw="SD-ALL")
    ]


def test_lbw_preserves_existing_extension():
    result = extract_lora_tags_lbw("<lora:my.safetensors:0.8:LBW=SD-ALL>")

    assert result == [
        LoraEntry("my.safetensors", 0.8, 0.8, lbw="SD-ALL")
    ]


def test_lbw_multiple_entries():
    result = extract_lora_tags_lbw(
        "<lora:a:0.5:LBW=SD-ALL> text <lora:b:0.7>"
    )

    assert result == [
        LoraEntry("a.safetensors", 0.5, 0.5, lbw="SD-ALL"),
        LoraEntry("b.safetensors", 0.7, 0.7),
    ]


def test_lbw_a_only():
    result = extract_lora_tags_lbw("<lora:x:0.8:LBW=SD-ALL:A=2.0>")

    assert result == [
        LoraEntry("x.safetensors", 0.8, 0.8, lbw="SD-ALL", lbw_a=2.0)
    ]


def test_lbw_name_only():
    result = extract_lora_tags_lbw("<lora:x>")

    assert result == [LoraEntry("x.safetensors", 1.0, 1.0)]
