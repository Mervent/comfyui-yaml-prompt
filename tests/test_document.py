def test_empty_document(parser):
    blocks = parser.parse_document({})

    assert blocks == []


def test_vars_section_skipped(parser):
    blocks = parser.parse_document({"vars": {"a": "b"}, "section": ["x"]})

    assert len(blocks) == 1
    assert blocks[0] == ["x"]


def test_multiple_sections(parser):
    blocks = parser.parse_document({"s1": ["a"], "s2": ["b"], "s3": ["c"]})

    assert len(blocks) == 3


def test_global_vars_shared(parser):
    doc = {"vars": {"x": "hi"}, "s1": ["$x"], "s2": ["$x"]}

    blocks = parser.parse_document(doc)

    assert blocks[0] == ["hi"]
    assert blocks[1] == ["hi"]


def test_full_document_integration(parser):
    doc = {
        "vars": {"mood": "fierce", "animal": "__animals__"},
        "meta": ["detailed", "masterpiece"],
        "subject": {
            "template": "($value:1.2)",
            "values": [
                "$mood warrior",
                {"choice": {"values": ["sword", "axe", "spear"]}},
            ],
        },
        "companion": ["a $animal companion"],
    }

    blocks = parser.parse_document(doc)

    assert len(blocks) == 3
    assert blocks[0] == ["detailed, masterpiece"]

    subject_line = blocks[1][0]
    assert subject_line.startswith("(fierce warrior, ")
    assert subject_line.endswith(":1.2)")

    companion_line = blocks[2][0]
    assert "companion" in companion_line
    assert any(a in companion_line for a in ("cat", "dog", "bird"))


def test_section_ordering_preserved(parser):
    blocks = parser.parse_document({"first": ["a"], "second": ["b"], "third": ["c"]})

    assert blocks == [["a"], ["b"], ["c"]]


def test_rand_function_in_vars(parser):
    doc = {"vars": {"w": "rand(0.0, 1.0)"}, "s": ["weight is $w"]}

    blocks = parser.parse_document(doc)

    text = blocks[0][0]
    assert text.startswith("weight is ")
    w_val = float(text.removeprefix("weight is "))
    assert 0.0 <= w_val <= 1.0
