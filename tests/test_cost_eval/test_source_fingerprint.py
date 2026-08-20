from cost_eval.source_fingerprint import python_source_tree_sha256


def test_source_fingerprint_is_path_independent_and_ignores_non_python(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for root in (first, second):
        (root / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (first / "ignored.txt").write_text("different", encoding="utf-8")
    assert python_source_tree_sha256(first) == python_source_tree_sha256(second)
