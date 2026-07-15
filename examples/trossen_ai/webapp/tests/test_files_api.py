from webapp.files_api import list_directory


def test_lists_subdirs_and_parent(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "file.txt").write_text("x")
    out = list_directory(str(tmp_path))
    names = [e["name"] for e in out["entries"]]
    assert names == ["a", "b"]  # dirs only, sorted; file.txt excluded
    assert out["parent"] == str(tmp_path.parent)
    assert out["path"] == str(tmp_path)


def test_marks_datasets(tmp_path):
    ds = tmp_path / "mydataset"
    (ds / "meta").mkdir(parents=True)
    (ds / "meta" / "info.json").write_text("{}")
    out = list_directory(str(tmp_path))
    entry = next(e for e in out["entries"] if e["name"] == "mydataset")
    assert entry["is_dataset"] is True


def test_error_for_missing_dir(tmp_path):
    out = list_directory(str(tmp_path / "nope"))
    assert "error" in out
