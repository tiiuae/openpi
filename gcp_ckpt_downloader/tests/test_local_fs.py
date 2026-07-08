from __future__ import annotations

from gcp_ckpt_downloader import local_fs

# ---------------------------------------------------------------------------
# list_directory
# ---------------------------------------------------------------------------


def test_list_directory_lists_only_dirs_sorted_by_name(tmp_path):
    (tmp_path / "b_dir").mkdir()
    (tmp_path / "a_dir").mkdir()
    (tmp_path / "a_file.txt").write_text("not a dir")

    result = local_fs.list_directory(str(tmp_path))

    assert result["path"] == str(tmp_path)
    assert result["parent"] == str(tmp_path.parent)
    assert result["entries"] == [
        {"name": "a_dir", "type": "dir", "path": str(tmp_path / "a_dir")},
        {"name": "b_dir", "type": "dir", "path": str(tmp_path / "b_dir")},
    ]


def test_list_directory_expands_user_and_resolves(tmp_path, monkeypatch):
    (tmp_path / "sub").mkdir()
    # Simulate "~" expansion by pointing HOME at tmp_path.
    monkeypatch.setenv("HOME", str(tmp_path))

    result = local_fs.list_directory("~")

    assert result["path"] == str(tmp_path)
    assert [e["name"] for e in result["entries"]] == ["sub"]


def test_list_directory_returns_error_for_missing_path(tmp_path):
    missing = tmp_path / "does_not_exist"

    result = local_fs.list_directory(str(missing))

    assert result == {"error": f"not a directory: {missing}"}


def test_list_directory_returns_error_for_a_file_not_a_directory(tmp_path):
    f = tmp_path / "some_file.txt"
    f.write_text("hi")

    result = local_fs.list_directory(str(f))

    assert result == {"error": f"not a directory: {f}"}


def test_list_directory_parent_is_none_at_filesystem_root():
    result = local_fs.list_directory("/")

    assert result["path"] == "/"
    assert result["parent"] is None


# ---------------------------------------------------------------------------
# existing_names
# ---------------------------------------------------------------------------


def test_existing_names_returns_only_names_that_exist_under_dest(tmp_path):
    (tmp_path / "step_10000").mkdir()
    (tmp_path / "step_20000.json").write_text("{}")
    # "step_30000" deliberately not created.

    result = local_fs.existing_names(str(tmp_path), ["step_10000", "step_20000.json", "step_30000"])

    assert result == ["step_10000", "step_20000.json"]


def test_existing_names_empty_when_dest_does_not_exist(tmp_path):
    missing_dest = tmp_path / "does_not_exist"

    result = local_fs.existing_names(str(missing_dest), ["anything"])

    assert result == []


def test_existing_names_empty_names_list_returns_empty(tmp_path):
    assert local_fs.existing_names(str(tmp_path), []) == []
