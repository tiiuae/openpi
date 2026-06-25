from pathlib import Path

from webapp.feedback_store import save_feedback


def test_save_feedback_writes_dated_markdown(tmp_path):
    path = save_feedback(tmp_path, name="Ada", email="ada@example.com",
                         feedback="The replay charts are great.")
    p = Path(path)
    assert p.exists()
    assert p.suffix == ".md"
    text = p.read_text()
    assert "Ada" in text
    assert "ada@example.com" in text
    assert "The replay charts are great." in text


def test_save_feedback_filenames_are_unique(tmp_path):
    a = save_feedback(tmp_path, name="A", email="a@x.com", feedback="one")
    b = save_feedback(tmp_path, name="B", email="b@x.com", feedback="two")
    assert a != b
