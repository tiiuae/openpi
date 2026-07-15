import pytest

from webapp.config_store import ConfigStore


def test_save_then_load_round_trip(tmp_path):
    store = ConfigStore(tmp_path)
    cfg = {"policy_host": "192.168.1.9", "control_freq": 25, "smoothing": True}
    store.save("rig-a", cfg)
    assert store.load("rig-a") == cfg


def test_list_names_sorted(tmp_path):
    store = ConfigStore(tmp_path)
    store.save("b", {}); store.save("a", {})
    assert store.list_names() == ["a", "b"]


def test_delete_removes(tmp_path):
    store = ConfigStore(tmp_path)
    store.save("x", {"a": 1})
    store.delete("x")
    assert store.list_names() == []


def test_load_missing_raises(tmp_path):
    with pytest.raises(KeyError):
        ConfigStore(tmp_path).load("nope")


def test_name_is_sanitized(tmp_path):
    store = ConfigStore(tmp_path)
    with pytest.raises(ValueError):
        store.save("../escape", {})
