import pytest

from ensemble import make_ensemble


def test_factory_cogact_mode_is_configurable():
    assert make_ensemble("cogact", cogact_mode="cogact").mode == "cogact"
    assert make_ensemble("cogact", cogact_mode="latest").mode == "latest"
    assert make_ensemble("cogact", cogact_mode="hybrid").mode == "hybrid"


def test_factory_cogact_default_is_cogact_not_latest():
    # regression: the old factory hardcoded mode="latest"
    assert make_ensemble("cogact").mode == "cogact"


def test_factory_unknown_type_raises():
    with pytest.raises(ValueError):
        make_ensemble("bogus")


def test_get_latest_raw_is_gone():
    assert not hasattr(make_ensemble("exp"), "get_latest_raw")
    assert not hasattr(make_ensemble("cogact"), "get_latest_raw")
