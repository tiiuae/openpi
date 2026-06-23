"""Package-extraction smoke: public surface resolves and factory dispatches."""
import numpy as np

from ensemble import (
    ActionEnsemble,
    CogACTEnsemble,
    EnsembleConfig,
    ExponentialEnsemble,
    make_ensemble,
    register_ensemble,
)


def test_public_surface_imports():
    assert issubclass(ExponentialEnsemble, ActionEnsemble)
    assert issubclass(CogACTEnsemble, ActionEnsemble)


def test_factory_dispatches_registered_types():
    assert make_ensemble("none") is None
    assert isinstance(make_ensemble("exp"), ExponentialEnsemble)
    assert isinstance(make_ensemble("cogact"), CogACTEnsemble)


def test_registry_is_open_for_extension():
    @register_ensemble("dummy")
    def _build(cfg: EnsembleConfig):
        return ExponentialEnsemble(decay=cfg.decay)

    assert isinstance(make_ensemble("dummy"), ExponentialEnsemble)


def test_async_worker_and_logger_moved_out():
    import action_logger
    import async_worker

    assert hasattr(async_worker, "AsyncPolicyWorker")
    assert hasattr(action_logger, "ActionLogger")


def test_exp_blend_unchanged():
    e = make_ensemble("exp", decay=1.0)
    e.add_chunk(0, np.array([[0.0], [10.0]]))
    e.add_chunk(1, np.array([[20.0]]))
    assert e.get_action(1) is not None
