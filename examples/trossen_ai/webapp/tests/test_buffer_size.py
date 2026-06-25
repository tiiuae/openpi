import numpy as np

from ensemble import CogACTEnsemble, ExponentialEnsemble


def test_exp_buffer_size_counts_buffered_entries():
    e = ExponentialEnsemble(decay=1.0)
    assert e.buffer_size() == 0
    e.add_chunk(0, np.ones((5, 2)))  # 5 future steps buffered
    assert e.buffer_size() == 5


def test_cogact_buffer_size_counts_chunks():
    e = CogACTEnsemble(max_buffer_size=25, mode="cogact")
    assert e.buffer_size() == 0
    e.add_chunk(0, np.ones((10, 4)))
    e.add_chunk(1, np.ones((10, 4)))
    assert e.buffer_size() == 2
