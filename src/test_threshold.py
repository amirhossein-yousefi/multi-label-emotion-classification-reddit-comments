import numpy as np
from emoclass.threshold import best_threshold_by_micro_f1

def test_best_threshold_monotonic_grid():
    y_true = np.array([[1,0,1],[0,1,0],[1,1,0]])
    y_prob = np.array([[0.9,0.1,0.8],[0.2,0.7,0.1],[0.6,0.55,0.4]])
    t = best_threshold_by_micro_f1(y_true, y_prob, 0.1, 0.9, 0.1)
    assert 0.1 <= t <= 0.9
