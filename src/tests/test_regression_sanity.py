import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score


def test_tiny_synthetic_training_monotonic_scores():
    X = pd.DataFrame(
        {
            "signal_strength": [0, 0.1, 0.2, 0.8, 0.9, 1.0],
            "noise": [0.5, 0.4, 0.6, 0.5, 0.4, 0.6],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 1])

    clf = LogisticRegression()
    clf.fit(X, y)

    proba = clf.predict_proba(X)[:, 1]
    assert not np.isnan(roc_auc_score(y, proba))
    assert not np.isnan(accuracy_score(y, clf.predict(X)))

    # Monotonicity: higher signal_strength should yield higher predicted score
    assert proba[-1] > proba[0]
