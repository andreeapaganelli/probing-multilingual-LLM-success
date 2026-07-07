import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


def clip_success_rate_predictions(y_pred: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(y_pred, dtype=np.float32), 0.0, 1.0)


class LogisticProbe:
    """Logistic probe with the same predict() API as Ridge.

    The multilingual probe scripts persist probe bundles with joblib. Keeping
    this wrapper in an importable module avoids fragile ``__main__`` pickles
    when bundles are loaded later by analysis scripts.
    """

    def __init__(
        self,
        C: float,
        class_weight: str | dict | None = None,
        max_iter: int = 2000,
        random_state: int = 42,
    ) -> None:
        self._scaler = StandardScaler()
        self._clf = LogisticRegression(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=random_state,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticProbe":
        y_bin = (np.asarray(y) > 0.5).astype(int)
        self._clf.fit(self._scaler.fit_transform(X), y_bin)
        return self

    def __setstate__(self, state: dict) -> None:
        # Probes saved before _scaler was added had only _clf; restore a fitted
        # identity scaler so predict() works transparently on old bundles.
        self.__dict__.update(state)
        if "_scaler" not in state:
            from sklearn.preprocessing import StandardScaler
            n = self._clf.n_features_in_
            scaler = StandardScaler()
            scaler.mean_ = np.zeros(n, dtype=np.float64)
            scaler.scale_ = np.ones(n, dtype=np.float64)
            scaler.var_ = np.ones(n, dtype=np.float64)
            scaler.n_features_in_ = n
            scaler.n_samples_seen_ = 1
            self._scaler = scaler

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return P(success=1), matching the Ridge probe score interface."""
        if len(self._clf.classes_) < 2:
            return np.full(len(X), float(self._clf.classes_[0]), dtype=np.float32)
        return self._clf.predict_proba(self._scaler.transform(X))[:, 1].astype(np.float32)


class BalancedLogisticProbe(LogisticProbe):
    """Class-balanced logistic probe with the same predict() API as Ridge."""

    def __init__(self, C: float, max_iter: int = 2000, random_state: int = 42) -> None:
        super().__init__(
            C=C,
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state,
        )


class ScaledRidgeProbe:
    """Ridge regression with train-split feature standardization."""

    def __init__(self, alpha: float) -> None:
        self._scaler = StandardScaler()
        self._ridge = Ridge(alpha=alpha)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "ScaledRidgeProbe":
        X_scaled = self._scaler.fit_transform(X)
        self._ridge.fit(X_scaled, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._ridge.predict(self._scaler.transform(X))


class BalancedRidgeProbe:
    """Ridge regression with class-balanced sample weights (no feature standardization).

    At fit time, examples whose binarised label (y > threshold) is the minority
    class are up-weighted so that both classes contribute equally to the loss.
    This mirrors ``BalancedLogisticProbe`` but for regression-style Ridge targets.
    """

    def __init__(self, alpha: float, threshold: float = 0.5) -> None:
        self._ridge = Ridge(alpha=alpha)
        self._threshold = threshold

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "BalancedRidgeProbe":
        if sample_weight is None:
            y_bin = (np.asarray(y) > self._threshold).astype(int)
            n_pos = int(y_bin.sum())
            n_neg = int(len(y_bin) - n_pos)
            if n_pos > 0 and n_neg > 0:
                w_pos = len(y_bin) / (2.0 * n_pos)
                w_neg = len(y_bin) / (2.0 * n_neg)
                sample_weight = np.where(y_bin == 1, w_pos, w_neg).astype(np.float32)
        self._ridge.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._ridge.predict(X)
