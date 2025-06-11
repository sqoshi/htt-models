"""AdaBoost model trainer."""

from httmodels.models.adaboost import AdaBoost
from httmodels.trainers.base import MLTrainer


class AdaBoostTrainer(MLTrainer):
    """Trainer for AdaBoost model."""

    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=42):
        """Initialize AdaBoost trainer.

        Args:
            n_estimators: Maximum number of estimators
            learning_rate: Learning rate shrinks the contribution of each classifier
            random_state: Random state for reproducibility
        """
        model = AdaBoost(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        super().__init__(model)
