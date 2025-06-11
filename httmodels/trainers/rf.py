"""Random Forest model trainer."""

from httmodels.models.random_forest import RandomForest
from httmodels.trainers.base import MLTrainer


class RandomForestTrainer(MLTrainer):
    """Trainer for Random Forest model."""

    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        random_state=42,
    ):
        """Initialize Random Forest trainer.

        Args:
            n_estimators: Number of trees in the forest
            criterion: Function to measure the quality of a split
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split a node
            random_state: Random state for reproducibility
        """
        model = RandomForest(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )
        super().__init__(model)
