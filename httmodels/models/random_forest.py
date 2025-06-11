"""Random Forest model implementation."""

from sklearn.ensemble import RandomForestClassifier

from httmodels.models.base import SklearnModel


class RandomForest(SklearnModel):
    """Random Forest classifier for image classification.

    Wrapper around scikit-learn's RandomForestClassifier with
    the project's BaseModel interface.
    """

    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        random_state=42,
    ):
        """Initialize RandomForest model.

        Args:
            n_estimators: Number of trees in the forest
            criterion: Function to measure the quality of a split
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split a node
            random_state: Random state for reproducibility
        """
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=-1,
            random_state=random_state,
        )
        super(RandomForest, self).__init__(model)

    def forward(self, x):
        """Predict class labels for samples in x.

        Args:
            x: Input features

        Returns:
            Predicted class labels
        """
        return self.model.predict(x)
