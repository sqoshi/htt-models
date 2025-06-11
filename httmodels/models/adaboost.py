"""AdaBoost model implementation."""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from httmodels.models.base import SklearnModel


class AdaBoost(SklearnModel):
    """AdaBoost classifier for image classification.

    Wrapper around scikit-learn's AdaBoostClassifier with
    the project's BaseModel interface.
    """

    def __init__(
        self, n_estimators=50, learning_rate=1.0, base_estimator=None, random_state=42
    ):
        """Initialize AdaBoost model.

        Args:
            n_estimators: Maximum number of estimators
            learning_rate: Learning rate shrinks the contribution of each classifier
            base_estimator: Base estimator from which the boosted ensemble is built
            random_state: Random state for reproducibility
        """
        # Use a decision tree with max_depth=1 (decision stump) as base estimator if not provided
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier(max_depth=1)

        model = AdaBoostClassifier(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        super(AdaBoost, self).__init__(model)

    def forward(self, x):
        """Predict class labels for samples in x.

        Args:
            x: Input features

        Returns:
            Predicted class labels
        """
        return self.model.predict(x)
