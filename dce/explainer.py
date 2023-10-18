from abc import ABC, abstractmethod

class CounterfactualExplainer(ABC):

    def __init__(self, model):
        """
        Initialize the counterfactual explainer.

        Parameters:
        - model: The predictive model to be explained.
        """
        self.model = model

    @abstractmethod
    def explain(self, instance, target_class=None):
        """
        Generate a counterfactual explanation for a given instance.

        Parameters:
        - instance: The input data instance for which the counterfactual is to be found.
        - target_class: (Optional) The desired target class for the counterfactual. 
                        If not provided, the target class will be the opposite of the model's prediction.

        Returns:
        - counterfactual: The counterfactual instance.
        - explanation: A textual explanation or additional data about the counterfactual.
        """
        pass

    @abstractmethod
    def fit(self, data):
        """
        Fit the explainer to the data, if necessary. This can be useful for methods that 
        require training or calibration on a dataset before generating explanations.

        Parameters:
        - data: The dataset to fit the explainer on.
        """
        pass
