import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression

class NIDSClient(fl.client.NumPyClient):
    def __init__(self, X_local, y_local):
        self.X_local = X_local
        self.y_local = y_local
        
        # 1. Automatically detect shapes
        num_features = X_local.shape[1] # This will be 52
        self.all_classes = np.array([0, 1, 2, 3, 4, 5, 6])
        num_classes = len(self.all_classes)
        
        # 2. Initialize Model
        self.model = LogisticRegression(
            max_iter=1, 
            warm_start=True, 
            multi_class='multinomial',
            solver='saga' # SAGA is faster for large datasets
        )

        # 3. Manual Shape Alignment
        self.model.classes_ = self.all_classes
        self.model.coef_ = np.zeros((num_classes, num_features))
        self.model.intercept_ = np.zeros(num_classes)

    def get_parameters(self, config):
        return [self.model.coef_, self.model.intercept_]

    def fit(self, parameters, config):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]
        # Train on local data
        self.model.fit(self.X_local, self.y_local)
        return self.get_parameters(config={}), len(self.X_local), {}

    def evaluate(self, parameters, config):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]
        accuracy = self.model.score(self.X_local, self.y_local)
        return 0.0, len(self.X_local), {"accuracy": float(accuracy)}
