from sklearn.linear_model import LogisticRegression
import numpy as np

def create_initial_model():
    """
    Initializes the model with manual shape alignment.
    This was the fix for the 'matmul' and 'dimensionality' errors.
    """
    # 1. Define hyperparameters
    model = LogisticRegression(
        max_iter=1, 
        warm_start=True, 
        multi_class='multinomial',
        solver='saga'
    )

    # 2. Manual Shape Alignment (The key fix from our simulation)
    # We pre-set the 7 attack classes and 52 traffic features.
    model.classes_ = np.array([0, 1, 2, 3, 4, 5, 6])
    model.coef_ = np.zeros((7, 52))
    model.intercept_ = np.zeros(7)
    
    return model
