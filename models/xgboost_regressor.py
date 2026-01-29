import numpy as np
from xgboost import XGBRegressor


class XGBoostWrapper:
    def __init__(self, max_depth: int = 8, learning_rate: float = 0.1, n_estimators: int = 100,
                 subsample: float = 0.8, colsample_bytree: float = 0.8, reg_lambda: float = 1.0,
                 random_state: int = 42):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            objective='reg:squarederror',
            random_state=random_state,
            tree_method='hist',
        )
    
    def fit(self, X, y):
        X_flat = X.reshape(len(X), -1) if X.ndim > 2 else X
        self.model.fit(X_flat, y)
        return self
    
    def predict(self, X):
        X_flat = X.reshape(len(X), -1) if X.ndim > 2 else X
        return self.model.predict(X_flat)
    
    def get_booster(self):
        return self.model.get_booster()
    
    @property
    def feature_importances_(self):
        return self.model.feature_importances_
