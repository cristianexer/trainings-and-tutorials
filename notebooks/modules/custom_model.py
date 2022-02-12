
from sklearn.linear_model import LogisticRegression

class CustomModel(object):
    
    def __init__(self):
        self.model = None
    
    def fit(self, X, y):
        self.model = LogisticRegression(max_iter=10_000)
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict_proba(X)[:,1] - 0.05