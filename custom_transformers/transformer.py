from sklearn.base import TransformerMixin


class StandardScaler(TransformerMixin):
    """
    Scales the 'Age' and 'Fare' features of the dataset.
    z = (x - u) / s
    """
    def __init__(self):
        self.mean = {}
        self.std = {}
    
    def fit(self, X, *_):
        for c in ['Age', 'Fare']:
            self.mean[c] = X[c].mean()
            self.std[c] = X[c].std()

        return self
    
    def transform(self, X, *_):
        # keeps the original data unchanged
        X_copy = X.copy()
        
        # checks if fit was already called
        try:
            self.mean['Age']
            self.mean['Fare']
        except NameError:
            raise ValueError("The transformer needs to be fit before transforming data!")
        
        for c in ['Age', 'Fare']:
            X_copy[c] = (X[c] - self.mean[c]) / self.std[c]

        return X_copy