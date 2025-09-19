import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler, KBinsDiscretizer, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from mordred import Calculator, descriptors
from rdkit import Chem
import pandas as pd
from scipy import sparse
from typing import Literal, Optional


MAX_NA = 0.2

class NanFilter(object):
    def __init__(self):
        self._name = "nan_filter"

    def fit(self, X):
        max_na = int((1 - MAX_NA) * X.shape[0])
        idxs = []
        for j in range(X.shape[1]):
            c = np.sum(np.isnan(X[:, j]))
            if c > max_na:
                continue
            else:
                idxs += [j]
        self.col_idxs = idxs

    def transform(self, X):
        return X[:, self.col_idxs]

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


class Imputer(object):
    def __init__(self):
        self._name = "imputer"
        self._fallback = 0

    def fit(self, X):
        ms = []
        for j in range(X.shape[1]):
            vals = X[:, j]
            mask = ~np.isnan(vals)
            vals = vals[mask]
            if len(vals) == 0:
                m = self._fallback
            else:
                m = np.median(vals)
            ms += [m]
        self.impute_values = np.array(ms)

    def transform(self, X):
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = self.impute_values[j]
        return X

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


class VarianceFilter(object):
    def __init__(self):
        self._name = "variance_filter"

    def fit(self, X):
        self.sel = VarianceThreshold()
        self.sel.fit(X)
        self.col_idxs = np.where(self.sel.get_support())[0]

    def transform(self, X):
        return self.sel.transform(X)

    def save(self, file_name):
        joblib.dump(self, file_name)

    def load(self, file_name):
        return joblib.load(file_name)


def mordred_featurizer(smiles):
    calc = Calculator(descriptors, ignore_3D=True)
    df = calc.pandas([Chem.MolFromSmiles(smi) for smi in smiles])
    return df


class MordredDescriptor(object):
    def __init__(
        self,
        transform_type: Optional[str] = None,
        n_bins: int = 5,
        kbd_strategy: Literal["uniform", "quantile", "kmeans"] = "quantile"
    ):
        """ 
        Parameters:
        - max_na: Maximum allowed percentage of missing values in features.
        - transform_type: One of {"discretize", "standard_scaler", "robust_scaler", None}
        - n_bins: Number of bins used for discretization (if transform_type == "discretize").
        - kbd_strategy: Strategy used for binning. Options: 'uniform', 'quantile', 'kmeans'.
        """
        self.nan_filter = NanFilter()
        self.imputer = Imputer()
        self.variance_filter = VarianceFilter()
        self.transform_type = transform_type
        if transform_type == "discretize":
            self.transformer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=kbd_strategy)
        elif transform_type == "standard_scaler":
            self.transformer = StandardScaler()
        elif transform_type == "robust_scaler":
            self.transformer = RobustScaler()
        else:
            self.transformer = None

    def fit(self, smiles):
        df = mordred_featurizer(smiles)
        X = np.array(df, dtype=np.float64)
        self.nan_filter.fit(X)
        X = self.nan_filter.transform(X)
        self.imputer.fit(X)
        X = self.imputer.transform(X)
        self.variance_filter.fit(X)
        X = self.variance_filter.transform(X)
        if self.transformer:
            self.transformer.fit(X)
            self.transformer.transform(X)
        self.features = list(df.columns)
        self.features = [self.features[i] for i in self.nan_filter.col_idxs]
        self.features = [self.features[i] for i in self.variance_filter.col_idxs]
        # return pd.DataFrame(np.asarray(X), columns=self.features)

    def transform(self, smiles):
        df = mordred_featurizer(smiles)
        X = np.array(df, dtype=np.float32)
        X = self.nan_filter.transform(X)
        X = self.imputer.transform(X)
        X = self.variance_filter.transform(X)
        if self.transformer:
            X = self.transformer.transform(X)

        if self.transform_type == "discretize":
            if sparse.issparse(X):
                X = np.asarray(X.todense())# type: ignore
            else:
                X = np.asarray(X, dtype=np.float32)
            X = X.astype(int)
        return pd.DataFrame(np.asarray(X), columns=self.features)
    
    def save(self, file_name):
        joblib.dump(self, file_name)
        
    @classmethod
    def load(cls, file_name):
        return joblib.load(file_name)
