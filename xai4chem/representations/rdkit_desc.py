import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import RobustScaler, KBinsDiscretizer, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy import sparse
from typing import Literal, Optional

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.CalcMolDescriptors(mol, missingVal=0.0, silent=True)

def sanitize_array(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64, copy=False)

    # replace NaNs & infinities with bounded values
    X[np.isnan(X)] = 0.0
    X[np.isposinf(X)] = np.finfo(np.float64).max
    X[np.isneginf(X)] = np.finfo(np.float64).min
    
    # clip extremely large finite values
    maxv = np.finfo(np.float64).max / 10
    X = np.clip(X, -maxv, maxv) 
    
    assert np.isfinite(X).all(), "Sanitization failed: still invalid values!"
    return X

class NanFilter:
    def __init__(self, max_na_rate):
        self.max_na = max_na_rate
    def fit(self, X):
        max_bad = int((1 - self.max_na) * X.shape[0])
        self.cols = [j for j in range(X.shape[1]) if np.sum(~np.isfinite(X[:, j])) <= max_bad]
    def transform(self, X):
        return X[:, self.cols]

class Imputer:
    def fit(self, X):
        self.imps = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
    def transform(self, X):
        X = X.astype(np.float64, copy=False)
        mask = ~np.isfinite(X)
        X[mask] = np.take(self.imps, np.where(mask)[1])
        return X

class VarianceFilter:
    def fit(self, X):
        self.selector = VarianceThreshold()
        self.selector.fit(X)
    def transform(self, X):
        return self.selector.transform(X)

class RDKitDescriptor:
    def __init__(
        self,
        max_na: float = 0.1,
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
        self.nan_filter = NanFilter(max_na)
        self.imputer = Imputer()
        self.var_filter = VarianceFilter()
        self.transform_type = transform_type
        if transform_type == "discretize":
            self.transformer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=kbd_strategy)
        elif transform_type == "standard_scaler":
            self.transformer = StandardScaler()
        elif transform_type == "robust_scaler":
            self.transformer = RobustScaler()
        else:
            self.transformer = None

    def fit(self, smiles_list):
        df = pd.DataFrame([compute_descriptors(s) for s in tqdm(smiles_list)])
        df = df.dropna(how="all")
        X = df.values

        X = sanitize_array(X)
        self.feature_names = df.columns.tolist()

        self.nan_filter.fit(X)
        X = self.nan_filter.transform(X)
        self.imputer.fit(X)
        X = self.imputer.transform(X)
        self.var_filter.fit(X)
        X = self.var_filter.transform(X)
        if self.transformer:
            self.transformer.fit(X)
            self.transformer.transform(X)

        cols = np.array(self.feature_names)[self.nan_filter.cols]
        retained = cols[self.var_filter.selector.get_support()]
        self.feature_names = retained.tolist()

    def transform(self, smiles_list):
        df = pd.DataFrame([compute_descriptors(s) for s in smiles_list])
        X = df.values

        X = sanitize_array(X)
        X = self.nan_filter.transform(X)
        X = self.imputer.transform(X)
        X = self.var_filter.transform(X)
        if self.transformer:
            X = self.transformer.transform(X)

        if self.transform_type == "discretize":
            if sparse.issparse(X):
                X = np.asarray(X.todense())# type: ignore
            else:
                X = np.asarray(X, dtype=np.float32)
            X = X.astype(int)
 
        return pd.DataFrame(np.asarray(X), columns=self.feature_names)

    def save(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        return joblib.load(filename)
