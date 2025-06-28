import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import VarianceThreshold

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.CalcMolDescriptors(mol, missingVal=0.0, silent=True)  # safe descriptor calc :contentReference[oaicite:1]{index=1}

def sanitize_array(X):
    X = X.astype(np.float64, copy=False)
    # replace NaNs & infinities with bounded values
    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=np.finfo(np.float64).max,
        neginf=np.finfo(np.float64).min
    )  # ensures no NaN or inf :contentReference[oaicite:2]{index=2}

    # clip extremely large finite values
    maxv = np.finfo(np.float64).max / 10
    X = np.clip(X, -maxv, maxv)  # guard against overflow errors :contentReference[oaicite:3]{index=3}

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
    def __init__(self, max_na=0.1, discretize=True, n_bins=5, strategy="quantile"):
        """
        Parameters:
        - max_na: float, optional (default=0.1)
            Maximum allowed percentage of missing values in features. 
            Whether to apply feature scaling.
        - discretize: bool, optional (default=True)
            Whether to discretize features.
        - n_bins: int, optional (default=5)
            Number of bins used for discretization.
        - kbd_strategy: str, optional (default='quantile')
            Strategy used for binning. Options: 'uniform', 'quantile', 'kmeans'.
        """
        self.nan_filter = NanFilter(max_na)
        self.imputer = Imputer()
        self.var_filter = VarianceFilter()
        self.discretize = discretize
        if discretize:
            self.kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)

    def fit(self, smiles_list):
        df = pd.DataFrame([compute_descriptors(s) for s in tqdm(smiles_list)])
        df = df.dropna(how="all")
        X = df.values

        X = sanitize_array(X)
        self.feature_names = df.columns.tolist()

        self.nan_filter.fit(X); X = self.nan_filter.transform(X)
        self.imputer.fit(X); X = self.imputer.transform(X)
        self.var_filter.fit(X); X = self.var_filter.transform(X)
        if self.discretize:
            self.kbd.fit(X)

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
        if self.discretize:
            X = self.kbd.transform(X)

        X = X.astype(int)
        return pd.DataFrame(X, columns=self.feature_names)

    def save(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename):
        return joblib.load(filename)
