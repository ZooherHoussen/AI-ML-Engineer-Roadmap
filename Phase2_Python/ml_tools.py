
"""
Module utilitaire pour le ML
"""
import numpy as np

def normaliser(X):
    """Normalise les données entre 0 et 1"""
    return (X - X.min()) / (X.max() - X.min())

def standardiser(X):
    """Standardise les données (μ=0, σ=1)"""
    return (X - X.mean()) / X.std()

def train_test_split_simple(X, y, test_size=0.2, seed=42):
    """Split simple sans scikit-learn"""
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    split = int(n * (1 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def accuracy(y_true, y_pred):
    """Calcule l'accuracy"""
    return np.mean(y_true == y_pred)

def afficher_stats(X, nom="Dataset"):
    """Affiche les statistiques d'un tableau"""
    print(f"=== {nom} ===")
    print(f"Shape  : {X.shape}")
    print(f"Mean   : {X.mean():.4f}")
    print(f"Std    : {X.std():.4f}")
    print(f"Min    : {X.min():.4f}")
    print(f"Max    : {X.max():.4f}")

ML_TOOLS_VERSION = "1.0.0"
