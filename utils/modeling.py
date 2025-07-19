from sklearn.model_selection import KFold
import numpy as np

def fit_cv_models(df, features, target_col, model_class, k=10, seed=42, model_kwargs=None):
    """
    Trains k-fold models and returns a list of fitted models.
    """
    if model_kwargs is None:
        model_kwargs = {}
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    models = []
    for train_idx, _ in kf.split(df):
        sub = df.iloc[train_idx]
        X = sub[features].values
        y = sub[target_col].values
        model = model_class(**model_kwargs)
        model.fit(X, y)
        models.append(model)
    return models


def average_linear_coeffs(models):
    coefs = np.array([m.coef_ for m in models])
    intercepts = np.array([m.intercept_ for m in models])
    return coefs.mean(axis=0), intercepts.mean()

def predict_cv_ensemble(df, features, models):
    X = df[features].values
    preds = np.column_stack([m.predict(X) for m in models])
    return preds.mean(axis=1)  # Average predictions from all models
