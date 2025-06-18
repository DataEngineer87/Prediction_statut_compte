#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Scripts/target_encoder.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------------
# Target Encoding avec smoothing
# -------------------------------

def target_encode_smooth(X, y, alpha=10):
    """Encode target smoothing pour un DataFrame X par rapport à y."""
    X = X.copy()
    df = X.copy()
    df['target'] = y
    classes = y.unique()
    global_probas = y.value_counts(normalize=True)

    encoded = pd.DataFrame(index=X.index)
    
    for col in X.columns:
        stats = df.groupby(col)['target'].value_counts().unstack().fillna(0)
        totals = stats.sum(axis=1)

        for cls in classes:
            n_cy = stats[cls] if cls in stats.columns else 0
            p_y = global_probas[cls]
            smoothed = (n_cy + alpha * p_y) / (totals + alpha)
            encoded[f'{col}_enc_{cls}'] = X[col].map(smoothed)
    
    return encoded


class TargetEncoder(TransformerMixin, BaseEstimator):
    """Encodage target smoothing personnalisé compatible scikit-learn."""
    def __init__(self, alpha=10):
        self.alpha = alpha

    def fit(self, X, y):
        self.y_ = y
        return self

    def transform(self, X):
        return target_encode_smooth(X, self.y_, self.alpha).reindex(X.index)

# -------------------------------
# Utilitaire : Conversion notebook -> script Python
# -------------------------------
if __name__ == "__main__":
    import nbformat
    from nbconvert import PythonExporter

    def convert_notebook_to_script(notebook_path, script_path):
        """Convertit un notebook Jupyter en script Python."""
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        exporter = PythonExporter()
        script, _ = exporter.from_notebook_node(nb)

        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)

    #  Adapter ces chemins selon ton environnement
    notebook_path = "/home/sacko/Documents/ProjetAchats/Scripts/target_encoder.ipynb"
    script_path = "/home/sacko/Documents/ProjetAchats/Scripts/target_encoder.py"

    convert_notebook_to_script(notebook_path, script_path)
    print("✅ Conversion réussie !")


# In[ ]:




