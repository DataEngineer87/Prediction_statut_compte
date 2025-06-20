#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# Chargement du modèle
model = joblib.load("/home/sacko/Documents/ProjetAchats/Scripts/models/model.joblib")

# --- Re-définition des fonctions d'encodage utilisées à l'entraînement ---
def target_encode_smooth(df, col, target, alpha=40):
    df_copy = df[[col, target]].copy()
    classes = df[target].unique()
    global_probas = df[target].value_counts(normalize=True)

    stats = df_copy.groupby(col)[target].value_counts().unstack().fillna(0)
    totals = stats.sum(axis=1)

    encoded = pd.DataFrame(index=df.index)
    for cls in classes:
        n_cy = stats[cls] if cls in stats.columns else 0
        p_y = global_probas[cls]
        smooth = (n_cy + alpha * p_y) / (totals + alpha)
        encoded[f"{col}_enc_{cls}"] = df[col].map(smooth)
    return encoded

def encode_features(df, target_col='account_status', alpha=40):
    df = df.copy()
    
    dummy_cols = ['gender', 'marital_status', 'employment_status', 
                  'education_level', 'subscription_type', 'age_group']
    num_cols = ['number_of_children', 'children_per_age', 'log_annual_income']
    
    df_dummies = pd.get_dummies(df[dummy_cols], prefix=dummy_cols)
    df_numeric = df[num_cols]
    df_country_enc = target_encode_smooth(df, col='country', target=target_col, alpha=alpha)

    final_df = pd.concat([df_dummies, df_country_enc, df_numeric], axis=1)
    return final_df

# --- Endpoint de prédiction ---
@app.post("/predict")
def predict(data: dict):
    # Entrée utilisateur en DataFrame
    df_input = pd.DataFrame([data])

    # Ajout d’une valeur fictive pour permettre l’encodage
    df_input['account_status'] = 'Unknown'

    # Encodage
    encoded_df = encode_features(df_input, target_col='account_status', alpha=40)

    # Alignement des colonnes avec le modèle
    encoded_df = encoded_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Prédiction (le modèle renvoie déjà un label texte)
    pred = model.predict(encoded_df)[0]

    return {"prediction": pred}

# # Convertir le notebook en script Python
# import nbformat
# from nbconvert import PythonExporter
# import os

# # Définition des chemins 
# notebook_path = "/home/sacko/Documents/ProjetAchats/Scripts/api_model.ipynb"
# script_path = "/home/sacko/Documents/ProjetAchats/Scripts/Utils_model.py"

# # Fonction pour convertir le notebook en script Python
# def convert_notebook_to_script(notebook_path, script_path):
#     with open(notebook_path) as f:
#         nb = nbformat.read(f, as_version=4)
#     exporter = PythonExporter()
#     script, _ = exporter.from_notebook_node(nb)

#     with open(script_path, 'w') as f:
#         f.write(script)

# # Exécuter la conversion
# convert_notebook_to_script(notebook_path, script_path)
# print(f"Le notebook {notebook_path} a été converti en script Python.")


# In[ ]:




