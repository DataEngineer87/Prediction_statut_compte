#!/usr/bin/env python
# coding: utf-8

# # FastAPI

from pydantic import BaseModel
from typing import Literal
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os

app = FastAPI()

# ROUTE RACINE
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction du statut de compte."}

# Chargement du modèle
MODEL_PATH = "/home/sacko/Documents/ProjetAchats/Scripts/models/model.joblib"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")

# Re-définition des fonctions d'encodage utilisées à l'entraînement 
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

# Définition du schéma d'entrée
class InputData(BaseModel):
    gender: Literal["Male", "Female"]
    marital_status: Literal["Married", "Single", "Divorced", "Widowed"]
    employment_status: Literal["Employed", "Unemployed", "Student", "Retired"]
    education_level: Literal["High School", "Bachelor", "Master", "PhD"]
    subscription_type: Literal["Basic", "Standard", "Premium"]
    age_group: Literal["18-24", "25-34", "35-44", "45-54", "55+"]
    number_of_children: int
    children_per_age: float
    log_annual_income: float
    country: str

# Endpoint de prédiction 
@app.post("/predict")
def predict(data: InputData):
    try:
        # Transformation des données
        df_input = pd.DataFrame([data.dict()])
        df_input['account_status'] = 'Unknown'  # Dummy target for encoding

        # Encodage
        encoded_df = encode_features(df_input, target_col='account_status', alpha=40)

        # Alignement avec les colonnes du modèle
        encoded_df = encoded_df.reindex(columns=model.feature_names_in_, fill_value=0)

        # Prédiction
        pred = model.predict(encoded_df)[0]

        return {"prediction": pred}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {e}")

