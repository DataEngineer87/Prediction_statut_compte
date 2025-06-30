import os
import sys
import streamlit as st
import pandas as pd
import joblib

# Chargement du modèle (plus besoin de LabelEncoder ici)
model = joblib.load("/home/sacko/Documents/ProjetAchats/Scripts/models/model.joblib")

# --- Fonctions d'encodage utilisées à l'entraînement ---
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

# -------------------------------
# Interface utilisateur
# -------------------------------
st.title("Prédiction du Statut de Compte Client")

modalities = {
    'gender': ['Male', 'Female', 'Other'],
    'marital_status': ['Single', 'Married', 'Divorced', 'Widowed'],
    'employment_status': ['Employed', 'Unemployed', 'Student', 'Retired'],
    'education_level': ['High School', 'Bachelor', 'Master', 'PhD'],
    'country': ['France', 'Korea', 'Germany', 'Other'],
    'subscription_type': ['Basic', 'Premium', 'VIP'],
    'age_group': ['18-24', '25-34', '35-49', '50-64', '65+'],
}

input_data = {}
for col in modalities:
    input_data[col] = st.selectbox(f"{col.replace('_', ' ').title()}", modalities[col])

# Numériques
input_data['number_of_children'] = st.number_input("Nombre d'enfants", min_value=0, value=0)
input_data['children_per_age'] = st.number_input("Enfants par tranche d'âge", min_value=0.0, value=0.0, step=0.1)
input_data['log_annual_income'] = st.number_input("Log revenu annuel", min_value=0.0, value=10.0, step=0.1)

# Données utilisateur
df_input = pd.DataFrame([input_data])
df_input['account_status'] = 'Unknown'  # requis pour target encoding

# Encodage
df_encoded = encode_features(df_input, target_col='account_status', alpha=40)

# Alignement des features avec celles vues à l'entraînement
df_encoded = df_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

# -------------------------------
# Prédiction
# -------------------------------
if st.button("Prédire le statut du compte"):
    try:
        pred = model.predict(df_encoded)[0]  # ✅ Le modèle retourne déjà une string : "Active", etc.
        st.success(f"✅ Statut prédit : **{pred}**")
    except Exception as e:
        st.error(f"❌ Erreur de prédiction : {e}")
