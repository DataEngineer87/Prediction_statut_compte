# AppStreamlit.py

import os
import sys
import streamlit as st
import pandas as pd
import joblib

# 🔧 Ajout du chemin pour importer les modules personnalisés
sys.path.append("/home/sacko/Documents/ProjetAchats/Scripts")

# ✅ Import correct de TargetEncoder depuis le fichier renommé
from target_encoder_utils import TargetEncoder

# -------------------------------
# 🔄 Chargement des objets persistés
# -------------------------------
model = joblib.load("/home/sacko/Documents/ProjetAchats/Scripts/models/Model.joblib")
le = joblib.load("/home/sacko/Documents/ProjetAchats/Scripts/models/label_encoder.joblib")
target_encoder = joblib.load("/home/sacko/Documents/ProjetAchats/Scripts/models/target_encoder_pipeline.joblib")

# -------------------------------
# 🎨 Interface utilisateur
# -------------------------------
st.title("📊 Prédiction du Statut de Compte Client")

# 🔽 Choix pour les variables catégorielles
modalities = {
    'gender': ['Male', 'Female', 'Other'],
    'marital_status': ['Single', 'Married', 'Divorced', 'Widowed'],
    'employment_status': ['Employed', 'Unemployed', 'Student', 'Retired'],
    'education_level': ['High School', 'Bachelor', 'Master', 'PhD'],
    'country': ['France', 'USA', 'Germany', 'Other'],
    'subscription_type': ['Basic', 'Premium', 'VIP'],
    'age_group': ['18-25', '26-35', '36-50', '51+'],
}

# 📥 Collecte des inputs utilisateur
input_data = {}
for col in modalities:
    input_data[col] = st.selectbox(f"{col.replace('_', ' ').title()}", modalities[col])

# ➕ Variables numériques
input_data['number_of_children'] = st.number_input("Nombre d'enfants", min_value=0, value=0)
input_data['children_per_age'] = st.number_input("Enfants par tranche d'âge", min_value=0.0, value=0.0, step=0.1)
input_data['log_annual_income'] = st.number_input("Log revenu annuel", min_value=0.0, value=10.0, step=0.1)

# 🧾 DataFrame brut
df_input = pd.DataFrame([input_data])

# 🧠 Encodage (seules les variables que target_encoder attend, ici 'country')
cat_vars = ['country']
X_cat = df_input[cat_vars]
X_num = df_input.drop(columns=cat_vars)

# 🔁 Application de l'encodage
X_cat_encoded = target_encoder.transform(X_cat)

# 🔗 Fusion des données encodées avec le reste
df_encoded = pd.concat([X_cat_encoded, X_num], axis=1)

# -------------------------------
# 📈 Prédiction
# -------------------------------
if st.button("🔍 Prédire le statut du compte"):
    try:
        pred = model.predict(df_encoded)[0]
        pred_label = le.inverse_transform([pred])[0]
        st.success(f"✅ Statut prédit : **{pred_label}**")
    except Exception as e:
        st.error(f"❌ Erreur de prédiction : {e}")
