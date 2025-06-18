import streamlit as st
import pandas as pd
import joblib

# Chargement du modèle, label encoder et pipeline d'encodage
model = joblib.load("/home/sacko/Documents/ProjetAchats/Scripts/models/model.joblib")
le = joblib.load("/home/sacko/Documents/ProjetAchats/Scripts/models/label_encoder.joblib")
pipeline = joblib.load("/home/sacko/Documents/ProjetAchats/models/TargetEncoder.joblib")

# Modalités possibles pour les variables catégorielles (exemples, adapte-les à tes données)
modalities = {
    'gender': ['Male', 'Female', 'Other'],
    'marital_status': ['Single', 'Married', 'Divorced', 'Widowed'],
    'employment_status': ['Employed', 'Unemployed', 'Student', 'Retired'],
    'education_level': ['High School', 'Bachelor', 'Master', 'PhD'],
    'country': ['France', 'USA', 'Germany', 'Other'],
    'subscription_type': ['Basic', 'Premium', 'VIP'],
    'age_group': ['18-25', '26-35', '36-50', '51+'],
}

st.title("Prédiction du statut de compte client")

# Collecte des inputs utilisateur
input_data = {}

# Variables catégorielles
for cat_var in ['gender', 'marital_status', 'employment_status', 'education_level', 'country', 'subscription_type', 'age_group']:
    input_data[cat_var] = st.selectbox(f"Choisissez {cat_var.replace('_', ' ').title()}", options=modalities[cat_var])

# Variables numériques
#input_data['annual_income'] = st.number_input("Annual Income", min_value=0.0, value=30000.0, step=1000.0)
input_data['number_of_children'] = st.number_input("Number of Children", min_value=0, max_value=20, value=0, step=1)
input_data['children_per_age'] = st.number_input("Children per Age", min_value=0.0, value=0.0, step=0.1)
input_data['log_annual_income'] = st.number_input("Log Annual Income", min_value=0.0, value=10.0, step=0.1)

# Préparation du DataFrame d'entrée
df_input = pd.DataFrame([input_data])

# Encodage des variables catégorielles avec pipeline
df_encoded = pipeline.transform(df_input)

# Ajout des variables numériques au DataFrame encodé (si pipeline ne les inclut pas)
for num_var in ['number_of_children', 'children_per_age', 'log_annual_income']:
    df_encoded[num_var] = df_input[num_var]

if st.button("Prédire"):
    try:
        pred = model.predict(df_encoded)[0]
        pred_label = le.inverse_transform([pred])[0]
        st.success(f"✅ Statut prédit : **{pred_label}**")
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")
