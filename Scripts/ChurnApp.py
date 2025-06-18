import streamlit as st
import pandas as pd
import joblib
import streamlit.components.v1 as components
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

# Chargement du modèle et du label encoder
model = joblib.load("/home/sacko/Documents/ProjetAchats/models/model.joblib")
le = joblib.load("/home/sacko/Documents/ProjetAchats/models/label_encoder.joblib")

# Chargement des données de référence
df_reference = pd.read_csv("/home/sacko/Documents/ProjetAchats/Scripts/Customer_cleaned.csv").drop(columns=["account_status"])

# Définir les variables encodées
encoded_variables = {
    "gender": ['gender_enc_Active', 'gender_enc_Suspended', 'gender_enc_Inactive'],
    "marital_status": ['marital_status_enc_Active', 'marital_status_enc_Suspended', 'marital_status_enc_Inactive'],
    "employment_status": ['employment_status_enc_Active', 'employment_status_enc_Suspended', 'employment_status_enc_Inactive'],
    "education_level": ['education_level_enc_Active', 'education_level_enc_Suspended', 'education_level_enc_Inactive'],
    "country": ['country_enc_Active', 'country_enc_Suspended', 'country_enc_Inactive'],
    "subscription_type": ['subscription_type_enc_Active', 'subscription_type_enc_Suspended', 'subscription_type_enc_Inactive'],
    "age_group": ['age_group_enc_Active', 'age_group_enc_Suspended', 'age_group_enc_Inactive'],
}

st.title("Prédiction du statut de compte client avec suivi de dérive")

# Formulaire utilisateur
fields = {}
with st.form("formulaire_client"):
    for var_name, enc_options in encoded_variables.items():
        choix = st.selectbox(f"{var_name.replace('_', ' ').title()}", options=["Active", "Suspended", "Inactive"])
        for opt in enc_options:
            fields[opt] = int(choix.lower() in opt.lower())
    
    submitted = st.form_submit_button("Prédire")

# Si l'utilisateur a soumis le formulaire
if submitted:
    df_input = pd.DataFrame([fields])

    import os

# ...
    try:
        pred = model.predict(df_input)[0]
        pred_label = le.inverse_transform([pred])[0]
        st.success(f"✅ Statut prédit : **{pred_label}**")

        # ✅ Création du dossier si nécessaire
        os.makedirs("reports", exist_ok=True)

        # 📊 Génération dynamique du rapport Evidently
        with st.spinner("Génération du rapport Evidently..."):
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=df_reference, current_data=df_input)
            report.save_html("reports/dynamic_drift_report.html")

        st.subheader("📊 Rapport de dérive (Evidently)")
        with open("reports/dynamic_drift_report.html", "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=800, scrolling=True)

    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction ou du rapport Evidently : {e}")

        