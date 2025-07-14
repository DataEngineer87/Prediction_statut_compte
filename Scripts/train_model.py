#!/usr/bin/env python
# coding: utf-8

# # Train_model.py

# In[21]:


import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn 
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from mlflow.models.signature import infer_signature


# # Importation des données nettoyées

# In[14]:


# Données d'entrainement
df_train_cleaned = pd.read_csv("data/df_train_cleaned.csv")

print(df_train_cleaned.shape)
df_train_cleaned.head() # - Affichage des premières lignes du jeu de données pour visualiser rapidement la structure et les premières valeurs.

# Données test
df_test_cleaned = pd.read_csv("data/df_test_cleaned.csv")

print(df_test_cleaned.shape)
df_test_cleaned.head() # - Affichage des premières lignes du jeu de données pour visualiser rapidement la structure et les premières valeurs.


# # Modélisation

# In[16]:


# Séparation des variables explicatives (features) et de la variable cible ("account_status")
X_train = df_train_cleaned.drop(["account_status"], axis = 1)

y_train = df_train_cleaned["account_status"]


# In[17]:


# Séparation des variables explicatives (features) et de la variable cible ("account_status")
X_test = df_test_cleaned.drop(["account_status"], axis = 1)

y_test = df_test_cleaned["account_status"]


# In[22]:


# ----- FONCTION DE TARGET ENCODING -----

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

def encode_features(df, target_col='account_status', alpha=10):
    df = df.copy()
    dummy_cols = ['gender', 'marital_status', 'employment_status', 
                  'education_level', 'subscription_type', 'age_group']
    
    df_dummies = pd.get_dummies(df[dummy_cols], prefix=dummy_cols)
    country_enc = target_encode_smooth(df, col='country', target=target_col, alpha=alpha)

    numeric_cols = df.drop(columns=dummy_cols + ['country', target_col]).copy()
    
    # 🔧 Transformation : tous les entiers en float64 (pour éviter le warning MLflow)
    numeric_cols = numeric_cols.astype({col: 'float64' for col in numeric_cols.select_dtypes('int').columns})

    final_df = pd.concat([df_dummies, country_enc, numeric_cols], axis=1)
    final_df[target_col] = df[target_col]
    
    return final_df

# ------- PRÉPARATION DONNÉES -------

os.makedirs("models", exist_ok=True)

# Remplace avec tes propres X_train/y_train
# Exemple :
# X_train, X_test, y_train, y_test = train_test_split(...)

train_encoded = encode_features(X_train.assign(account_status=y_train), target_col='account_status')
test_encoded = encode_features(X_test.assign(account_status=y_test), target_col='account_status')

X_train_encoded = train_encoded.drop(columns='account_status')
y_train_encoded = train_encoded['account_status']
X_test_encoded = test_encoded.drop(columns='account_status')
y_test_encoded = test_encoded['account_status']

# Réalignement des colonnes test ↔ train
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# ----- ENTRAÎNEMENT AVEC MLflow -----

model = RandomForestClassifier(n_estimators=100, random_state=42)

mlflow.set_experiment("account_status_prediction")

with mlflow.start_run():
    model.fit(X_train_encoded, y_train_encoded)
    preds = model.predict(X_test_encoded)

    report = classification_report(y_test_encoded, preds, output_dict=True)
    acc = report['accuracy']

    # Créer dossier et sauvegarder rapport
    os.makedirs("reports", exist_ok=True)
    with open("reports/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)
    mlflow.log_artifact("reports/evaluation_report.json")

    input_example = X_train_encoded.iloc[:1]
    signature = infer_signature(X_train_encoded, model.predict(X_train_encoded))

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(
    sk_model=model,
    artifact_path="model",  
    input_example=input_example,
    signature=signature,
    registered_model_name="account_status_rf"
)


    joblib.dump(model, "models/model.joblib")


# ----- DONNÉES TEST POUR API -----
data = {
    "gender": "Male",
    "marital_status": "Single",
    "employment_status": "Employed",
    "education_level": "Bachelor",
    "subscription_type": "Standard",
    "age_group": "25-34",  
    "number_of_children": 2,
    "children_per_age": 0.5,
    "log_annual_income": 10.5,
    "country": "France"
}


# In[ ]:




