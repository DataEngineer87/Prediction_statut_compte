#!/usr/bin/env python
# coding: utf-8

# # Train_model.py

# In[1]:


import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from mlflow.models.signature import infer_signature


# # Importation des données nettoyées

# In[2]:


# Données d'entrainement
df_train_cleaned = pd.read_csv("/home/sacko/Documents/ProjetAchats/Donnees/df_train_cleaned.csv")
print(df_train_cleaned.shape)
df_train_cleaned.head() # - Affichage des premières lignes du jeu de données pour visualiser rapidement la structure et les premières valeurs.


# In[3]:


# Données test
df_test_cleaned = pd.read_csv("/home/sacko/Documents/ProjetAchats/Donnees/df_test_cleaned.csv")
print(df_test_cleaned.shape)
df_test_cleaned.head() # - Affichage des premières lignes du jeu de données pour visualiser rapidement la structure et les premières valeurs.


# # Modélisation

# In[4]:


# Séparation des variables explicatives (features) et de la variable cible ("account_status")
X_train = df_train_cleaned.drop(["account_status"], axis = 1)

y_train = df_train_cleaned["account_status"]


# In[5]:


# Séparation des variables explicatives (features) et de la variable cible ("account_status")
X_test = df_test_cleaned.drop(["account_status"], axis = 1)

y_test = df_test_cleaned["account_status"]


# In[ ]:


# Fonction d'encodage personnalisée 
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

    final_df = pd.concat([df_dummies, country_enc], axis=1)
    final_df[target_col] = df[target_col]

    return final_df

# Préparation des données et encodage ---
os.makedirs("models", exist_ok=True)

# Encodage complet avec ta fonction (remplace le TargetEncoder sklearn)
train_encoded = encode_features(X_train.assign(account_status=y_train), target_col='account_status', alpha=10)
test_encoded = encode_features(X_test.assign(account_status=y_test), target_col='account_status', alpha=10)

# Séparation des features / cibles après encodage
X_train_encoded = train_encoded.drop(columns='account_status')
y_train_encoded = train_encoded['account_status']

X_test_encoded = test_encoded.drop(columns='account_status')
y_test_encoded = test_encoded['account_status']

# Conversion des colonnes int en float pour MLflow (optionnel mais conseillé)
X_train_encoded = X_train_encoded.astype({col: 'float' for col in X_train_encoded.select_dtypes('int').columns})
X_test_encoded = X_test_encoded.astype({col: 'float' for col in X_test_encoded.select_dtypes('int').columns})

# Réalignement des colonnes test sur train (au cas où)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Définition du modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)

# MLflow tracking 

# Définition d'expérience MLflow (crée ou réutilise une expérience existante)
mlflow.set_experiment("account_status_prediction")

# Démarrage d'un nouveau run MLflow pour enregistrer les infos du modèle entraîné
with mlflow.start_run():
    
    # Entraînement du modèle RandomForest sur les données encodées
    model.fit(X_train_encoded, y_train_encoded)
    
    # Prédiction des étiquettes sur l’ensemble de test
    preds = model.predict(X_test_encoded)
    
    # Génèration d'un rapport de classification (précision, recall, f1-score,...)
    report = classification_report(y_test_encoded, preds, output_dict=True)
    
    # Récupèratioin de la précision globale (accuracy) du modèle
    acc = report['accuracy']

    # Exemple d'entrée pour documenter le modèle dans MLflow
    input_example = X_train_encoded.iloc[:1]
    
    # Génèration de la signature du modèle (types des entrées/sorties)
    signature = infer_signature(X_train_encoded, model.predict(X_train_encoded))

    # Enregistrement d'un paramètre du run : ici, le type de modèle utilisé
    mlflow.log_param("model_type", "RandomForest")
    
    # Enregistrement d'une métrique : ici, la précision obtenue
    mlflow.log_metric("accuracy", acc)
    
    # Enregistrement du modèle entraîné dans MLflow avec l'exemple d'entrée et sa signature
    mlflow.sklearn.log_model(model, "model", input_example=input_example, signature=signature)
    
    # Sauvegardes du modèle
    joblib.dump(model, "models/model.joblib")
    
    # données pour test_api.py
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


# # Convertir le notebook en script Python

# In[ ]:


import nbformat
from nbconvert import PythonExporter
import os

# Définition des chemins 
notebook_path = "/home/sacko/Documents/ProjetAchats/Scripts/train_model.ipynb"
script_path = "/home/sacko/Documents/ProjetAchats/Scripts/train_model.py"

# Fonction pour convertir le notebook en script Python
def convert_notebook_to_script(notebook_path, script_path):
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    exporter = PythonExporter()
    script, _ = exporter.from_notebook_node(nb)

    with open(script_path, 'w') as f:
        f.write(script)

# Exécution de la conversion
convert_notebook_to_script(notebook_path, script_path)
print(f"Le notebook {notebook_path} a été converti en script Python.")


# In[ ]:





# In[ ]:




