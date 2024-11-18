import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Charger les données
def load_data(file_path):
    # Charger le fichier CSV
    df = pd.read_csv(file_path)

    # Convertir les labels en valeurs numériques
    df['Label'] = df['Label'].map({'Normal': 0, 'Anomaly': 1})

    # Séparer les caractéristiques (features) et les étiquettes (labels)
    X = df.drop(columns=['BlockId', 'Label'])  # Supprimer BlockId et Label pour obtenir les features
    y = df['Label']  # Label (cible)

    return X, y

# Diviser les données en ensembles d'entraînement, validation, test
def split_data(X, y):
    # Diviser les données en 60% entraînement, 20% validation, 20% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test

# Évaluer le modèle
def evaluate_model(model, X, y, model_name="Model"):
    y_pred = model.predict(X)
    y_pred_prob = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_prob)

    print(f"Performance of {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"AUC: {auc:.4f}")
    print('-' * 30)

# Charger et traiter les données
file_path = 'resultat2.csv'  # Remplacez par le chemin vers votre fichier
X, y = load_data(file_path)

# Diviser les données
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Pipeline pour normalisation et entraînement
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Entraîner le modèle Random Forest
pipeline_rf.fit(X_train, y_train)
print("Validation Performance (Random Forest):")
evaluate_model(pipeline_rf, X_val, y_val, model_name="Random Forest")

# Entraîner le modèle Logistic Regression
pipeline_lr.fit(X_train, y_train)
print("Validation Performance (Logistic Regression):")
evaluate_model(pipeline_lr, X_val, y_val, model_name="Logistic Regression")

# Tester les modèles sur l'ensemble de test
print("Test Performance (Random Forest):")
evaluate_model(pipeline_rf, X_test, y_test, model_name="Random Forest")

print("Test Performance (Logistic Regression):")
evaluate_model(pipeline_lr, X_test, y_test, model_name="Logistic Regression")
