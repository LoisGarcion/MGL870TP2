import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
import os
#from imblearn.under_sampling import SMOTE
from imblearn.over_sampling import SMOTE




# Function to load data, train, and evaluate models
def process_data(file_path, model_name_suffix=""):
    # Load the data
    df = pd.read_csv(file_path)
    df['Label'] = df['Label'].map({'Normal': 0, 'Anomaly': 1})

    output_dir="data_splits"
    model_name_suffix=""
    os.makedirs(output_dir, exist_ok=True)
    
    
    # Split the data into 60% training, 20% validation, and 20% testing
    train_size = int(0.60 * len(df))
    val_size = int(0.20 * len(df))

    # Adjusted Splits for Time-Series Data
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    # Separate features and labels
    X_train = train_df.drop(columns=['BlockId', 'Label'])  # Features
    y_train = train_df['Label'].astype(int)  # Labels
    
    

    X_val = val_df.drop(columns=['BlockId', 'Label'])
    y_val = val_df['Label'].astype(int)
    X_val_resampled, y_val_resampled = SMOTE().fit_resample(X_val, y_val)


    X_test = test_df.drop(columns=['BlockId', 'Label'])
    y_test = test_df['Label'].astype(int)
    X_test_resampled, y_test_resampled = SMOTE().fit_resample(X_test, y_test)


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Train Random Forest
    rf_model = RandomForestClassifier(class_weight='balanced')
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict_proba(X_test)[:, 1]
    rf_val = rf_model.predict_proba(X_val)[:, 1]

    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict_proba(X_test)[:, 1]
    
    
    # Evaluation function
    def evaluate_model(y_true, y_pred, model_name, threshold):
        rf_val_pred = (y_pred >= threshold).astype(int)
        accuracy = accuracy_score(y_true, rf_val_pred)
        precision = precision_score(y_true, rf_val_pred)
        recall = recall_score(y_true, rf_val_pred)
        auc = roc_auc_score(y_true, y_pred)

        print(f"Performance of {model_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC: {auc:.4f}")
        print('-' * 30)

    # Evaluate each model on the test set
    evaluate_model(y_test, rf_pred, f"Random Forest {model_name_suffix}", threshold=0.5)
    evaluate_model(y_test, lr_pred, f"Logistic Regression {model_name_suffix}",threshold=0.5)


process_data('resultat2.csv', model_name_suffix="ERROR PREDICTION")
