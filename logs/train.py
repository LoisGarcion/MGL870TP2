import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Function to load data, train, and evaluate models
def process_data(file_path, model_name_suffix=""):
    # Load the data
    df = pd.read_csv(file_path)

    # Ensure the Datetime column is in datetime format
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    # Sort the data by Datetime to ensure it's time-ordered
    df = df.sort_values(by='Datetime')

    # Split the data into 60% training, 20% validation, and 20% testing
    train_size = int(0.60 * len(df))
    val_size = int(0.20 * len(df))
    test_size = len(df) - train_size - val_size

    # Adjusted Splits for Time-Series Data
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    # Separate features and labels
    X_train = train_df.drop(columns=['Datetime', 'Anomaly'])  # Features
    y_train = train_df['Anomaly'].astype(int)  # Labels

    X_val = val_df.drop(columns=['Datetime', 'Anomaly'])
    y_val = val_df['Anomaly'].astype(int)

    X_test = test_df.drop(columns=['Datetime', 'Anomaly'])
    y_test = test_df['Anomaly'].astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Train Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict_proba(X_test)[:, 1]
    rf_val = rf_model.predict_proba(X_val)[:, 1]

    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=10000000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict_proba(X_test)[:, 1]

    # Evaluation function
    def evaluate_model_random_forest(y_true, y_pred, model_name):
        threshold = 0.3
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

    # Evaluation function
    def evaluate_model_logistic_regression(y_true, y_pred, model_name):
        threshold = 0.5
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
    evaluate_model_random_forest(y_test, rf_pred, f"Random Forest {model_name_suffix}")
    evaluate_model_logistic_regression(y_test, lr_pred, f"Logistic Regression {model_name_suffix}")

# Process the new 10-minute interval data
process_data('BGL_template_counts_error_prediction.csv', model_name_suffix="ERROR PREDICTION")

# Process the original 30-minute interval data
process_data('BGL_template_counts_error_detection.csv', model_name_suffix="ERROR DETECTION")