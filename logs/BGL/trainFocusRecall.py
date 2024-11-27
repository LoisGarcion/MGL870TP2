import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

def find_balanced_threshold(model, X, y_true, recall_target=0.7, min_precision=0.6):
    from sklearn.metrics import precision_recall_curve
    import numpy as np

    y_pred_probs = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probs)

    for i in range(len(thresholds)):
        if recall[i] >= recall_target and precision[i] >= min_precision:
            return thresholds[i], y_pred_probs

    print("Warning: No balanced threshold found. Defaulting to max 0.5 threshold.")
    max_recall_index = np.argmax(recall)
    return 0.5, y_pred_probs

def process_data_with_balanced_recall(file_path, model_name_suffix=""):
    df = pd.read_csv(file_path)

    df['Datetime'] = pd.to_datetime(df['Datetime'])

    df = df.sort_values(by='Datetime')

    train_size = int(0.60 * len(df))
    val_size = int(0.20 * len(df))
    test_size = len(df) - train_size - val_size

    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    X_train = train_df.drop(columns=['Datetime', 'Anomaly'])
    y_train = train_df['Anomaly'].astype(int)

    X_val = val_df.drop(columns=['Datetime', 'Anomaly'])
    y_val = val_df['Anomaly'].astype(int)

    X_test = test_df.drop(columns=['Datetime', 'Anomaly'])
    y_test = test_df['Anomaly'].astype(int)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
    rf_model = RandomForestClassifier(random_state=42)
    rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=3, scoring='recall', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    rf_best_model = rf_grid.best_estimator_
    print(f"Best Random Forest Parameters: {rf_grid.best_params_}")

    lr_param_grid = {'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear']}
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_grid = GridSearchCV(lr_model, lr_param_grid, cv=3, scoring='recall', n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    lr_best_model = lr_grid.best_estimator_
    print(f"Best Logistic Regression Parameters: {lr_grid.best_params_}")

    rf_threshold, rf_val_probs = find_balanced_threshold(rf_best_model, X_val, y_val, recall_target=0.7, min_precision=0.5)
    print(f"Balanced Threshold for Random Forest: {rf_threshold}")

    lr_threshold, lr_val_probs = find_balanced_threshold(lr_best_model, X_val, y_val, recall_target=0.7, min_precision=0.5)
    print(f"Balanced Threshold for Logistic Regression: {lr_threshold}")

    def evaluate_model(y_true, y_probs, threshold, model_name):
        y_pred = (y_probs >= threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_probs)

        print(f"Performance of {model_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC: {auc:.4f}")
        print('-' * 30)

    rf_test_probs = rf_best_model.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, rf_test_probs, rf_threshold, f"Random Forest {model_name_suffix}")

    lr_test_probs = lr_best_model.predict_proba(X_test)[:, 1]
    evaluate_model(y_test, lr_test_probs, lr_threshold, f"Logistic Regression {model_name_suffix}")

process_data_with_balanced_recall('../BGL_template_counts_error_detection.csv', model_name_suffix="ERROR DETECTION")

process_data_with_balanced_recall('../BGL_template_counts_error_prediction.csv', model_name_suffix="ERROR PREDICTION")
