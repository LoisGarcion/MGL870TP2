import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Load the data
df = pd.read_csv('BGL_template_counts_30min.csv')

# Ensure the Datetime column is in datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Sort the data by Datetime to ensure it's time-ordered
df = df.sort_values(by='Datetime')

# Split the data into 60% training, 20% validation, and 20% testing
train_size = int(0.60 * len(df))
val_size = int(0.20 * len(df))
test_size = len(df) - train_size - val_size

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

# Train Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict_proba(X_test)[:, 1]
rf_val = rf_model.predict_proba(X_val)[:, 1]
print(rf_pred)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=100000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict_proba(X_test)[:, 1]

# Train Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict_proba(X_test)[:, 1]


# Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    # accuracy = accuracy_score(y_true, y_pred)
    # precision = precision_score(y_true, y_pred)
    threshold = 0.5

    # Convert probabilities to binary predictions based on the threshold
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
evaluate_model(y_test, rf_pred, "Random Forest")
evaluate_model(y_test, lr_pred, "Logistic Regression")
evaluate_model(y_test, dt_pred, "Decision Tree")