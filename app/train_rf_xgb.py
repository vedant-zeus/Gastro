import os
import numpy as np
from data_pipeline import train_gen, val_gen, test_gen
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Helper to flatten image batches and labels

def extract_features(generator):
    X, y = [], []
    for _ in range(len(generator)):
        batch_x, batch_y = next(generator)
        X.append(batch_x)
        y.append(batch_y)
    X = np.concatenate(X)
    y = np.concatenate(y)
    X = X.reshape(X.shape[0], -1)  # Flatten images
    y = np.argmax(y, axis=1)      # One-hot to class index
    return X, y

print('Extracting features for train set...')
X_train, y_train = extract_features(train_gen)
print('Extracting features for val set...')
X_val, y_val = extract_features(val_gen)
print('Extracting features for test set...')
X_test, y_test = extract_features(test_gen)

# Train Random Forest
print('Training Random Forest...')
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print('Random Forest accuracy:', accuracy_score(y_test, rf.predict(X_test)))
print(classification_report(y_test, rf.predict(X_test)))
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models'), exist_ok=True)
joblib.dump(rf, os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest_gi.pkl'))

# Train XGBoost
print('Training XGBoost...')
xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
print('XGBoost accuracy:', accuracy_score(y_test, xgb.predict(X_test)))
print(classification_report(y_test, xgb.predict(X_test)))
joblib.dump(xgb, os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost_gi.pkl'))
