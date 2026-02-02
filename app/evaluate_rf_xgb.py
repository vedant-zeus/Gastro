import os
import numpy as np
from data_pipeline import test_gen
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Helper to flatten image batches and labels
def extract_features(generator):
    X, y = [], []
    for _ in range(len(generator)):
        batch_x, batch_y = next(generator)
        X.append(batch_x)
        y.append(batch_y)
    X = np.concatenate(X)
    y = np.concatenate(y)
    X = X.reshape(X.shape[0], -1)
    y = np.argmax(y, axis=1)
    return X, y

X_test, y_test = extract_features(test_gen)

# Load Random Forest model
rf_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest_gi.pkl')
rf = joblib.load(rf_path)

# Predict
rf_pred = rf.predict(X_test)

# Evaluation
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred))
cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap='Blues', annot=False)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
