import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import joblib

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load Random Forest model and class indices
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'random_forest_gi.pkl')
model = joblib.load(MODEL_PATH)

# You may want to save class_indices from your data pipeline for real deployment
# For now, use a placeholder list of class names (update with your actual class names)
CLASS_NAMES = [
    'Accessory tools', 'Angiectasia', 'Barretts esophagus', 'Blood in lumen', 'Cecum',
    'Colon diverticula', 'Colon polyps', 'Colorectal cancer', 'Duodenal bulb', 'Dyed-lifted-polyps',
    'Dyed-resection-margins', 'Erythema', 'Esophageal varices', 'Esophagitis', 'Gastric polyps',
    'Gastroesophageal_junction_normal z-line', 'Ileocecal valve', 'Mucosal inflammation large bowel',
    'Normal esophagus', 'Normal mucosa and vascular pattern in the large bowel', 'Normal stomach',
    'Pylorus', 'Resected polyps', 'Resection margins', 'Retroflex rectum', 'Small bowel_terminal ileum', 'Ulcer'
]

IMG_SIZE = (224, 224)


# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction=None)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction='No file part')
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction='No selected file')
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # Preprocess image
        img = Image.open(filepath).convert('RGB').resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, -1)
        # Predict
        proba = model.predict_proba(img_array)[0]
        pred = np.argmax(proba)
        pred_class = CLASS_NAMES[pred] if pred < len(CLASS_NAMES) else str(pred)
        confidence = round(100 * proba[pred], 2)
        prediction = f"Predicted Disease: {pred_class} (Confidence: {confidence}%)"
        os.remove(filepath)
        return render_template('index.html', prediction=prediction)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
