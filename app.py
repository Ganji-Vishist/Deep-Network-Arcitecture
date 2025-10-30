# app.py
from flask import Flask, request, render_template_string
from model import DamageCNN
from utils import preprocess_image, predict_damage
import torch
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model (for demo, random weights)
model = DamageCNN()
# If you have a trained model, load it:
# model.load_state_dict(torch.load('model_weights.pth', map_location='cpu'))

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Building Damage Assessment</title>
    <style>
        body { font-family: Arial; text-align: center; margin-top: 50px; }
        .container { width: 400px; margin: auto; }
        img { width: 100%; margin-top: 10px; border-radius: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>🏚️ Building Damage Assessment</h2>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" required><br><br>
            <button type="submit">Predict Damage</button>
        </form>
        {% if prediction %}
            <h3>Prediction: {{ prediction }}</h3>
            <img src="{{ image_url }}" alt="Uploaded Image">
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            image_tensor = preprocess_image(filepath)
            prediction = predict_damage(model, image_tensor)
            image_url = filepath

    return render_template_string(HTML_PAGE, prediction=prediction, image_url=image_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
