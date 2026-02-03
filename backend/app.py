import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- IMPORTS ---
from utils.audio_processor import generate_spectrogram
from database.db import db
from database.models import User, AuditLog
from models.cnn_model import build_model 

# 1. Initialize App & Database
app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = '../data_store/uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///vaani.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db.init_app(app)
with app.app_context():
    db.create_all()

# 2. Load the Hybrid AI Model
try:
    # Build the empty architecture first
    global_model = build_model()
    
    # Check if we have a trained brain saved
    model_path = 'vaani_model.h5'
    
    if os.path.exists(model_path):
        print(f"📂 Found trained model weights at: {model_path}")
        global_model.load_weights(model_path)
        print("✅ Trained Intelligence Loaded Successfully!")
    else:
        print("⚠️ No trained model found. Using random weights (Demo Mode).")
        print("   (Run 'python -m models.train_model' to train the system)")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    # Fallback to empty model so server doesn't crash
    global_model = build_model()

# 3. Helper Function (This was missing!)
def prepare_image(image_path):
    """Prepares the spectrogram for the AI model."""
    # Load as Grayscale, Resize to 128x128
    img = load_img(image_path, color_mode='grayscale', target_size=(128, 128))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize (0 to 1)
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 1)
    return img_array

# 4. The Analysis Route
@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            # A. Save Audio
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # B. Generate Spectrogram
            image_filename = filename.replace('.', '_') + '_spec.png'
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            
            result_path = generate_spectrogram(file_path, image_path)
            
            if not result_path:
                 return jsonify({"error": "Spectrogram generation failed"}), 500

            # --- REPLACEMENT FOR SECTION C IN APP.PY ---

            # C. Run AI Prediction
            processed_image = prepare_image(image_path)
            prediction_value = global_model.predict(processed_image)[0][0]
            
            # Keras Alphabetical Order: 0 = Fake, 1 = Real
            
            if prediction_value > 0.5:
                # Closer to 1.0 -> Real
                label = "Real"
                confidence = prediction_value * 100
            else:
                # Closer to 0.0 -> Synthetic
                label = "Synthetic"
                confidence = (1 - prediction_value) * 100
            
            # -------------------------------------------
            
            # D. Log to Database
            new_log = AuditLog(
                user_id=1,
                filename=filename,
                file_hash="PENDING_HASH",
                prediction=label,
                confidence_score=round(confidence, 2)
            )
            db.session.add(new_log)
            db.session.commit()

            # E. Return Full Report
            return jsonify({
                "message": "Analysis Complete",
                "filename": filename,
                "spectrogram": image_path,
                "result": {
                    "label": label,
                    "confidence": f"{confidence:.2f}%",
                    "note": "Analysis performed by VAANI Hybrid CRNN Engine"
                }
            }), 200

        except Exception as e:
            print(f"Server Error: {e}")
            return jsonify({"error": f"Processing Failed: {str(e)}"}), 500

if __name__ == '__main__':
    print("🚀 VAANI Forensic Server is starting...")
    app.run(debug=True, port=5000)