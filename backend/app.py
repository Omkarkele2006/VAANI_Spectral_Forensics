import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from utils.audio_processor import generate_spectrogram

# 1. Initialize Flask App
app = Flask(__name__)
CORS(app)  # Allow React (Frontend) to talk to Flask (Backend)

# 2. Configuration
UPLOAD_FOLDER = '../data_store/uploads'
REPORT_FOLDER = '../data_store/reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 3. The API Route (The "Doorway")
@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """
    Receives an audio file, converts it to a spectrogram, 
    and (in the future) will run the AI detection.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # A. Secure the filename (prevents hacking via file names)
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # B. Save the file temporarily
        file.save(file_path)
        
        # C. Generate Spectrogram (The function we wrote earlier!)
        # We save the image in the same folder as the audio for now
        image_filename = filename.replace('.', '_') + '_spec.png'
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        
        generate_spectrogram(file_path, image_path)

        # D. Return JSON response
        return jsonify({
            "message": "File processed successfully",
            "filename": filename,
            "spectrogram_path": image_path,
            "status": "Analysis Complete (AI Model Pending)"
        }), 200

# 4. Run the Server
if __name__ == '__main__':
    print("VAANI Forensic Server is starting on Port 5000...")
    app.run(debug=True, port=5000)