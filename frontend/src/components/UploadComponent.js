import React, { useState } from 'react';
import axios from 'axios';

const UploadComponent = () => {
    // STATE: React's way of remembering things
    const [selectedFile, setSelectedFile] = useState(null);
    const [status, setStatus] = useState(""); // "Uploading...", "Success", etc.
    const [result, setResult] = useState(null); // Stores the backend response (JSON)
    const [previewUrl, setPreviewUrl] = useState(null); // To show the spectrogram

    // 1. Handle File Selection
    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setSelectedFile(file);
            setStatus(""); // Reset status
            setResult(null); // Reset previous results
            setPreviewUrl(null);
        }
    };

    // 2. Handle the "Analyze" Button Click
    const handleUpload = async () => {
        if (!selectedFile) {
            alert("Please select a file first!");
            return;
        }

        // Prepare the form data (like an envelope)
        const formData = new FormData();
        formData.append("file", selectedFile);

        setStatus("Analyzing audio artifacts... (This may take 5-10 seconds)");

        try {
            // Send POST request to Flask Backend (Port 5000)
            const response = await axios.post("http://127.0.0.1:5000/analyze", formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });

            // Handle Success
            setStatus("Analysis Complete!");
            setResult(response.data); // Save the JSON response
            
            // Construct the image URL. Note: In production, we'd serve static files differently.
            // For now, we assume the backend saves it in a place we can reach, 
            // OR we rely on the backend sending a valid path.
            // *Crucial Fix for Localhost*: We need to fetch the image properly.
            // For this specific lab setup, let's just show the JSON data first to prove connection.
            
        } catch (error) {
            console.error("Upload Error:", error);
            setStatus("Error: Could not connect to the Forensic Engine.");
        }
    };

    return (
        <div className="card shadow-lg p-4">
            <h3 className="card-title text-center mb-4">EVIDENCE UPLOAD</h3>
            
            {/* File Input */}
            <div className="mb-3">
                <input 
                    className="form-control form-control-lg" 
                    type="file" 
                    accept=".mp3, .wav"
                    onChange={handleFileChange} 
                />
            </div>

            {/* Analyze Button */}
            <div className="d-grid gap-2">
                <button 
                    className="btn btn-primary btn-lg" 
                    onClick={handleUpload}
                    disabled={!selectedFile}
                >
                    🔍 Analyze Spectrogram
                </button>
            </div>

            {/* Status Message */}
            {status && <div className="alert alert-info mt-3">{status}</div>}

            {/* Results Display Area */}
            {result && result.result ? (
                <div className="mt-4 p-3 bg-light border rounded">
                    <h4 className="text-dark">Forensic Report</h4>
                    <hr />
                    <p><strong>Filename:</strong> {result.filename}</p>
                    <p><strong>Prediction:</strong> <span className={result.result.label === "Synthetic" ? "text-danger fw-bold" : "text-success fw-bold"}>{result.result.label}</span></p>
                    <p><strong>Confidence:</strong> {result.result.confidence}</p>
                    <p><strong>Note:</strong> {result.result.note}</p>
                    
                    <small className="text-muted d-block mt-3">Raw Server Response:</small>
                    <pre className="small text-muted">{JSON.stringify(result, null, 2)}</pre>
                </div>
            ) : result && (
                // Fallback if result exists but AI data is missing (handles the crash case)
                <div className="mt-4 p-3 bg-warning bg-opacity-25 border border-warning rounded">
                    <h4 className="text-dark">⚠️ Processing Error</h4>
                    <p>The server responded, but the forensic analysis failed.</p>
                    <pre className="small text-muted">{JSON.stringify(result, null, 2)}</pre>
                </div>
            )}
        </div>
    );
};

export default UploadComponent;