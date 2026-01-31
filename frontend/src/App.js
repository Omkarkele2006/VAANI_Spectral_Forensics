import React from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import UploadComponent from './components/UploadComponent'; // Import the new block

function App() {
  return (
    <div className="container mt-5">
      {/* Header */}
      <div className="text-center mb-5">
        <h1 className="display-4 fw-bold text-primary">VAANI</h1>
        <p className="lead text-secondary">
          AI-Based Spectral Forensics for Synthetic Voice Detection
        </p>
        <hr className="w-50 mx-auto" />
      </div>

      {/* Main Upload Interface */}
      <div className="row justify-content-center">
        <div className="col-md-8">
            <UploadComponent />
        </div>
      </div>
    </div>
  );
}

export default App;