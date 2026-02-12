import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [symptoms, setSymptoms] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return alert("Upload an image");

    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("symptoms", symptoms);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/predict/",
        formData
      );
      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert("Backend error");
    }

    setLoading(false);
  };

  return (
    <div className="container">
      <div className="background-blur">
      <span style={{ width: "200px", height: "200px", left: "10%", top: "70%" }}></span>
      <span style={{ width: "300px", height: "300px", left: "70%", top: "60%" }}></span>
      <span style={{ width: "150px", height: "150px", left: "40%", top: "80%" }}></span>
      <span style={{ width: "250px", height: "250px", left: "80%", top: "30%" }}></span>
      <span style={{ width: "180px", height: "180px", left: "20%", top: "40%" }}></span>
      </div>
      <div className="hero">
        <h1>AI Skin Diagnostics</h1>
        <p className="tagline">
           Intelligent multi-modal analysis for early and accurate skin disease detection.
        </p>

        <div className="card">
          <form onSubmit={handleSubmit}>
            <input
              type="file"
              onChange={(e) => setFile(e.target.files[0])}
            />

            <textarea
              placeholder="Describe your symptoms..."
              value={symptoms}
              onChange={(e) => setSymptoms(e.target.value)}
            />

            <button type="submit">
              {loading ? "Analyzing..." : "Analyze"}
            </button>
          </form>
        </div>

        {result && (
          <div className={`result ${result.severity_color}`}>
            <h2>⚠ {result.disease}</h2>
            <p><strong>Confidence:</strong> {result.confidence}%</p>
<p>{result.consultation_message}</p>

          </div>
        )}
      </div>
    </div>
  );
}

export default App;
