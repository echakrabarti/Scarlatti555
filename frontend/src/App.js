import { useState, useRef } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const mimeTypeRef = useRef("");

  const getMimeType = () => {
    const types = [
      "audio/webm",
      "audio/mp4",
      "audio/ogg",
      "audio/wav",
    ];
    return types.find(type => MediaRecorder.isTypeSupported(type)) || "";
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const mimeType = getMimeType();
      mimeTypeRef.current = mimeType;

      const mediaRecorder = new MediaRecorder(
        stream,
        mimeType ? { mimeType } : {}
      );

      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.start();
      setIsRecording(true);
      setStatus("recording");
      setResults([]);
      setError("");

    } catch (err) {
      setError("Microphone access denied. Please allow microphone access.");
      setStatus("error");
    }
  };

  const stopRecording = async () => {
    const mediaRecorder = mediaRecorderRef.current;
    if (!mediaRecorder) return;

    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach((t) => t.stop());
    setIsRecording(false);
    setStatus("processing");

    mediaRecorder.onstop = async () => {
      const blob = new Blob(
        chunksRef.current,
        { type: mimeTypeRef.current || "audio/webm" }
      );

      const formData = new FormData();
      formData.append("file", blob, "hum.webm");

      try {
        const response = await axios.post(
          "http://127.0.0.1:8000/identify",
          formData,
          { headers: { "Content-Type": "multipart/form-data" } }
        );
        setResults(response.data.results);
        setStatus("done");
      } catch (err) {
        setError("Could not identify — try humming more clearly.");
        setStatus("error");
      }
    };
  };

  const handleButton = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Scarlatti555</h1>
        <p>Identify classical music by humming</p>
      </header>

      <main className="main">
        <button
          className={`record-btn ${isRecording ? "recording" : ""}`}
          onClick={handleButton}
          disabled={status === "processing"}
        >
          {status === "processing"
            ? "Identifying..."
            : isRecording
            ? "Stop"
            : "Hold to Hum"}
        </button>

        {status === "recording" && (
          <p className="recording-hint">Hum your melody... press Stop when done</p>
        )}

        {error && <p className="error">{error}</p>}

        {results.length > 0 && (
          <div className="results">
            <h2>Results</h2>
            {results.map((r, i) => (
              <div key={i} className="result-item">
                <div className="result-meta">
                  <span className="rank">#{i + 1}</span>
                  <span className="title">{r.title}</span>
                  <span className="composer">{r.composer}</span>
                  <span className="score">{(r.score * 100).toFixed(1)}%</span>
                </div>
                <div className="bar-bg">
                  <div
                    className="bar-fill"
                    style={{ width: `${r.score * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;