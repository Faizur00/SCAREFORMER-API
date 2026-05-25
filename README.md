# Scar Image Classification API

A FastAPI service designed to run local, real-time inference on a TensorFlow Lite (TFLite) image classification model, distinguishing between target categories: **Hypertrophic** and **Keloid** scars.

---

## API Capabilities and Design

* **Dual-Class Classification:** Specifically maps images of scars to `"Hypertrophic"` or `"Keloid"` outputs.
* **TFLite Engine Integration:** Powered by a fast, optimized `tflite-runtime` interpreter, offering rapid responses without the memory footprint of full TensorFlow.
* **Auto-Pre-processing:** Automatically maps input upload files (images), dynamic sizes to match the pre-defined target input shape configured inside your TFLite model, and maps pixel arrays to `[0.0, 1.0]` floating-point values.
* **Cross-Origin Access (CORS):** Pre-equipped with liberal CORS access middlewares so JavaScript frontends can connect without encountering security blocks.

---

## API Endpoints Reference

### 1. Root / Service Health Status
Checks if the underlying web application service is running and whether the custom `.tflite` model was successfully imported into the runtime memory buffer.

* **Method:** `GET`
* **Path:** `/`
* **Response Signature (`application/json`):**
  ```json
  {
    "status": "active",
    "message": "Scar Classification API is running.",
    "model_loaded": true,
    "instructions": "Send a POST request with an image file to /predict/."
  }
  ```

---

### 2. File Inference & Scar Categorization
Submit raw file records directly to have the system parse, pre-process, calculate predictions, and return classifications.

* **Method:** `POST`
* **Path:** `/predict/`
* **Content-Type:** `multipart/form-data`
* **Payload Parameters:**
  
  | Field Name | Type | Source | Definition |
  | :--- | :--- | :--- | :--- |
  | `file` | `Binary (File upload)` | Form Data | The scar image to classify (`JPEG`, `PNG`, etc.) |

* **Response Signature (`application/json`):**
  ```json
  {
    "prediction": "Keloid",
    "confidence": 0.8941295742988586,
    "all_scores": {
      "Hypertrophic": 0.10587042570114136,
      "Keloid": 0.8941295742988586
    }
  }
  ```

* **Possible Non-200 Status Codes:**
  * `400 Bad Request` - Provided file type is not a valid image format.
  * `503 Service Unavailable` - Model binary is missing from the server root directory layout.
  * `500 Internal Server Error` - Underlying numpy matrix operations or math processing failed.

---

## Web Frontend Integration Guide

Below are production-ready recipes for frontend developers accessing your deployed API.

### 1. Asynchronous JavaScript Standard HTTP Integration

```javascript
/**
 * Analyze an image using the Scar Classification API.
 * @param {File} imageFile - Local image file object acquired from an HTML file input element.
 * @param {string} apiBaseUrl - Base Host URL of your deployed Railway endpoint.
 * @returns {Promise<object>} Returns categorization and probability weights.
 */
async function classifyScarImage(imageFile, apiBaseUrl) {
  if (!imageFile) {
    throw new Error('Please select an image file first.');
  }

  // Create standard multipart payload
  const payload = new FormData();
  // Ensure the field name is 'file' to match FastAPI's route requirement
  payload.append('file', imageFile);

  try {
    const targetEndpoint = `${apiBaseUrl.replace(/\/$/, '')}/predict/`;
    
    const response = await fetch(targetEndpoint, {
      method: 'POST',
      body: payload,
      // Note: Leave headers empty. The browser automatically generates 
      // the 'multipart/form-data' boundary code structure.
    });

    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({}));
      throw new Error(errorPayload.detail || `Server responded with status: ${response.status}`);
    }

    const output = await response.json();
    return {
      prediction: output.prediction,
      confidence: output.confidence,
      percentages: {
        hypertrophic: output.all_scores.Hypertrophic * 100,
        keloid: output.all_scores.Keloid * 100
      }
    };
  } catch (error) {
    console.error('Inference Service API Error:', error);
    throw error;
  }
}
```

### 2. Clean React Integration Core Component (Tailwind CSS)

```tsx
import React, { useState } from 'react';

interface PredictionResult {
  prediction: string;
  confidence: number;
  all_scores: {
    Hypertrophic: number;
    Keloid: number;
  };
}

export function ScarInferenceUI({ apiUrl }: { apiUrl: string }) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [data, setData] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const onFileSelected = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setData(null);
      setErr(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setErr(null);

    const fd = new FormData();
    fd.append('file', selectedFile);

    try {
      const response = await fetch(`${apiUrl}/predict/`, {
        method: 'POST',
        body: fd
      });
      if (!response.ok) throw new Error('Model returned an evaluation error.');
      const result: PredictionResult = await response.json();
      setData(result);
    } catch (error: any) {
      setErr(error.message || 'Error executing request.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto p-6 bg-neutral-900 border border-neutral-800 rounded-xl shadow">
      <h2 className="text-lg font-semibold text-neutral-100 mb-4">Scar Classification</h2>
      
      <input 
        type="file" 
        accept="image/*" 
        onChange={onFileSelected} 
        className="block w-full text-sm text-neutral-400 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-neutral-800 file:text-neutral-200 hover:file:bg-neutral-700 cursor-pointer mb-4" 
      />

      {previewUrl && (
        <div className="relative aspect-video rounded-lg overflow-hidden border border-neutral-800 mb-4 bg-black">
          <img src={previewUrl} alt="Scar Preview" className="object-contain w-full h-full" />
        </div>
      )}

      {selectedFile && (
        <button
          onClick={handlePredict}
          disabled={loading}
          className="w-full bg-emerald-600 text-white py-2 rounded-lg font-medium hover:bg-emerald-500 disabled:opacity-50 transition-colors"
        >
          {loading ? 'Evaluating details...' : 'Analyze Image'}
        </button>
      )}

      {err && (
        <div className="mt-4 p-3 bg-red-950/40 border border-red-900 text-red-300 text-xs rounded-lg">
          {err}
        </div>
      )}

      {data && (
        <div className="mt-4 p-4 bg-neutral-950 rounded-lg border border-neutral-800 text-sm">
          <div className="flex justify-between items-center mb-3">
            <span className="text-neutral-400">Class Label:</span>
            <span className="font-bold text-emerald-400">{data.prediction}</span>
          </div>
          <div className="flex justify-between items-center mb-4">
            <span className="text-neutral-400">Confidence Match:</span>
            <span className="font-mono text-neutral-200">{(data.confidence * 100).toFixed(2)}%</span>
          </div>
          
          <div className="space-y-2 border-t border-neutral-800 pt-3 text-xs">
            <div>
              <div className="flex justify-between text-neutral-400 mb-1">
                <span>Hypertrophic:</span>
                <span>{(data.all_scores.Hypertrophic * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-neutral-800 h-1.5 rounded">
                <div 
                  className="bg-emerald-500 h-1.5 rounded transition-all" 
                  style={{ width: `${data.all_scores.Hypertrophic * 100}%` }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-neutral-400 mb-1">
                <span>Keloid:</span>
                <span>{(data.all_scores.Keloid * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-neutral-800 h-1.5 rounded">
                <div 
                  className="bg-emerald-500 h-1.5 rounded transition-all" 
                  style={{ width: `${data.all_scores.Keloid * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
```
