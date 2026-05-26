# Scar Image Classification API

A FastAPI service designed to run local, real-time inference on a TensorFlow Lite (TFLite) image classification model, distinguishing between target categories: **Hypertrophic** and **Keloid** scars.

---

## Quick Start (Local Dev)

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the API + ngrok tunnel
bash run.sh
```

You'll see output like:
```
🚀 Starting FastAPI server on port 8000...
🔗 Starting ngrok tunnel to port 8000...
✅ Ngrok tunnel active at: https://xxxx-xx-xxx-xxx-xxx.ngrok-free.app
📋 API Root:   https://xxxx-xx-xxx-xxx-xxx.ngrok-free.app/
📋 Predict:    https://xxxx-xx-xxx-xxx-xxx.ngrok-free.app/predict/
```

Share that `ngrok-free.app` URL with anyone — they can hit your API without running the server locally.

---

## Ngrok Tunneling (Remote Access)

While the FastAPI server runs locally on `http://localhost:8000`, `run.sh` automatically creates a **public ngrok tunnel** so your friends can test the API from anywhere during development.

### Prerequisites

Install ngrok (one-time):

```bash
# Arch Linux
yay -S ngrok
# or
paru -S ngrok

# Linux (Debian/Ubuntu)
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok

# macOS
brew install ngrok
```

### Authentication

Sign up at [ngrok.com](https://ngrok.com) (free), then authenticate:

```bash
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

> **Without authentication**, ngrok still works but shows a banner page before your API. The free plan gives you a random URL each restart and ~40 connections/minute — fine for dev testing.

### How It Works

1. `run.sh` starts **uvicorn** on `localhost:8000`
2. `run.sh` starts **ngrok** via `pyngrok`, creating a secure tunnel to `localhost:8000`
3. The public URL is printed to the console
4. Press `Ctrl+C` to stop both the server and tunnel

### Rate Limits (Free Plan)

| Limit | Value |
| :--- | :--- |
| Connections per minute | 40 |
| Tunnel lifetime | 2 hrs (auto-restart with new URL) |
| Custom subdomains | Paid only |

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

