# Scar Image Classification API

A FastAPI service that classifies scar images as **Hypertrophic** or **Keloid** using a TFLite model. Supports local and remote access via ngrok tunnel.

---

## Quick Start

```bash
# One-time setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the API with public tunnel
bash run.sh
```

The script prints a public URL like:
```
https://xxxx-xxx-xxx.ngrok-free.dev
```

Share that URL with anyone to give them access to your API.

---

## Using the Public URL

### Check if the API is running

Open the URL in a browser or use curl:

```
GET https://xxxx-xxx-xxx.ngrok-free.dev/
```

Response:
```json
{
  "status": "active",
  "message": "Scar Classification API is running.",
  "model_loaded": true
}
```

### Get a scar prediction

Send a POST request with an image file:

```
POST https://xxxx-xxx-xxx.ngrok-free.dev/predict/
Content-Type: multipart/form-data

file: <scar-image.jpg>
```

**curl example:**
```bash
curl -X POST https://xxxx-xxx-xxx.ngrok-free.dev/predict/ \
  -F "file=@scar_image.jpg"
```

Response:
```json
{
  "prediction": "Keloid",
  "confidence": 0.8941,
  "all_scores": {
    "Hypertrophic": 0.1059,
    "Keloid": 0.8941
  }
}
```

---

## API Endpoints

### GET /

Health check. Returns whether the model is loaded.

### POST /predict/

Classifies a scar image.

| Parameter | Type | Location | Description |
| :--- | :--- | :--- | :--- |
| `file` | Binary | Form Data | Image file (JPEG, PNG, etc.) |

Possible errors:
- `400` - File is not an image
- `503` - Model file not found
- `500` - Inference error

---

## Ngrok Tunnel Setup (One Time)

### Install ngrok

**Arch Linux:**
```bash
yay -S ngrok
```

**Debian/Ubuntu:**
```bash
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok
```

**macOS:**
```bash
brew install ngrok
```

### Authenticate

Sign up at https://ngrok.com (free), then:

```bash
ngrok config add-authtoken YOUR_TOKEN
```

### Free Plan Limits

| Limit | Value |
| :--- | :--- |
| Connections per minute | 40 |
| Tunnel lifetime | 2 hours (new URL on restart) |
| Custom subdomains | Paid only |

---

## Stopping the Server

Press `Ctrl+C` in the terminal where `run.sh` is running.
