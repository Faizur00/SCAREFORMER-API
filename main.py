import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Try to import tflite_runtime first, fallback to standard tensorflow if needed
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

app = FastAPI(
    title="Scar Classification API",
    description="API for classifying scar images into Hypertrophic or Keloid using a TFLite model.",
    version="1.0.0"
)

# Allow CORS for frontends to communicate with this backend seamlessly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model/scar_model.tflite"
CLASS_NAMES = ["Hypertrophic", "Keloid"]

interpreter = None
input_details = None
output_details = None

def load_model():
    global interpreter, input_details, output_details
    if interpreter is not None:
        return
    if os.path.exists(MODEL_PATH):
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. Prediction endpoints will return 503 until uploaded.")

def preprocess_image(image_bytes: bytes, target_shape):
    """
    Load image, resize to model's expected shape, and normalize.
    Assumes standard 0-1 normalization for TFLite models.
    Adjust if your model expects -1 to 1.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # target_shape is typically [1, height, width, channels]
    height, width = target_shape[1], target_shape[2]
    image = image.resize((width, height))
    
    image_np = np.array(image, dtype=np.float32)
    image_np = image_np / 255.0  # Normalize to [0, 1]
    image_np = np.expand_dims(image_np, axis=0) # Add batch dimension
    
    return image_np

@app.get("/")
def read_root():
    return {
        "status": "active",
        "message": "Scar Classification API is running.",
        "instructions": "Send a POST request with an image file to /predict/."
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    load_model()
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please ensure scar_model.tflite is in the /model directory.")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        image_bytes = await file.read()
        target_shape = input_details[0]['shape']
        
        # Preprocess the input image
        input_data = preprocess_image(image_bytes, target_shape)
        
        # Set the tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get output scores
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Assuming output is shape [1, 2] representing logic or probabilities 
        # e.g., [[prob_hypertrophic, prob_keloid]]
        scores = output_data[0]
        prediction_idx = np.argmax(scores)
        confidence = float(scores[prediction_idx])
        
        return {
            "prediction": CLASS_NAMES[prediction_idx],
            "confidence": confidence,
            "all_scores": {
                "Hypertrophic": float(scores[0]),
                "Keloid": float(scores[1])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
