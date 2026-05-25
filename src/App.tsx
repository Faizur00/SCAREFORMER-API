import { useState } from 'react';
import { Folder, FileCode, Server, Terminal, Download, Github, CheckCircle2, Copy, FileText } from 'lucide-react';

const FASTAPI_FILES = {
  'main.py': `import os
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

app = FastAPI(
    title="Scar Classification API",
    description="Classifies scar images into Hypertrophic / Keloid via TFLite.",
    version="1.0.0"
)

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

@app.on_event("startup")
async def load_model():
    global interpreter, input_details, output_details
    if os.path.exists(MODEL_PATH):
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model missing at {MODEL_PATH}")

def preprocess_image(image_bytes: bytes, target_shape):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    height, width = target_shape[1], target_shape[2]
    image = image.resize((width, height))
    
    image_np = np.array(image, dtype=np.float32)
    image_np = image_np / 255.0  # Normalize [0, 1]
    image_np = np.expand_dims(image_np, axis=0)
    
    return image_np

@app.get("/")
def read_root():
    return {"status": "active"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if interpreter is None:
        raise HTTPException(status_code=503, detail="Model missing.")
        
    image_bytes = await file.read()
    target_shape = input_details[0]['shape']
    input_data = preprocess_image(image_bytes, target_shape)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    scores = output_data[0]
    prediction_idx = np.argmax(scores)
    
    return {
        "prediction": CLASS_NAMES[prediction_idx],
        "confidence": float(scores[prediction_idx]),
        "all_scores": {
            "Hypertrophic": float(scores[0]),
            "Keloid": float(scores[1])
        }
    }
`,
  'requirements.txt': `fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
numpy==1.26.2
Pillow==10.1.0
tflite-runtime==2.14.0`,
  'Dockerfile': `FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p model
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]`,
  'README.md': `# Scar Classification API

Deployment ready FastAPI project for your TFLite Model.

1. Ensure your model is at \`/model/scar_model.tflite\`
2. Push to GitHub
3. Link the repository to Railway`,
};

export default function App() {
  const [activeTab, setActiveTab] = useState<'code' | 'instructions'>('instructions');
  const [activeFile, setActiveFile] = useState<keyof typeof FASTAPI_FILES>('main.py');
  const [copied, setCopied] = useState(false);

  const copyToClipboard = () => {
    navigator.clipboard.writeText(FASTAPI_FILES[activeFile]);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="min-h-screen bg-neutral-900 text-neutral-100 font-sans flex flex-col items-center">
      <div className="w-full max-w-5xl px-6 py-12">
        
        {/* Header Section */}
        <header className="mb-10 text-center">
          <div className="inline-flex items-center justify-center p-4 bg-emerald-500/10 rounded-full mb-6">
            <Server className="w-8 h-8 text-emerald-400" />
          </div>
          <h1 className="text-3xl font-medium tracking-tight mb-3">Scar Classification API</h1>
          <p className="text-neutral-400 max-w-xl mx-auto leading-relaxed">
            I've generated a full production-ready FastAPI directory in this workspace configured with Docker for direct Railway deployment.
          </p>
        </header>

        {/* Navigation Tabs */}
        <div className="flex border-b border-neutral-800 mb-8">
          <button
            onClick={() => setActiveTab('instructions')}
            className={`px-6 py-4 font-medium text-sm transition-colors relative ${activeTab === 'instructions' ? 'text-white' : 'text-neutral-500 hover:text-neutral-300'}`}
          >
            How to Deploy
            {activeTab === 'instructions' && (
              <div className="absolute bottom-0 left-0 w-full h-0.5 bg-emerald-500 rounded-t-full" />
            )}
          </button>
          <button
            onClick={() => setActiveTab('code')}
            className={`px-6 py-4 font-medium text-sm transition-colors relative ${activeTab === 'code' ? 'text-white' : 'text-neutral-500 hover:text-neutral-300'}`}
          >
            Generated Files
            {activeTab === 'code' && (
              <div className="absolute bottom-0 left-0 w-full h-0.5 bg-emerald-500 rounded-t-full" />
            )}
          </button>
        </div>

        {/* Content Area */}
        {activeTab === 'instructions' ? (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="col-span-2 space-y-8">
              <div className="bg-neutral-800/50 rounded-2xl p-8 border border-neutral-800">
                <h2 className="text-xl font-medium mb-6 flex items-center gap-3">
                  <span className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-500/20 text-emerald-400 text-sm font-bold">1</span>
                  Export Project to GitHub
                </h2>
                <p className="text-neutral-400 mb-6 leading-relaxed">
                  The FastAPI application files are sitting in the <code>/fastapi-app/</code> directory of this workspace right now. Use AI Studio's export functionality to push them directly to a GitHub repository.
                </p>
                <div className="flex items-start gap-4 p-4 rounded-xl bg-neutral-900 border border-neutral-800">
                  <Download className="w-6 h-6 text-neutral-500 shrink-0" />
                  <div>
                    <strong className="block text-white mb-1">Click the Settings/Menu icon at the top right</strong>
                    <span className="text-sm text-neutral-400">Select "Export to GitHub" or "Download ZIP".</span>
                  </div>
                </div>
              </div>

              <div className="bg-neutral-800/50 rounded-2xl p-8 border border-neutral-800">
                <h2 className="text-xl font-medium mb-6 flex items-center gap-3">
                  <span className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-500/20 text-emerald-400 text-sm font-bold">2</span>
                  Add your TFLite Model
                </h2>
                <p className="text-neutral-400 mb-4 leading-relaxed">
                  Before deploying, you need to add your trained model binary to the correct directory structure within your exported codebase.
                </p>
                <div className="p-4 bg-neutral-900 rounded-xl font-mono text-sm text-emerald-400 border border-neutral-800">
                   mkdir model <br />
                   # Place your .tflite model inside this directory <br />
                   mv MyModel.tflite model/scar_model.tflite
                </div>
              </div>

              <div className="bg-neutral-800/50 rounded-2xl p-8 border border-neutral-800">
                <h2 className="text-xl font-medium mb-6 flex items-center gap-3">
                  <span className="flex items-center justify-center w-8 h-8 rounded-full bg-emerald-500/20 text-emerald-400 text-sm font-bold">3</span>
                  Deploy to Railway
                </h2>
                <p className="text-neutral-400 mb-6 leading-relaxed">
                  Railway will automatically detect the provided <code>Dockerfile</code>.
                </p>
                <ul className="space-y-4 text-neutral-300">
                  <li className="flex items-center gap-3">
                    <CheckCircle2 className="w-5 h-5 text-emerald-500" /> Go to Railway.app and click "New Project"
                  </li>
                  <li className="flex items-center gap-3">
                    <CheckCircle2 className="w-5 h-5 text-emerald-500" /> Select "Deploy from GitHub repo"
                  </li>
                  <li className="flex items-center gap-3">
                    <CheckCircle2 className="w-5 h-5 text-emerald-500" /> Choose the repo where you exported this code
                  </li>
                  <li className="flex items-center gap-3">
                    <CheckCircle2 className="w-5 h-5 text-emerald-500" /> Railway builds the Dockerfile and links your domain automatically!
                  </li>
                </ul>
              </div>
            </div>

            <div className="col-span-1">
              <div className="bg-blue-500/10 border border-blue-500/20 rounded-2xl p-6">
                <h3 className="font-medium text-blue-400 mb-3 flex items-center gap-2">
                  <Terminal className="w-4 h-4" /> Endpoint Specs
                </h3>
                <p className="text-sm text-blue-300/80 mb-6 leading-relaxed">
                  Once deployed, your API will respond to the following endpoints.
                </p>

                <div className="space-y-4">
                  <div className="p-3 bg-neutral-900 rounded-lg border border-neutral-800">
                    <div className="text-xs font-bold text-emerald-400 mb-1">GET /</div>
                    <div className="text-xs text-neutral-400">Healthcheck & model status.</div>
                  </div>
                  <div className="p-3 bg-neutral-900 rounded-lg border border-neutral-800">
                    <div className="text-xs font-bold text-blue-400 mb-1">POST /predict/</div>
                    <div className="text-xs text-neutral-400 mb-2">Requires form-data: <code>file=image.jpg</code></div>
                    <div className="text-xs font-mono text-neutral-500 mt-2 p-2 bg-black rounded">
                      {'{ "prediction": "Hyper...", "confidence": 0.95 }'}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex h-[600px] border border-neutral-800 rounded-2xl overflow-hidden bg-neutral-900">
            {/* Sidebar File Tree */}
            <div className="w-64 bg-neutral-950 border-r border-neutral-800 p-4">
              <div className="flex items-center gap-2 text-sm font-medium text-neutral-400 mb-4 px-2">
                <Folder className="w-4 h-4" /> /fastapi-app
              </div>
              <nav className="space-y-1">
                {Object.keys(FASTAPI_FILES).map((fileName) => (
                  <button
                    key={fileName}
                    onClick={() => setActiveFile(fileName as keyof typeof FASTAPI_FILES)}
                    className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                      activeFile === fileName 
                        ? 'bg-emerald-500/10 text-emerald-400 font-medium' 
                        : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-800/50'
                    }`}
                  >
                    {fileName.endsWith('.py') ? <FileCode className="w-4 h-4" /> : <FileText className="w-4 h-4" />}
                    {fileName}
                  </button>
                ))}
              </nav>
            </div>

            {/* Code Viewer Area */}
            <div className="flex-1 flex flex-col bg-[#0d0d0d]">
              <div className="h-14 border-b border-neutral-800 flex items-center justify-between px-6 bg-neutral-900/50">
                <div className="text-sm font-mono text-neutral-400">{activeFile}</div>
                <button 
                  onClick={copyToClipboard}
                  className="flex items-center gap-2 text-xs font-medium text-neutral-400 hover:text-white transition-colors"
                >
                  {copied ? <CheckCircle2 className="w-4 h-4 text-emerald-500" /> : <Copy className="w-4 h-4" />}
                  {copied ? 'Copied!' : 'Copy Code'}
                </button>
              </div>
              <div className="flex-1 overflow-auto p-6">
                <pre className="text-sm font-mono text-neutral-300 leading-relaxed">
                  <code>{FASTAPI_FILES[activeFile]}</code>
                </pre>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

