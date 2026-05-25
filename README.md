# Scar Classification API

A FastAPI application designed straight for TFLite inference, configured to predict between "Hypertrophic" and "Keloid" scar images.

## Project Structure

- `main.py` - Core FastAPI web application handling pre-processing, TFLite inference, and API routing.
- `requirements.txt` - Required python versions and packages, optimized out with `tflite-runtime` instead of the full tensorflow library for smaller build sizes.
- `Dockerfile` - Configuration for Railway deployment.

## Next Steps

1. Create a `model` directory at the root of this project:
   ```bash
   mkdir model
   ```
2. Place your TFLite model in the `model` folder and name it exactly `scar_model.tflite`.
   - Your final path should be: `/model/scar_model.tflite`
3. Push this directory to your GitHub Repository.
4. Go to **Railway.app**, click **New > GitHub Repo**. It will detect the Dockerfile and deploy your model immediately.

## Testing Locally
If you want to run this on your own machine before pushing to Railway:
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
Once it starts, you can visit the auto-generated documentation at `/docs` to upload an image and test it!
