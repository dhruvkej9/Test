from fastapi import FastAPI
import os

app = FastAPI()

@app.get("/")
def get_accuracy():
    acc_path = os.path.join(os.path.dirname(__file__), "..", "backend", "accuracy.txt")
    if os.path.exists(acc_path):
        try:
            with open(acc_path, "r") as f:
                acc = f.read().strip()
            return {"accuracy": float(acc)}
        except Exception:
            pass
    return {"accuracy": None, "error": "Accuracy not available. Train the model first."}

# For Vercel serverless compatibility
def handler(request):
    return app(request)