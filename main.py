# main.py (FastAPI app to serve image classifier from Supabase)

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import models, transforms
from PIL import Image
import io
import requests

app = FastAPI()

# === Load the model from Supabase ===
MODEL_URL = "https://jjkpszszpdcnbcweslno.supabase.co/storage/v1/object/public/model/model.pt"
response = requests.get(MODEL_URL)
buffer = io.BytesIO(response.content)

model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load(buffer, map_location="cpu"))
model.eval()

# === Define image transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Prediction endpoint ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(output, dim=1).item()

    return JSONResponse({"prediction": prediction})