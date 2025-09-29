from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io

#load at startup
model = YOLO("./model/last.pt") 

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    #image reading
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    #run inference
    results = model(img)
    r = results[0]

    probs = r.probs.data.tolist() 
    names = r.names                 

    response = {
        names[i]: float(probs[i]) for i in range(len(probs))
    }

    return JSONResponse(content=response)
