from fastapi import *
from google.cloud import storage
import io
from PIL import Image
import os
import uvicorn
import api_config as config
import func as f
from api_config import Tags
import random
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()
port = int(os.getenv("PORT"))


@app.get("/")
def read_root():
    return {"Welcome to TechWas (Technology Waste)"}


@app.post("/predict/", tags=[Tags.predict])
async def predict(file: UploadFile = File(...)):
    time = f.timer(None)
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg")
    if not extension:
        return "Image must be jpg or jpeg format!"
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image.save(file.filename)
    tf_image = f.preprocess_image(image)
    data_predict = f.predict_image(tf_image)
    os.remove(file.filename)
    return {"predictions": data_predict,
     "time_taken": f.timer(time),
     }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)
