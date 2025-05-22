import io
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
from Main import FARIS
from gtts import gTTS
import uuid
import os
from fastapi.responses import FileResponse
from fastapi import FastAPI
import uvicorn

FEATURE_NAMES = ['Cu', 'Fe', 'Cr', 'Al', 'Si', 'Pb', 'Sn', 'Ni', 'Na', 'B', 'P', 'Zn',
                'Mo', 'Ca', 'Mg', 'TBN', 'V100', 'V40', 'OXI', 'TAN', 'water_flag', 'antifreeze_flag']

faris = FARIS()
FARIS_API = FastAPI()


class InputData(BaseModel):
    features: List[float]  # 22 floats

@FARIS_API.post("/predict/")
def predict(data: InputData):

    if len(data.features) != 22:
        return {"error": "Exactly 22 features are required."}

    # Convert to numpy array
    X = pd.DataFrame(np.array(data.features).reshape(1, -1), columns=FEATURE_NAMES)

    # Model prediction
    output = faris.predict(X)
    text1 = f"Prediction: {output[0]}"
    text2 = "Thank you for using our model."

    # Generate audio using gTTS
    audio_text = f"{text1}. {text2}"
    tts = gTTS(audio_text)
    audio_filename = f"output_{uuid.uuid4()}.mp3"
    tts.save(audio_filename)

    return {
        "cls": text1,
        "reason": text2,
        "audio_url": f"/audio/{audio_filename}"
    }

@FARIS_API.get("/audio/{filename}")
def get_audio(filename: str):
    file_path = filename

    if not os.path.exists(file_path):
        return {"error": "File not found"}
    
    return FileResponse(path=file_path, media_type="audio/mpeg", filename=filename)



if  __name__ == "__main__":
    uvicorn.run(FARIS_API, host='127.0.0.1', port=8001)
