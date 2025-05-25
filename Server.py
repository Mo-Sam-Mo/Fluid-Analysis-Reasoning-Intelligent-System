from pydantic import BaseModel
from typing import List
from Main import FARIS
import os
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import uvicorn


origins = [
    "http://localhost",
    "http://localhost:8080", # If you're running your FlutterFlow web app locally
    "https://faris-7o48hz.flutterflow.app/", # REPLACE with your actual FlutterFlow app's domain
    "https://app.flutterflow.io", # FlutterFlow's development domain
    "*" # FOR DEVELOPMENT ONLY: Allows all origins. REMOVE OR REPLACE FOR PRODUCTION!
]

CLASS_MAPPING = {
            'machine_depreciation': 'Machine Deperciation',
            'water_contamination': 'Water Contamination',
            'dirt_in_oil': 'Dirt in Oil',
            'sludge_formation': 'Sludge Formation',
            'oil_change_needed': 'Oil Consumed',
            'normal': 'Normal'
        }

faris = FARIS()
FARIS_API = FastAPI()


FARIS_API.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers (Content-Type, Authorization, etc.)
)


class InputData(BaseModel):
    features: List[float]  # 22 floats

@FARIS_API.post("/predict/")
def predict(data: InputData):

    if len(data.features) != 22:
        return {"error": "Exactly 22 features are required."}
    
    cls, reasoning, audio = faris.predict(data.features)

    return {
        "cls": CLASS_MAPPING[cls],
        "reason": reasoning.replace("*", ""),
        "audio_url": f"{audio}"
    }


@FARIS_API.get("/audio/{filename}")
def get_audio(filename: str):
    file_path = 'audio/' + filename

    if not os.path.exists(file_path):
        return {"error": "File not found"}
    
    return FileResponse(path=file_path, media_type="audio/mpeg", filename=filename)



if __name__ == "__main__":
    uvicorn.run(FARIS_API, host='127.0.0.1', port=8001)
