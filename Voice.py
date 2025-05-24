import requests
import uuid

MODELS_URL = "https://api.elevenlabs.io/v1/models"

class TextToSpeach():

    def __init__(self, actor='kdmDKE6EkgrWrrykO9Qt', api='sk_37c77b64acacc5b7bd898449cb1a05210f5ddfe634a81bce'):
        self.actor = f"https://api.elevenlabs.io/v1/text-to-speech/{actor}"
        self.api_key = api

    
    def speach(self, text):

        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        data = {
            "text": text,
            "model_id": "eleven_turbo_v2",
            "voice_settings": {
                "stability": 0.7,
                "similarity_boost": 0.8,
                "style": 0.5,
                "speaker_boost": True
            }
        }

        response = requests.post(
            self.actor,
            headers=headers,
            json=data,
            stream=True
        )

        voice_path = f"output_{uuid.uuid4()}.mp3"

        with open('audio/' + voice_path, "wb") as f:
            f.write(response.content)
        
        return voice_path