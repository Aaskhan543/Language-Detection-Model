import whisper
import os
import warnings

# Suppress some common Whisper warnings
warnings.filterwarnings("ignore", category=UserWarning)

class AudioProcessor:
    def __init__(self, model_size="tiny.en"):
        print(f"[System] Loading Whisper '{model_size}' model...")
        self.model = whisper.load_model(model_size)

    def transcribe_file(self, file_path: str) -> str:
        """
        Takes a saved audio file from the web frontend and transcribes it.
        """
        print("[System] Transcribing incoming web audio...")
        
        # Custom dictionary to help Whisper understand specific terms
        custom_vocabulary = "Mohd Aas Khan, Galgotias University, B.Tech, Noida, AI, Machine Learning, Python, LLM, UPSC."
        
        result = self.model.transcribe(
            file_path,
            initial_prompt=custom_vocabulary
        )
        
        transcribed_text = result["text"].strip()
        print(f"[Transcription]: {transcribed_text}")
        return transcribed_text