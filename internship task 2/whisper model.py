import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import warnings
import ollama
import json

# Ignore basic warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# ==========================================
# MODULE 1: The Ears (Live Microphone & Transcription)
# ==========================================
def record_and_transcribe(filename="live_audio.wav", duration=8, sample_rate=44100):
    print(f"\n🎤 MIC IS LIVE! Please speak now... (Listening for {duration} seconds)")
    
    # Record audio from your laptop mic
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    print("✅ Recording complete! Processing audio...\n")
    write(filename, sample_rate, recording)
    
    # Transcribe audio using Whisper
    model = whisper.load_model("base") 
    result = model.transcribe(filename)
    
    return result["text"]

# ==========================================
# MODULE 2: The Brain (Ollama Translation & Detection)
# ==========================================
def local_detect_and_translate(raw_text):
    prompt = f"""
    You are an expert multilingual analysis engine. Analyze the text below. 
    It may contain any global language, regional Indian languages (like Marathi, Bengali, Tamil, Telugu, Hindi), English, or a mix of them.
    
    Your job is to:
    1. Fix any minor spelling/transcription errors made by the audio listener (e.g., recognizing Romanized regional words).
    2. Detect the exact primary language(s) used. Do not default to Hindi unless it is actually Hindi.
    3. Translate the entire meaning purely into formal English.
    
    Return a strict JSON object with these exact keys:
    - 'detected_language' (string)
    - 'translated_text' (string, formal English)
    - 'confidence_score' (integer between 0 and 100)
    
    Text to analyze: {raw_text}
    """
    try:
        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}], format='json')
        return json.loads(response['message']['content'])
    except Exception as e:
        return {"error": f"Translation failed: {e}"}

# ==========================================
# THE LIVE TRANSLATOR FLOW
# ==========================================
print("Starting Live Multilingual Voice Translator...\n")

# 1. Listen to whatever the user says (Set to 10 seconds)
raw_transcription = record_and_transcribe(duration=10)
print(f"📝 Raw Whisper Transcription: {raw_transcription}")

# 2. Translate and fix the text
print("🧠 Ollama is analyzing and translating...")
translation_result = local_detect_and_translate(raw_transcription)

# 3. Final Output
final_record = {
    "step_1_raw_audio_text": raw_transcription,
    "step_2_translation": translation_result
}

print("\n=== FINAL TRANSLATION REPORT ===")
print(json.dumps(final_record, indent=4))