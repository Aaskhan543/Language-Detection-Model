from flask import Flask, request, jsonify, render_template
import os
import asyncio
import base64
import edge_tts
from dotenv import load_dotenv
from llm_handler import InterviewAgent

# --- SECURE API KEY LOADING ---
load_dotenv() 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("🚨 ERROR: No API key found! Please check your .env file.")

app = Flask(__name__)

# --- INITIALIZE MODELS ---
# Notice: Whisper and AudioProcessor are completely gone! The app is now much lighter.
print("\n--- Booting up AI Brain (Connecting to Groq Cloud...) ---")
agent = InterviewAgent(session_id="web_cand_001", api_key=GROQ_API_KEY)

topics = ["Python programming", "Machine Learning models", "Data structures and algorithms"]
current_topic_index = 0
current_q = "Welcome to the interview. Could you briefly introduce yourself and your technical background?"

def generate_voice(text):
    """Converts text to speech using Microsoft's Neural engine."""
    voice_path = "temp_voice.mp3"
    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    asyncio.run(communicate.save(voice_path))
    
    with open(voice_path, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode('utf-8')
        
    if os.path.exists(voice_path):
        os.remove(voice_path)
        
    return audio_data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/get_question', methods=['GET'])
def get_question():
    """Sends the first question and its audio."""
    global current_q
    audio_b64 = generate_voice(current_q)
    return jsonify({"question": current_q, "audio_base64": audio_b64})

@app.route('/api/process_answer', methods=['POST'])
def process_answer():
    """Now ONLY accepts text, making the app lightning fast."""
    global current_q, current_topic_index
    
    if not request.is_json:
        return jsonify({"error": "Expected JSON text data"}), 400
        
    data = request.get_json()
    candidate_ans = data.get('text', '').strip()
    
    if not candidate_ans:
        return jsonify({"error": "Empty text provided"}), 400

    print(f"\n[System] Candidate: {candidate_ans}")
    print("[System] Groq is thinking...")
    
    agent.process_candidate_answer(current_q, candidate_ans)
    topic = topics[current_topic_index % len(topics)]
    current_q = agent.generate_next_question(next_topic_goal=topic)
    current_topic_index += 1

    print("[System] Generating Microsoft Voice audio...")
    audio_b64 = generate_voice(current_q)

    return jsonify({
        "transcription": candidate_ans, 
        "next_question": current_q,
        "audio_base64": audio_b64
    })

if __name__ == "__main__":
    print("\n🚀 Server starting... waiting for frontend connections!")
    app.run(debug=True, port=5000)