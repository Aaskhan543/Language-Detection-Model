import os
import asyncio
import base64
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import edge_tts
from llm_handler import InterviewAgent

# Load Environment Variables (Your Groq API Key)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the Flask App
app = Flask(__name__)

# Initialize our AI Brain
agent = InterviewAgent(session_id="session_001", api_key=GROQ_API_KEY)

# --- MICROSOFT AZURE TTS UTILITY ---
def generate_audio(text):
    """Synchronous wrapper for edge-tts to generate Base64 audio."""
    VOICE = "en-US-ChristopherNeural" # Professional male voice
    
    async def _generate():
        communicate = edge_tts.Communicate(text, VOICE)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return base64.b64encode(audio_data).decode('utf-8')
        
    return asyncio.run(_generate())

# --- ROUTES ---
@app.route('/')
def home():
    """Serves the main Frosted Glass UI."""
    return render_template('index.html')

@app.route('/api/get_question', methods=['POST'])
def get_question():
    """Starts the interview and resets the counter."""
    data = request.json
    selected_domain = data.get('domain', 'General Technology') 
    
    # 1. Reset the interview state for a brand new session!
    agent.question_count = 1 
    agent.memory.history = [] 
    
    first_question = f"Welcome to your technical interview for the {selected_domain} role. Could you briefly introduce yourself and highlight your experience in this field?"
    
    # Store it so the AI remembers what it just asked
    agent.memory.add_interaction(first_question, "[Candidate has not answered yet]")
    
    try:
        audio_base64 = generate_audio(first_question)
        return jsonify({"question": first_question, "audio_base64": audio_base64})
    except Exception as e:
        print(f"TTS Error: {e}")
        return jsonify({"question": first_question, "audio_base64": None})

@app.route('/api/process_answer', methods=['POST'])
def process_answer():
    """Checks the limit: Asks questions 2-5, then generates the Final Report."""
    data = request.json
    candidate_ans = data.get('text')
    selected_domain = data.get('domain', 'General Technology') 

    if not candidate_ans:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Get the AI's last question from memory
        current_q = agent.memory.history[-2]['content'] if len(agent.memory.history) >= 2 else "Please introduce yourself."
        
        # 1. Save candidate's answer
        agent.process_candidate_answer(current_q, candidate_ans)
        
        # --- THE MASTER CONTROLLER LOGIC ---
        if agent.question_count >= 5:
            # INTERVIEW OVER! Generate the report.
            final_report = agent.generate_final_summary(selected_domain)
            
            # Create a polite, short closing voice message
            closing_speech = "Thank you for your time. Your interview is now complete. I have generated a structured summary of your performance on the screen."
            audio_base64 = generate_audio(closing_speech)
            
            return jsonify({
                "next_question": "✅ **INTERVIEW COMPLETE** ✅\n\n" + final_report,
                "audio_base64": audio_base64
            })
        else:
            # INTERVIEW CONTINUES! Increment the counter and ask the next question.
            agent.question_count += 1
            next_question = agent.generate_next_question(domain=selected_domain)
            audio_base64 = generate_audio(next_question)
            
            return jsonify({
                "next_question": next_question,
                "audio_base64": audio_base64
            })
            
    except Exception as e:
        print(f"Error processing answer: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n--- Booting up Enterprise AI Router ---")
    print("🚀 Server starting... waiting for frontend connections!")
    app.run(debug=True)