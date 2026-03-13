import whisper
import warnings
import ollama
import json
import os
import sqlite3

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# ==========================================
# 1. The Audio Path
# ==========================================
def handle_audio_input(audio_file_path):
    print(f"🎧 AUDIO DETECTED: Loading Whisper model to transcribe...")
    try:
        model = whisper.load_model("small") 
        result = model.transcribe(audio_file_path)
        return result["text"]
    except Exception as e:
        print(f"❌ Audio error: {e}")
        return None

# ==========================================
# 2. The Text Path
# ==========================================
def handle_text_input(text_string):
    print(f"✍️ TEXT DETECTED: Bypassing Whisper. Sending text directly to the Brain...")
    return text_string

# ==========================================
# 3. The Translator Brain
# ==========================================
def detect_and_translate(raw_text):
    print("🧠 Translating to English and detecting language...")
    prompt = f"""
    You are an expert linguistic AI. 
    Fix spelling errors, detect the exact primary language (or if it is code-mixed like Hinglish), and translate it to formal English.
    CRITICAL RULE: Do NOT translate or alter Proper Nouns (names of people/places).
    
    Return a strict JSON object with: 'detected_language', 'translated_text', and 'confidence_score'.
    Text: {raw_text}
    """
    try:
        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}], format='json')
        return json.loads(response['message']['content'])
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 4. The Evaluator (Grading the English Text)
# ==========================================
def evaluate_answer(translated_text, question_text, grading_rubric):
    print("⚖️ Evaluating the candidate's answer...")
    prompt = f"""
    You are an unbiased technical interviewer. 
    Question: {question_text}
    Rubric: {grading_rubric}
    Candidate Answer: {translated_text}
    
    Evaluate the answer strictly based on the rubric.
    Return a strict JSON object with:
    - 'score' (integer out of 10)
    - 'english_feedback' (string, max 2 sentences)
    - 'is_conceptually_correct' (boolean)
    """
    try:
        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}], format='json')
        return json.loads(response['message']['content'])
    except Exception as e:
        return {"error": str(e)}

# ==========================================
# 5. The Reverse Translator (Feedback to Original Language)
# ==========================================
def translate_feedback_to_original(english_feedback, target_language):
    # If the user spoke English, we don't need to translate the feedback!
    if "english" in target_language.lower() and "hinglish" not in target_language.lower():
        return english_feedback
        
    print(f"🔄 Translating feedback back to {target_language}...")
    prompt = f"""
    Translate the following interview feedback into {target_language}. 
    Keep the tone professional and encouraging.
    
    Feedback to translate: "{english_feedback}"
    
    Return a strict JSON object with ONE key:
    - 'localized_feedback' (string)
    """
    try:
        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}], format='json')
        result = json.loads(response['message']['content'])
        return result.get('localized_feedback', english_feedback)
    except Exception as e:
        return english_feedback

# ==========================================
# 6. The Main System Router (Database Prep)
# ==========================================
def process_candidate_answer(user_input, question_text, rubric):
    print("\n" + "="*50)
    
    input_lower = user_input.lower()
    if input_lower.endswith((".wav", ".mp3", ".m4a")):
        raw_text = handle_audio_input(user_input)
    else:
        raw_text = handle_text_input(user_input)
        
    translation_data = detect_and_translate(raw_text)
    detected_lang = translation_data.get("detected_language", "Unknown")
    english_text = translation_data.get("translated_text", "")
    confidence = translation_data.get("confidence_score", 0)
    
    # Low Confidence Warning
    if confidence < 75:
        print(f"⚠️ SYSTEM WARNING: Low translation confidence ({confidence}%). Flagging for manual review.")
        
    evaluation_data = evaluate_answer(english_text, question_text, rubric)
    
    localized_feedback = translate_feedback_to_original(
        english_feedback=evaluation_data.get("english_feedback", ""),
        target_language=detected_lang
    )
    
    # The Final Output
    final_database_record = {
        "candidate_id": "candidate_101",
        "original_answer": raw_text,
        "detected_language": detected_lang,
        "translation_confidence": confidence,
        "translated_english_answer": english_text,
        "score": evaluation_data.get("score"),
        "feedback_english": evaluation_data.get("english_feedback"),
        "feedback_localized": localized_feedback,
        "flag_for_review": confidence < 75
    }
    
    print("\n=== FINAL PIPELINE OUTPUT (READY FOR DATABASE) ===")
    print(json.dumps(final_database_record, indent=4))
    print("="*50)
    # Save the output to the database
    save_to_database(final_database_record)

# ==========================================
# 7. The Database Manager
# ==========================================
def save_to_database(record):
    print("💾 Connecting to local database...")
    
    # Connect to SQLite (This creates the file if it doesn't exist)
    conn = sqlite3.connect("interview_database.db")
    cursor = conn.cursor()
    
    # Create the table using your rubric's exact requirements
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS candidate_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id TEXT,
            original_answer TEXT,
            detected_language TEXT,
            confidence_score INTEGER,
            translated_english TEXT,
            score INTEGER,
            feedback_english TEXT,
            feedback_localized TEXT,
            requires_review BOOLEAN
        )
    ''')
    
    # Insert the AI's final record into the database
    cursor.execute('''
        INSERT INTO candidate_responses (
            candidate_id, original_answer, detected_language, confidence_score,
            translated_english, score, feedback_english, feedback_localized, requires_review
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        record["candidate_id"], record["original_answer"], record["detected_language"],
        record.get("translation_confidence", 0), record["translated_english_answer"],
        record.get("score", 0), record["feedback_english"], record["feedback_localized"],
        record["flag_for_review"]
    ))
    
    # Save and close
    conn.commit()
    conn.close()
    print("✅ SUCCESS: Candidate record permanently saved to 'interview_database.db'!")
# ==========================================
# Run the Test
# ==========================================
question = "What is the role of an operating system?"
rubric = "Must mention managing hardware resources and providing a user interface."

process_candidate_answer("Operating system computer ka hardware manage karta hai.", question, rubric)