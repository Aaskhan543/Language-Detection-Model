import streamlit as st
import whisper
import ollama
import json
import os
import sqlite3
import warnings

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# ==========================================
# 1. Database Setup
# ==========================================
def init_db():
    conn = sqlite3.connect("interview_database.db")
    cursor = conn.cursor()
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
    conn.commit()
    conn.close()

def save_to_database(record):
    conn = sqlite3.connect("interview_database.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO candidate_responses (
            candidate_id, original_answer, detected_language, confidence_score,
            translated_english, score, feedback_english, feedback_localized, requires_review
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        record.get("candidate_id", "candidate_101"), 
        record.get("original_answer", ""), 
        record.get("detected_language", "Unknown"),
        record.get("translation_confidence", 0), 
        record.get("translated_english_answer", ""),
        record.get("score", 0), 
        record.get("feedback_english", ""), 
        record.get("feedback_localized", ""),
        record.get("flag_for_review", False)
    ))
    conn.commit()
    conn.close()

# Initialize the database as soon as the app loads
init_db()

# ==========================================
# 2. AI Models & Logic
# ==========================================
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("small")

def detect_and_translate(raw_text):
    prompt = f"""
    You are an expert linguistic AI. 
    Fix spelling errors, detect the exact primary language (or if it is code-mixed like Hinglish), and translate it to formal English.
    CRITICAL RULE 1: Do NOT translate or alter Proper Nouns.
    CRITICAL RULE 2: 'confidence_score' MUST be a whole number between 0 and 100 (e.g., 90, not 0.90).
    
    Return a strict JSON object with: 'detected_language', 'translated_text', and 'confidence_score'.
    Text: {raw_text}
    """
    try:
        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}], format='json')
        return json.loads(response['message']['content'])
    except Exception as e:
        return {"error": str(e)}

def evaluate_answer(translated_text, question_text, grading_rubric):
    prompt = f"""
    You are an unbiased technical interviewer. 
    Question: {question_text}
    Rubric: {grading_rubric}
    Candidate Answer: {translated_text}
    
    Evaluate the answer strictly based on the rubric.
    Return a strict JSON object with: 'score' (integer out of 10), 'english_feedback' (string), and 'is_conceptually_correct' (boolean).
    """
    try:
        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}], format='json')
        return json.loads(response['message']['content'])
    except Exception as e:
        return {"error": str(e)}

def translate_feedback_to_original(english_feedback, target_language):
    if "english" in target_language.lower() and "hinglish" not in target_language.lower():
        return english_feedback
        
    prompt = f"""
    Translate the following interview feedback into {target_language}. 
    Keep the tone professional and encouraging.
    Feedback: "{english_feedback}"
    Return a strict JSON object with ONE key: 'localized_feedback' (string).
    """
    try:
        response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}], format='json')
        result = json.loads(response['message']['content'])
        return result.get('localized_feedback', english_feedback)
    except Exception as e:
        return english_feedback

# ==========================================
# 3. The Web Dashboard (UI)
# ==========================================
st.set_page_config(page_title="AI Interview Engine", page_icon="🚀", layout="wide")

st.title("🚀 Multilingual AI Interview Engine")
st.markdown("### Powered by Llama 3.2 & Whisper")

# Interview Context Settings
st.sidebar.header("📋 Interview Settings")
question = st.sidebar.text_area("Interview Question", "What is the role of an operating system?")
rubric = st.sidebar.text_area("Grading Rubric", "Must mention managing hardware resources and providing a user interface.")

tab1, tab2 = st.tabs(["✍️ Type Answer", "🎤 Record Audio"])

def process_pipeline(raw_text):
    with st.spinner("🧠 1/3: Detecting Language & Translating..."):
        translation_data = detect_and_translate(raw_text)
        detected_lang = translation_data.get("detected_language", "Unknown")
        english_text = translation_data.get("translated_text", "")
        confidence = translation_data.get("confidence_score", 0)
        
    with st.spinner("⚖️ 2/3: AI is Evaluating the Answer..."):
        evaluation_data = evaluate_answer(english_text, question, rubric)
        
    with st.spinner(f"🔄 3/3: Translating Feedback to {detected_lang}..."):
        localized_feedback = translate_feedback_to_original(
            english_feedback=evaluation_data.get("english_feedback", ""),
            target_language=detected_lang
        )
        
    # Build final record
    final_record = {
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
    
    # Save to Database
    save_to_database(final_record)
    
    # Display Results beautifully on the webpage
    st.success("✅ Evaluation Complete and Saved to Database!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Candidate Input")
        st.info(f"**Original:** {raw_text}\n\n**Language:** {detected_lang} (Confidence: {confidence}%)")
        st.warning(f"**English Translation:** {english_text}")
        
    with col2:
        st.subheader("AI Evaluation")
        st.metric("Score", f"{final_record['score']}/10")
        st.success(f"**English Feedback:** {final_record['feedback_english']}")
        st.info(f"**Localized Feedback:** {final_record['feedback_localized']}")
        
    if final_record["flag_for_review"]:
        st.error("⚠️ This answer was flagged for manual review due to low translation confidence.")

# --- Tab 1: Text Input ---
with tab1:
    user_text = st.text_area("Candidate Answer (Hinglish/Regional):")
    if st.button("Submit Text Answer"):
        if user_text:
            process_pipeline(user_text)
        else:
            st.warning("Please enter an answer first.")

# --- Tab 2: Audio Input ---
with tab2:
    st.write("Record candidate answer directly:")
    audio_value = st.audio_input("Record Voice")
    
    if audio_value is not None:
        with st.spinner("🎧 Transcribing Audio with Whisper..."):
            with open("temp_browser_audio.wav", "wb") as f:
                f.write(audio_value.getbuffer())
                
            model = load_whisper_model()
            system_hint = "The following is a technical interview answer in Hinglish, Hindi, or English."
            transcription = model.transcribe("temp_browser_audio.wav", initial_prompt=system_hint)
            st.write(f"**Raw Transcription:** {transcription['text']}")
            
        process_pipeline(transcription["text"])