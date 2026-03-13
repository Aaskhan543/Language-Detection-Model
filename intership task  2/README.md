# 🌍 Multilingual Evaluation & Translation Engine

## 📌 Project Overview

The **Multilingual Evaluation Engine** is an advanced AI pipeline designed to standardize technical interviews across diverse linguistic backgrounds. 

It ingests candidate answers (via text or raw voice audio), translates regional dialects and mixed languages (like Hinglish) into formal English, grades the technical accuracy against a rubric, and translates the AI's feedback back into the candidate's original language. All interactions are securely logged in a local database for fairness auditing.

---

## ✨ Features

* Detects language and translates to formal English
* Dual Input Processing (Text & Audio)
* Offline Audio Transcription using Whisper
* Native support for code-mixed languages (Hinglish, Hindi + English)
* Strict Proper Noun Protection during translation
* Automated LLM-based rubric grading
* Reverse-translation of interview feedback
* Persistent SQLite database logging
* Interactive Web Dashboard built with Streamlit

---

## 🧠 Technologies Used

* Python
* Streamlit (Frontend/UI)
* OpenAI Whisper (Audio Transcription)
* Ollama & Llama 3.2 (Local LLM Engine)
* SQLite3 (Database Management)
* JSON (Data Structuring)

---

## 📂 Project Structure

```text
Multilingual-Evaluation-Engine/
│
├── app.py
├── software_controller.py
├── interview_database.db
├── README.md
└── requirements.txt

⚙️ How the Model Works
1️⃣ Import Libraries
The project starts by importing required libraries:

Streamlit → Web dashboard UI

Whisper → Audio to text

Ollama → Local LLM inference

SQLite3 → Database management

2️⃣ Database Initialization
conn = sqlite3.connect("interview_database.db")
cursor.execute('''CREATE TABLE IF NOT EXISTS candidate_responses (...)''')

Creates a persistent local database table to store original answers, translations, confidence scores, and final grades.

3️⃣ Input Ingestion (Audio or Text)
Python
audio_value = st.audio_input("Record Voice")
# OR
user_text = st.text_area("Candidate Answer:")
The system accepts either direct text input from the candidate or a live microphone recording from the browser.

4️⃣ Audio Transcription (Whisper)
Python
model = whisper.load_model("small")
transcription = model.transcribe("audio.wav", initial_prompt="Hinglish technical answer")
If audio is provided, Whisper converts the spoken words into text. We use an initial_prompt to prevent the AI from hallucinating when it hears code-mixed languages like Hinglish.

5️⃣ Language Detection & Translation
Python
prompt = f"Detect language, fix spelling, and translate to English. Do NOT alter Proper Nouns."
response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt}])
The raw text is sent to Llama 3.2. Strict prompt engineering ensures the AI calculates a confidence_score and protects candidate names from being translated.

6️⃣ LLM Evaluation
Python
prompt = f"Question: {q} \n Rubric: {rubric} \n Candidate Answer: {translated_text}"
The translated English text is graded against the interviewer's specific technical rubric to generate a score out of 10 and a brief feedback summary.

7️⃣ Reverse Feedback Translation
Python
prompt = f"Translate the following interview feedback into {target_language}."
If the candidate spoke in a regional language, the English feedback is translated back into their native tongue so they can easily understand their performance.

8️⃣ Flagging Low Confidence
Python
if confidence < 75:
    flag_for_review = True
The system automatically flags any translation with a confidence score below 75% so a human reviewer can double-check the AI's translation for fairness.

9️⃣ Data Logging
Python
cursor.execute('''INSERT INTO candidate_responses (...) VALUES (...)''')
The entire pipeline output (original text, translation, scores, localized feedback) is committed to the SQLite database.

🔟 Interactive Dashboard
The Streamlit frontend updates dynamically, showing the user the real-time transcription, the AI's technical score, and the localized feedback side-by-side.

▶️ Installation & Setup
Clone repository:

Bash
git clone [https://github.com/YOUR_USERNAME/Multilingual-Evaluation-Engine.git](https://github.com/YOUR_USERNAME/Multilingual-Evaluation-Engine.git)
cd Multilingual-Evaluation-Engine
Install dependencies:

Bash
pip install streamlit openai-whisper ollama
Start the local LLM (Requires Ollama installed on your system):

Bash
ollama run llama3.2
Run project:

Bash
python -m streamlit run app.py
📊 Model Performance
Transcription: OpenAI Whisper (small model) for fast, offline accuracy.

Intelligence Engine: Llama 3.2 via Ollama ensures 100% local, privacy-compliant data processing.

Latency: Optimized for local CPU execution without relying on paid cloud APIs.

📈 Future Improvements
AI Interview Conversation Context Memory Engine (Task 3)

Automated Fairness Consistency PDF Report Generation

Migration to FastAPI for backend endpoints

Cloud deployment using Docker

👨‍💻 Author
Mohd Aas Khan
Machine Learning & AI Enthusiast

⭐ If you like this project, give it a star on GitHub!