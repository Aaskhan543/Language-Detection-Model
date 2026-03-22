# 🎙️ AI Interviewer Pro (Enterprise Edition)

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-lightgrey)
![Groq](https://img.shields.io/badge/Groq-Llama_3.1-orange)
![Azure TTS](https://img.shields.io/badge/Azure_Neural_TTS-Audio-blue)
![Vanilla JS](https://img.shields.io/badge/Vanilla_JS-Frontend-yellow)

A highly responsive, multimodal AI technical interviewer application. Built with a Flask backend and a modern Glassmorphism frontend, this app leverages the Groq Cloud API for near-instantaneous LLM inference and Microsoft's Neural Text-to-Speech engine for professional vocal responses.

## ✨ Key Features

* **Multimodal Interaction:** Candidates can choose to type their responses or use the built-in microphone for real-time, live voice-to-text transcription (powered by the Web Speech API).
* **Enterprise-Grade Voice:** Utilizes `edge-tts` to tap into Microsoft Azure's Neural TTS models, providing a highly professional, human-sounding interviewer voice.
* **Blazing Fast Inference:** Powered by Llama 3.1 (8B) running on Groq's LPU infrastructure, reducing AI response times to milliseconds.
* **Dynamic Memory Engine:** Features a custom-built context memory architecture that tracks conversation history, allowing the AI to ask contextual follow-up questions and probe deeper into candidate answers.
* **Automated Summarization:** Implements a token-saving summarization algorithm that automatically compresses older conversation history into a running summary after 6 interactions, preventing API crashes during long interviews.
* **Fluent Design UI:** A responsive, modern interface featuring mesh gradients, frosted glass panels (Glassmorphism), and micro-animations.

## 🧠 Architecture Details

This project is separated into modular components to ensure scalability and clean code architecture:

1.  **`app.py` (The Routing Brain):** A lightweight Flask server that handles API requests, manages the secure environment variables, and generates the Base64 audio packages.
2.  **`llm_handler.py` (The Integration Module):** Manages the Groq API connection, injects system prompts (behavioral rules and topic nudges), and triggers the memory engine.
3.  **`memory_engine.py` (The Context Store):** A dedicated class that handles short-term conversation history, tracks interaction counts, and executes long-term summarization logic.
4.  **`index.html` (The Client):** Handles the Web Speech API transcription, audio playback, and UI state management. Includes a robust toggle system to prevent audio feedback loops.

## 🛠️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/YourUsername/AI-Interviewer-Pro.git](https://github.com/YourUsername/AI-Interviewer-Pro.git)
   cd AI-Interviewer-Pro

2. Install Dependencies
Bash
pip install flask groq edge-tts python-dotenv

3. Environment Variables
Create a file named .env in the root directory and add your Groq API key:

Code snippet
GROQ_API_KEY=your_secret_api_key_here

4.Run the Server

Bash
python app.py

Gemini said
I completely understand wanting to grab everything in one quick copy-paste! However, squashing a project README into a single, giant paragraph is actually a huge red flag in the corporate engineering world. Engineering managers and recruiters skim repositories in seconds; if they see a massive wall of text instead of clean bullet points and sections, they will likely skip it entirely.

To keep that "Microsoft-level" professional look while making it incredibly easy for you, I have placed the entire, perfectly formatted Markdown text into one single code block below.

All you have to do is click the Copy button at the top right of the box, and you can paste the entire thing straight into GitHub in one go!

Markdown
# 🎙️ AI Interviewer Pro (Enterprise Edition)

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-lightgrey)
![Groq](https://img.shields.io/badge/Groq-Llama_3.1-orange)
![Azure TTS](https://img.shields.io/badge/Azure_Neural_TTS-Audio-blue)
![Vanilla JS](https://img.shields.io/badge/Vanilla_JS-Frontend-yellow)

A highly responsive, multimodal AI technical interviewer application. Built with a Flask backend and a modern Glassmorphism frontend, this app leverages the Groq Cloud API for near-instantaneous LLM inference and Microsoft's Neural Text-to-Speech engine for professional vocal responses.

## ✨ Key Features

* **Multimodal Interaction:** Candidates can choose to type their responses or use the built-in microphone for real-time, live voice-to-text transcription (powered by the Web Speech API).
* **Enterprise-Grade Voice:** Utilizes `edge-tts` to tap into Microsoft Azure's Neural TTS models, providing a highly professional, human-sounding interviewer voice.
* **Blazing Fast Inference:** Powered by Llama 3.1 (8B) running on Groq's LPU infrastructure, reducing AI response times to milliseconds.
* **Dynamic Memory Engine:** Features a custom-built context memory architecture that tracks conversation history, allowing the AI to ask contextual follow-up questions and probe deeper into candidate answers.
* **Automated Summarization:** Implements a token-saving summarization algorithm that automatically compresses older conversation history into a running summary after 6 interactions, preventing API crashes during long interviews.
* **Fluent Design UI:** A responsive, modern interface featuring mesh gradients, frosted glass panels (Glassmorphism), and micro-animations.

## 🧠 Architecture Details

This project is separated into modular components to ensure scalability and clean code architecture:

1.  **`app.py` (The Routing Brain):** A lightweight Flask server that handles API requests, manages the secure environment variables, and generates the Base64 audio packages.
2.  **`llm_handler.py` (The Integration Module):** Manages the Groq API connection, injects system prompts (behavioral rules and topic nudges), and triggers the memory engine.
3.  **`memory_engine.py` (The Context Store):** A dedicated class that handles short-term conversation history, tracks interaction counts, and executes long-term summarization logic.
4.  **`index.html` (The Client):** Handles the Web Speech API transcription, audio playback, and UI state management. Includes a robust toggle system to prevent audio feedback loops.

## 🛠️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/YourUsername/AI-Interviewer-Pro.git](https://github.com/YourUsername/AI-Interviewer-Pro.git)
   cd AI-Interviewer-Pro
Install Dependencies

Bash
pip install flask groq edge-tts python-dotenv
Environment Variables
Create a file named .env in the root directory and add your Groq API key:

Code snippet
GROQ_API_KEY=your_secret_api_key_here
Run the Server

Bash
python app.py
The server will start on http://127.0.0.1:5000.

🎯 Usage Workflow
Launch: Open the application in your browser. The AI will automatically introduce itself and speak the first question out loud.

Interact: Click "🎙️ Start Speaking" to answer. The browser will type your words live on the screen.

Review & Edit: Click the stop button. The UI allows you to review and manually edit your transcribed text before submission to ensure perfect accuracy.

Send: Click Send. The custom memory engine will evaluate your answer, generate a contextual follow-up, and reply using the Microsoft Neural Voice.

📋 Project Requirements Fulfilled
This application was built to successfully demonstrate the following technical requirements:

[x] Design conversation memory architecture.

[x] Implement context storage for interview sessions (memory_engine.py).

[x] Build retrieval system for previous candidate responses.

[x] Integrate contextual memory with LLM prompts (llm_handler.py).

[x] Implement summarization logic for long interview sessions.

Developed by AAS KHAN 
