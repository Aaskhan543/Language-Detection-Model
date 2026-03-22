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
* **Fluent Design UI:** A responsive, modern interface featuring mesh gradients, frosted glass panels (Glassmorphism), and micro-animations (bouncing typing indicators, pulse-recording buttons).

## 🧠 Architecture Details

This project is separated into modular components to ensure scalability and clean code architecture:

1.  **`app.py` (The Routing Brain):** A lightweight Flask server that handles API requests, manages the secure environment variables, and generates the Base64 audio packages.
2.  **`llm_handler.py` (The Integration Module):** Manages the Groq API connection, injects system prompts (behavioral rules and topic nudges), and triggers the memory engine.
3.  **`memory_engine.py` (The Context Store):** A dedicated class that handles short-term conversation history, tracks interaction counts, and executes long-term summarization logic.
4.  **`index.html` (The Client):** Handles the Web Speech API transcription, audio playback, and UI state management. Includes a robust toggle system to prevent audio feedback loops.

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.8+
* A free API key from [Groq Cloud](https://console.groq.com/)
* A modern web browser (Google Chrome or Microsoft Edge recommended for full Web Speech API support).

### 1. Clone the Repository
```bash
git clone [https://github.com/YourUsername/AI-Interviewer-Pro.git](https://github.com/YourUsername/AI-Interviewer-Pro.git)
cd AI-Interviewer-Pro
