# AI Engineering Internship Projects

This repository contains a collection of Artificial Intelligence and Natural Language Processing (NLP) projects developed during my internship. The primary focus is on building scalable, multilingual AI systems using state-of-the-art open-source models.

## Repository Structure

The project is divided into modular tasks. Each folder contains the specific code, dependencies, and documentation for that module.

### 📁 Task 1: Language Detection Model
# Language-Detection-Model
A Machine Learning and NLP project that identifies the language of a given text using natural language processing, feature engineering, and classification algorithms.
# 🌍 Language Detection Model

## 📌 Project Overview

The **Language Detection Model** is a Machine Learning and Natural Language Processing (NLP) project that automatically predicts the language of a given text input.

The model is trained using a multilingual dataset and applies text vectorization and the Naive Bayes classification algorithm to identify languages accurately.

---

## 🚀 Features

* Detects language from user input text
* Uses NLP text vectorization
* Machine Learning based classification
* Interactive user prediction system
* Implemented using Python & Scikit-learn
* Jupyter Notebook implementation included

---

## 🧠 Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Natural Language Processing (NLP)
* Multinomial Naive Bayes

---

## 📂 Project Structure

```
Language-Detection-Model/
│
├── language_detection.ipynb
├── language_detection.py
├── language.csv
├── README.md
└── requirements.txt
```

---

## ⚙️ How the Model Works

### 1️⃣ Import Libraries

The project starts by importing required libraries:

* NumPy → numerical operations
* Pandas → dataset handling
* Scikit-learn → machine learning tools

---

### 2️⃣ Load Dataset

```
data = pd.read_csv("language.csv")
```
### 📁 Task 2: Multilingual Voice-to-Text Translator
# Task 2: Multilingual Voice-to-Text Translator

## Overview
This project is a fully offline, AI-powered voice translation module. It is designed to capture live audio from a microphone, transcribe spoken words, detect the primary language (including complex code-mixing like Hinglish), and translate the meaning into formal English. 

A key achievement of this module is its strict adherence to data privacy. By utilizing local models, all audio processing and translation occur directly on the host machine without relying on external or paid cloud APIs.

## System Architecture
The system operates in two main phases:
1. **The Ears (Speech-to-Text):** Utilizes OpenAI's open-source `Whisper` model to capture live microphone input and generate raw text transcripts. It handles multiple regional and global languages automatically.
2. **The Brain (Analysis & Translation):** Utilizes Meta's `Llama 3.2` model via `Ollama`. It receives the raw text, corrects minor transcription errors, identifies the specific language spoken, and outputs a structured JSON report with the formal English translation.

## Prerequisites
To run this model on a local Windows machine, the following software must be installed:
* **Python 3.8+**
* **FFmpeg:** Required for audio processing (`winget install ffmpeg`).
* **Ollama:** Required to run the local language model. 

### Python Libraries
Install the required libraries using pip:
```bash
pip install openai-whisper sounddevice scipy ollama

### 📁 Task 3: [Upcoming Task]
* *Details will be added as the internship progresses.*

---

## 🚀 Setup and Installation

To run any of the modules in this repository locally, you will need Python installed on your system. 

**General Prerequisites:**
1. Clone this repository to your local machine.
2. Navigate to the specific task folder.
3. Install the required Python libraries (listed in each task's folder).
4. For Task 2, ensure you have **FFmpeg** and **Ollama** installed on your system.

## 🧠 Tech Stack Overview
* **Language:** Python
* **AI/ML Models:** Whisper, Llama 3.2 (via Ollama)
* **Libraries:** `whisper`, `ollama`, `sounddevice`, `scipy`

---
*Developed as part of an AI Technical Internship.*
