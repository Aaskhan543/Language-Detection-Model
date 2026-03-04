from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
import json

# This ensures our language detection gives consistent results every time
DetectorFactory.seed = 0

def detect_and_translate_opensource(candidate_answer):
    try:
        # 1. Detect the language
        # It will return a language code like 'hi' for Hindi, 'en' for English, 'ta' for Tamil
        detected_lang_code = detect(candidate_answer)
        
        # 2. Translate the text to English
        # We set source='auto' so it handles mixed languages (like Hinglish) surprisingly well
        translator = GoogleTranslator(source='auto', target='en')
        translated_text = translator.translate(candidate_answer)
        
        # 3. Format our output to match our Database Blueprint
        result = {
            "detected_language_code": detected_lang_code,
            "translated_text": translated_text,
            "translation_engine": "Open-Source Traditional NLP",
            "status": "Success"
        }
        return result

    except Exception as e:
        return {"error": f"Something went wrong: {e}"}

# --- Let's test it with a mixed-language administrative example ---
sample_answer = "Constitution ka Preamble basically identity card hai, aur secularism basic structure ka part hai according to Kesavananda Bharati case."

print("Analyzing candidate answer (No GenAI)...")
print(f"Original: {sample_answer}\n")

# Run our function
output = detect_and_translate_opensource(sample_answer)

# Print the result nicely formatted
print("System Output:")
print(json.dumps(output, indent=4))

