import ollama
import json

def local_detect_and_translate(candidate_answer):
    # The Prompt-Based Approach from your methodology
    prompt_instructions = f"""
    You are an expert multilingual analysis engine. Analyze the text below. 
    It contains mixed languages (Hinglish, Hindi, and English technical terms).
    
    Return a strict JSON object with exactly these three keys:
    - 'detected_language' (string)
    - 'translated_text' (string, translated purely to formal English)
    - 'confidence_score' (integer between 0 and 100)
    
    Text to analyze: {candidate_answer}
    """

    print("Sending text to local AI... (This might take a few seconds on the first run)")
    
    # We are calling the local model. Change 'llama3.2' to 'qwen2.5' if you downloaded that one instead!
    try:
        response = ollama.chat(
            model='llama3.2', 
            messages=[{'role': 'user', 'content': prompt_instructions}],
            format='json' # This forces the local AI to only spit out clean JSON
        )
        
        result_json = json.loads(response['message']['content'])
        return result_json
    except Exception as e:
        return {"error": f"Something went wrong: {e}"}

# --- Let's test it out! ---
sample_answer = input("Enter your mixed-language candidate answer here: ")

print(f"Original: {sample_answer}\n")
output = local_detect_and_translate(sample_answer)

print("System Output:")
print(json.dumps(output, indent=4))