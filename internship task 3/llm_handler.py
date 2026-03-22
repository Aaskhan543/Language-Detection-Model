from groq import Groq
from memory_engine import InterviewMemoryEngine

class InterviewAgent:
    def __init__(self, session_id: str, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.1-8b-instant"
        
        # Initialize the dedicated memory component
        self.memory = InterviewMemoryEngine(session_id=session_id)
        
        self.base_system_prompt = """You are an expert AI technical interviewer. 
        1. React to what the candidate just said. Do not just read off a list.
        2. If the answer is short/lazy, probe deeper. Ask "Why?"
        3. Keep your responses under 3 sentences for natural voice playback."""

    def process_candidate_answer(self, ai_question: str, candidate_answer: str):
        """Saves memory and triggers the required Summarization Logic if full."""
        self.memory.add_interaction(ai_question, candidate_answer)
        
        # --- SUMMARIZATION LOGIC ---
        if self.memory.needs_summarization():
            self._execute_summarization()

    def _execute_summarization(self):
        """Asks Groq to compress the old conversation history into a tight summary."""
        print("\n[System] Memory limit reached. Executing Summarization Logic...")
        
        # Grab the text we want to compress
        text_to_compress = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.memory.history])
        
        prompt = f"""Summarize the following interview conversation concisely. 
        Focus ONLY on the candidate's technical skills, background, and specific claims.
        Current ongoing summary: {self.memory.running_summary}
        
        New conversation to add to summary:
        {text_to_compress}"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        new_summary = response.choices[0].message.content.strip()
        self.memory.update_summary(new_summary)
        print("[System] Summarization complete.")

    def generate_next_question(self, next_topic_goal: str) -> str:
        """Prompt Context Integration Module."""
        
        # 1. Start with base personality
        messages_to_send = [{"role": "system", "content": self.base_system_prompt}]
        
        # 2. Retrieve and integrate memory context
        messages_to_send.extend(self.memory.get_retrieval_context())
        
        # 3. Add the hidden topic nudge
        nudge = f"If the candidate's last answer was good, transition to asking about: {next_topic_goal}."
        messages_to_send.append({"role": "system", "content": nudge})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages_to_send,
            temperature=0.7,
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()