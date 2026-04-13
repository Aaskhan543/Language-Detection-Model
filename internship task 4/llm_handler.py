from groq import Groq
from memory_engine import InterviewMemoryEngine

class InterviewAgent:
    def __init__(self, session_id: str, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.1-8b-instant"
        self.memory = InterviewMemoryEngine(session_id=session_id)
        
        # --- NEW: INTERVIEW CONTROLLER STATE ---
        self.question_count = 0 
        self.difficulties = ["Easy", "Medium", "Hard", "Expert"] 

    def process_candidate_answer(self, ai_question: str, candidate_answer: str):
        self.memory.add_interaction(ai_question, candidate_answer)
        if self.memory.needs_summarization():
            self._execute_summarization()

    def _execute_summarization(self):
        print("\n[System] Memory limit reached. Executing Summarization Logic...")
        text_to_compress = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.memory.history])
        
        prompt = f"""Summarize the following interview concisely. 
        Focus ONLY on technical skills and claims.
        Current summary: {self.memory.running_summary}
        New conversation: {text_to_compress}"""

        response = self.client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=200
        )
        self.memory.update_summary(response.choices[0].message.content.strip())
        print("[System] Summarization complete.")

    def generate_next_question(self, domain: str = "General") -> str:
        """Generates the next question and scales the difficulty."""
        # Calculate difficulty based on question number (Q2=Easy, Q3=Medium, etc.)
        difficulty_level = self.difficulties[min(self.question_count - 1, 3)]
        
        system_rules = f"""You are an expert AI technical interviewer. 
        1. The candidate is interviewing for the {domain} role. ALL questions MUST be about {domain}.
        2. This is Question {self.question_count + 1} of 5. The required difficulty level is: {difficulty_level.upper()}.
        3. React to what they just said, then ask the next {difficulty_level} {domain} question.
        4. Keep your spoken response conversational and under 3 sentences."""
        
        messages_to_send = [{"role": "system", "content": system_rules}]
        messages_to_send.extend(self.memory.get_retrieval_context())

        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages_to_send,
            temperature=0.7, max_tokens=150
        )
        return response.choices[0].message.content.strip()

    def generate_final_summary(self, domain: str) -> str:
        """Generates the structured final report for the boss/recruiter."""
        prompt = f"""The {domain} interview is now complete.
        Based on the conversation history, generate a structured, professional candidate evaluation summary.
        You MUST include these exact headings:
        - Overall Score (out of 10)
        - Key Technical Strengths
        - Areas for Improvement
        - Final Hiring Recommendation

        Format it cleanly with bullet points."""

        messages_to_send = [{"role": "system", "content": prompt}]
        messages_to_send.extend(self.memory.get_retrieval_context())

        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages_to_send,
            temperature=0.3, max_tokens=400
        )
        return response.choices[0].message.content.strip()