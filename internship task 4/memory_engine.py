class InterviewMemoryEngine:
    def __init__(self, session_id: str):
        self.session_id = session_id
        # The Context Memory Store
        self.history = []
        self.running_summary = ""
        self.turn_count = 0
        
        # Summarize after every 6 interactions to prevent API crashes
        self.summarization_threshold = 6 

    def add_interaction(self, ai_question: str, candidate_answer: str):
        """Stores the recent back-and-forth."""
        self.history.append({"role": "assistant", "content": ai_question})
        self.history.append({"role": "user", "content": candidate_answer})
        self.turn_count += 1

    def get_retrieval_context(self) -> list:
        """The Retrieval Engine: Combines the long-term summary with short-term history."""
        context = []
        
        # If we have a summary of older chat, inject it first
        if self.running_summary:
            context.append({
                "role": "system", 
                "content": f"[PRIOR CONVERSATION SUMMARY]: {self.running_summary}"
            })
            
        # Add the exact recent history
        context.extend(self.history)
        return context

    def needs_summarization(self) -> bool:
        """Checks if the memory bank is getting too full."""
        return self.turn_count >= self.summarization_threshold

    def update_summary(self, new_summary: str):
        """Saves the new summary and clears out the old history to save space."""
        self.running_summary = new_summary
        
        # Keep only the very last 2 messages for immediate context, delete the rest
        self.history = self.history[-2:]
        self.turn_count = 1