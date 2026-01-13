from core import get_llm


class BasePipeline:
    def __init__(self, ):
        self.llm = get_llm()
        self.history = {}
    
    def put_history(self, user_id: str, message: str):
        if user_id not in self.history:
            self.history[user_id] = []
        self.history[user_id].append(message)
    
    def get_history(self, user_id: str) -> str:
        if user_id not in self.history:
            return ""
        return "\n".join(self.history[user_id])

    def summarize_history(self, user_id: str) -> str:
        pass

    def run(self, data):
        pass