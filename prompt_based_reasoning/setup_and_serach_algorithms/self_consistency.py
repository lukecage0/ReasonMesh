import random
from reasoning_engine import ReasoningEngine

class SelfConsistency:
    def __init__(self, model_name="mistralai/Mistral-7B", samples=5):
        self.engine = ReasoningEngine(model_name=model_name, technique="CoT")
        self.samples = samples

    def run(self, question):
        responses = [self.engine.run(question) for _ in range(self.samples)]
        return max(set(responses), key=responses.count)  # Majority voting

if __name__ == "__main__":
    reasoner = SelfConsistency()
    question = "If Alice has 3 apples and gives 2 to Bob, how many does she have left?"
    print("Final Answer:", reasoner.run(question))
