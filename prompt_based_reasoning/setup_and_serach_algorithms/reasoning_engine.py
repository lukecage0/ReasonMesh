import logging
from prompt_based_reasoning.prompt_templates.prompt_builder import PromptBuilder
from prompt_based_reasoning.utils.model_loader import load_model, get_response

class ReasoningEngine:
    """
    A reasoning engine that applies different reasoning techniques on top
    of a HF text-generation pipeline, loaded with 4-bit quantization.
    """

    def __init__(self, 
                 model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct", 
                 technique: str = "CoT"):
        """
        Args:
            model_name (str): Name/path of the HF model to load.
            technique (str): Reasoning approach ("CoT", "Zero-Shot-CoT", 
                             "Self-Consistency", "ReAct", etc.).
        """
        self.technique = technique
        # Load model (pipeline) from model_loader
        self.generator = load_model(model_name)
        # Initialize PromptBuilder
        self.prompt_builder = PromptBuilder()

    def apply_reasoning_technique(self, question: str):
        """
        Construct a prompt or multiple prompts based on the selected reasoning technique.
        Returns a single string or a list of strings (for e.g., Self-Consistency).
        """

        # For "CoT" - we might choose 2-shot
        if self.technique == "CoT":
            return self.prompt_builder.build_prompt(
                question=question,
                technique="CoT",
                shuffle_prompt=False,
                num_shot=2
            )

        # For "Zero-Shot-CoT" - we do 0-shot
        elif self.technique == "Zero-Shot-CoT":
            return self.prompt_builder.build_prompt(
                question=question,
                technique="Zero-Shot-CoT",
                shuffle_prompt=False,
                num_shot=0
            )

        # For "Self-Consistency" - we can generate multiple prompts
        elif self.technique == "Self-Consistency":
            prompts = []
            for _ in range(5):
                sc_prompt = self.prompt_builder.build_prompt(
                    question=question,
                    technique="Self-Consistency",
                    shuffle_prompt=True,
                    num_shot=0  # or 2 if you want few-shot self-consistency
                )
                prompts.append(sc_prompt)
            return prompts

        # For "ReAct"
        elif self.technique == "ReAct":
            return self.prompt_builder.build_prompt(
                question=question,
                technique="ReAct",
                shuffle_prompt=False,
                num_shot=0
            )

        else:
            # Default fallback or custom technique
            return question

    def run(self, question: str) -> str:
        """
        Build the prompt(s), generate response(s).
        If Self-Consistency, gather multiple responses and do majority vote.
        Otherwise, return the single response.
        """
        prompt = self.apply_reasoning_technique(question)
        
        if self.technique == "Self-Consistency" and isinstance(prompt, list):
            # Generate multiple responses
            responses = [get_response(p, self.generator) for p in prompt]
            # Return the most common among them (majority vote)
            return max(set(responses), key=responses.count)
        else:
            # Single prompt
            return get_response(prompt, self.generator)
if __name__ == "__main__":
    engine = ReasoningEngine(technique="Zero-Shot-CoT")
    question = "Janet’s ducks lay 16 eggs per day. She eats 3 for breakfast and uses 4 for baking. She sells the remainder for $2 each. How much does she make daily?"
    answer = engine.run(question)
    print("Zero-Shot-CoT answer:", answer)

#     if __name__ == "__main__":
#     engine = ReasoningEngine(technique="CoT")
#     question = "Josh buys a house for $80,000 and invests $50,000 in repairs, increasing its value by 150%. How much profit did he make?"
#     answer = engine.run(question)
#     print("Few-Shot-CoT answer:", answer)
# if __name__ == "__main__":
#     engine = ReasoningEngine(technique="Self-Consistency")
#     question = "A robe takes 2 bolts of blue fiber and half that much white fiber. How many total bolts does it use?"
#     answer = engine.run(question)
#     print("Self-Consistency answer:", answer)



