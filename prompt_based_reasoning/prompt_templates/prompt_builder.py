import random

class PromptBuilder:
    """
    A PromptBuilder that stores prompt configurations (examples + prefix)
    for different reasoning techniques. You can select zero-shot, few-shot CoT,
    ReAct, Self-Consistency, etc., and specify how many examples to include.
    """

    def __init__(self):
        """
        self.init_prompts: Dictionary mapping each technique to a config dict:
          {
            "cot_pool": [...],
            "prefix": "...",
          }
        """
        self.init_prompts = {
            "CoT": {
                # A few CoT examples for demonstration if you want few-shot
                "cot_pool": [
                    (
                        "Q: If you have 3 apples and you eat 1, how many remain?\n"
                        "A: Let's think step by step.\n"
                        "1) Start with 3 apples.\n"
                        "2) Eat 1.\n"
                        "Thus, 2 remain.\n"
                        "#### 2\n\n"
                    ),
                    (
                        "Q: What is 2 + 2?\n"
                        "A: Let's think step by step.\n"
                        "1) 2+2=4.\n"
                        "#### 4\n\n"
                    ),
                ],
                "prefix": (
                    "Use the same style: reason step by step, then write the final numeric "
                    "answer alone after '####'.\n\n"
                ),
            },
            "Zero-Shot-CoT": {
                "cot_pool": [],  # no examples for zero-shot
                "prefix": (
                    "We'll solve this in a step-by-step manner and give the final numeric "
                    "answer alone after '####'.\n\n"
                ),
            },
            "Self-Consistency": {
                "cot_pool": [],  # typically you might reuse the same approach as CoT or no examples
                "prefix": (
                    "We'll think step by step in multiple ways. Then provide the final numeric "
                    "answer alone after '####'.\n\n"
                ),
            },
            "ReAct": {
                "cot_pool": [],
                "prefix": (
                    "Let's reason using ReAct, step by step, taking actions if needed, and "
                    "finally provide the numeric answer after '####'.\n\n"
                ),
            },
        }

    def build_prompt(
        self,
        question: str,
        technique: str = "CoT",
        shuffle_prompt: bool = False,
        num_shot: int = 0,
    ) -> str:
        """
        Build a prompt based on the chosen technique and how many few-shot
        examples we want to include.

        :param question: The user question or problem statement.
        :param technique: Which technique config to use ("CoT", "Zero-Shot-CoT", "Self-Consistency", etc.).
        :param shuffle_prompt: Whether to shuffle the few-shot examples.
        :param num_shot: Number of examples to include from the technique's `cot_pool`.
        :return: A fully composed prompt string.
        """
        # 1) Retrieve the relevant config
        technique_config = self.init_prompts.get(technique, self.init_prompts["CoT"])
        cot_pool = technique_config.get("cot_pool", [])
        prefix = technique_config.get("prefix", "")

        # 2) If we have a pool and the user wants a few-shot approach
        #    pick examples from the pool
        if cot_pool and num_shot > 0:
            if shuffle_prompt:
                examples = random.sample(cot_pool, min(num_shot, len(cot_pool)))
            else:
                examples = cot_pool[:num_shot]

            examples_str = "".join(examples)
        else:
            # zero-shot or empty pool
            examples_str = ""

        # 3) Combine the examples + prefix + final question
        prompt = (
            f"{examples_str}"
            f"{prefix}" 
            f"Q: {question}\n"
            "A: Let's think step by step.\n"
            "At the end, write your final numeric answer alone on a new line after '####'.\n"
        )
        return prompt
