import logging
import re
from datasets import load_dataset

# Adjust to your actual import path
from prompt_based_reasoning.setup_and_serach_algorithms.reasoning_engine import ReasoningEngine

def extract_final_answer(text: str, eos=None):
    """
    Attempt to extract a final integer answer from a string.

    1) If 'eos' is provided, first split at that token.
    2) Next, look for the substring after '####' (if present).
    3) Remove certain characters like ',', '$', '%' (common in money or numeric formats).
    4) Try to parse the result as an integer.
    5) If that fails, fallback to a regex-based search for digits in the entire string.

    Returns:
        (int or None) - The integer answer if found, otherwise None.
    """
    original_text = text  # Keep for fallback reference

    # 1) Optional: if there's a custom end-of-string marker
    if eos and eos in text:
        text = text.split(eos)[0].strip()

    # 2) If '####' is present, take the substring after it
    if '####' in text:
        text = text.split('####')[-1].strip()

    # 3) Remove common numeric formatting chars
    for remove_char in [',', '$', '%', 'g']:
        text = text.replace(remove_char, '')

    # 4) Try parsing as int directly
    try:
        return int(text.strip())
    except ValueError:
        pass  # We'll fallback to regex-based approach

    # 5) Regex fallback: find all digit sequences in the original text, 
    #    return the last one as an integer if it exists
    match = re.findall(r'(\d+)', original_text.replace(',', '').replace('$', ''))
    if match:
        return int(match[-1])
    
    # If we reach here, we couldn't parse an integer
    return None


class GSM8KBenchmark:
    """
    A simple benchmark class for evaluating a reasoning engine on GSM8K questions.
    """
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", technique="CoT"):
        """
        Args:
            model_name (str): HF model name/path to load.
            technique (str): Reasoning approach, e.g. "CoT", "Self-Consistency", etc.
        """
        self.engine = ReasoningEngine(model_name=model_name, technique=technique)
        logging.info(f"GSM8KBenchmark initialized with model={model_name}, technique={technique}")

    def evaluate(self, num_samples=5):
        """
        Evaluate on a subset of the GSM8K dataset from Hugging Face (test split).
        """
        logging.info("Loading GSM8K (socratic) from Hugging Face...")
        dataset = load_dataset("openai/gsm8k", "socratic")
        gsm8k_test = dataset["test"].select(range(num_samples))
        logging.info(f"Loaded test split with {len(gsm8k_test)} samples.")

        correct = 0
        total = len(gsm8k_test)

        for i in range(total):
            sample = gsm8k_test[i]
            question = sample["question"]
            expected_raw = sample["answer"]  # e.g. "#### 18" or step-by-step with "#### 18"

            # Run model to get prediction
            prediction = self.engine.run(question)

            # Extract numeric answers
            gold_val = extract_final_answer(expected_raw)
            pred_val = extract_final_answer(prediction)

            matched = (gold_val is not None and pred_val == gold_val)
            if matched:
                correct += 1

            # Logging
            logging.info(f"\n--- Sample {i+1}/{total} ---")
            logging.info(f"Question: {question}")
            logging.info(f"Expected: {expected_raw}  -> extracted numeric: {gold_val}")
            logging.info(f"Prediction: {prediction} -> extracted numeric: {pred_val}")
            logging.info(f"Matched? {matched}")

        accuracy = correct / total if total > 0 else 0
        logging.info(f"\nFinished evaluation on {total} samples. Accuracy = {accuracy:.2%}")
        print(f"\n🔥 GSM8K Benchmark Accuracy: {accuracy * 100:.2f}% (on {total} samples)")


if __name__ == "__main__":
    # Optional: configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Instantiate benchmark with your desired model & technique
    benchmark = GSM8KBenchmark(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        technique="Zero-Shot-CoT"
    )
    # Evaluate on a small sample
    benchmark.evaluate(num_samples=100)
