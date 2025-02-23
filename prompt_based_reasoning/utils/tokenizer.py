from transformers import AutoTokenizer

def load_tokenizer(model_name):
    """Loads tokenizer for a given model."""
    return AutoTokenizer.from_pretrained(model_name)
