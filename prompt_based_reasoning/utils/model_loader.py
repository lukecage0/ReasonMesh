import os
import torch
import transformers

def load_model(model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
    """
    Loads a 4-bit quantized LLaMA-based model from Hugging Face 
    using bitsandbytes (bnb) for efficient inference.

    Returns:
        A text-generation pipeline (generator) for the model.
    """
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise ValueError("Please set the HF_TOKEN environment variable, e.g. export HF_TOKEN='...'")

    # Configuration for 4-bit quantization using bitsandbytes
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=False,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 if your GPU supports it
    )

   # print(f"Loading model {model_name} with 4-bit quantization...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",                 # Automatically spread layers across available GPU(s)
        #quantization_config=bnb_config,    # The bitsandbytes 4-bit config
        token=HF_TOKEN,                    # Your Hugging Face token
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN,
    )

    generator = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,  # avoid an error if no pad_token_id
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    print("Model loaded successfully!")
    return generator


def get_response(prompt: str, generator) -> str:
    """
    Generates text from the given prompt using the provided pipeline.

    :param prompt: The input text prompt.
    :param generator: The text-generation pipeline from load_model().
    :return: The generated text (string).
    """
    outputs = generator(prompt)
    # Example structure: [{'generated_text': 'Prompt + ... answer text ...'}]
    return outputs[0]['generated_text']
