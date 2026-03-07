# llm_factory/get_llm.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from config.settings import Settings

settings = Settings()

# Cache
_current_model_name = None
_current_llm_instance = None

class HuggingFaceLLM:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )

    def generate(self, prompt: str, max_tokens: int = 200):
        inputs = self.tokenizer(prompt, return_tensors="pt")

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size = 3,
            eos_token_id = self.tokenizer.eos_token_id
        )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # remove prompt
        answer = generated_text[len(prompt):]
        # Remove unwanted conversation tags
        answer = answer.replace("User:", "")
        answer = answer.replace("Assistant:", "")
        # Stop generation if model starts next turn
        if "User:" in answer:
            answer = answer.split("User:")[0]

        return answer.strip()

def get_hf_llm(selected_model: str = None):
    """
    Return cached HuggingFace LLM instance
    """
    global _current_model_name, _current_llm_instance

    if selected_model is None:
        selected_model = settings.DEFAULT_MODEL

    model_name = settings.HF_MODEL_NAME[selected_model]

    if _current_model_name == model_name and _current_llm_instance is not None:
        return _current_llm_instance

    llm = HuggingFaceLLM(model_name)
    _current_model_name = model_name
    _current_llm_instance = llm
    return llm
# Example Usage
# Get a cached instance of the LLM (TinyLlama)
check_llm = get_hf_llm(selected_model="TinyLlama")

# Print the object and its type
print(check_llm)
print(type(check_llm))
