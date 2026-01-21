from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from loguru import logger
from .config import MODEL_NAME, DEVICE, MAX_TOKENS, TEMPERATURE, TOP_P


#AIraModel is the core interface to load and interact with the language model.

class AIraModel:

    def __init__(self):
        logger.info(f"Loading model '{MODEL_NAME}' on {DEVICE}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto" if DEVICE=="cuda" else None
        )
        self.model.eval()  # inference mode
        logger.info("Model loaded successfully!")

    def generate_text(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)  # give a dictionary of tensors
        # print(inputs)
        outputs = self.model.generate(
            **inputs,  # *args: Unpacks a list (positional arguments), **kwargs: Unpacks a dictionary (keyword arguments).
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
