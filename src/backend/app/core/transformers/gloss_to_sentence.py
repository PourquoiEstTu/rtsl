import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    EarlyStoppingCallback,
)


from .utils.config import Config
from .models.transformer2 import create_model

class Gloss_to_Sentence_Model:
    
    def __init__(self, ):
        config = Config()
        self.model = create_model(config.model.model_name)        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

    def run_inference(self, glosses):
        self.model.eval()
        self.model.to(self.device)
        
        inputs = self.tokenizer(
            glosses,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
        
        prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return prediction
