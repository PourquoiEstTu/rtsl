import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    EarlyStoppingCallback,
)


from .configs import Gloss_to_Sentence_Config
from .transformer import create_model

class Gloss_to_Sentence_Model:
    
    def __init__(self, ):
        config = Gloss_to_Sentence_Config()
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

# def main():
#     sentence_model = Gloss_to_Sentence_Model()
    
#     sample_glosses = [
#         "HELLO MY NAME J-O-H-N",
#         "NICE MEET YOU",
#         "HOW YOU",
#         "I FINE",
#         "YOUR NAME WHAT",
#         "WHERE YOU LIVE",
#         "I HUNGRY",
#         "I THIRSTY WANT WATER",
#         "TOMORROW STORE I GO",
#         "YESTERDAY SCHOOL I GO",
#         "LAST-WEEK MOVIE I WATCH",
#         "PAST NIGHT I SLEEP EARLY",
#         "COFFEE I DRINK WANT",
#         "MUSIC I LISTEN ENJOY",
#         "IF RAIN TOMORROW GAME CANCEL",
#         "TEACHER SAY TEST DIFFICULT BUT I STUDY HARD",
#         "AFTER EAT DINNER I WATCH TV",
#         "YESTERDAY WORK I GO WHY SICK I",
#         "FINISH HOMEWORK YOU WANT GO MOVIE?",
#         "MOTHER NOT-YET FEED CAT."
#     ]

#     for gloss in sample_glosses:
#         prediction = sentence_model.run_inference(gloss)
#         print(f"\nGloss: {gloss}")
#         print(f"Prediction:-----------------------\n{prediction}\n-----------------------")


# if __name__ == "__main__":
#     main()