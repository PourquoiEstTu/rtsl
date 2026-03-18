"""
Inference script for ASL Gloss-to-English translation
Uses pre-trained Glossa-BART directly from Hugging Face
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class ASLTranslator:
    def __init__(
        self, 
        model_name: str = "rrrr66254/Glossa-BART",
        device: str = None
    ):
        """
        Initialize ASL Gloss-to-English translator
        
        Args:
            model_name: Hugging Face model name (default: Glossa-BART)
            device: 'cuda' or 'cpu' (default: auto-detect)
        """
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully!")
    
    def translate(
        self,
        gloss: str,
        max_length: int = 128,
        num_beams: int = 4,
    ) -> str:
        """
        Translate ASL gloss to English
        
        Args:
            gloss: ASL gloss string (e.g., "YOUR NAME WHAT")
            max_length: Maximum output length
            num_beams: Number of beams for beam search
        
        Returns:
            English translation
        """
        # Tokenize (BART doesn't need prefix)
        inputs = self.tokenizer(
            gloss,
            return_tensors="pt",
            max_length=128,
            truncation=True,
        ).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )
        
        # Decode
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
    
    def translate_batch(self, glosses: list[str], **kwargs) -> list[str]:
        """Translate multiple glosses at once"""
        return [self.translate(g, **kwargs) for g in glosses]


# Create a global instance for easy import
translator = ASLTranslator()


def translate(gloss: str) -> str:
    """Quick translation function"""
    return translator.translate(gloss)


def main():
    """Demo"""
    test_glosses = [
        "ME NAME T-I-N-A. NAME YOU?",
        "IF SHE CAN'T COME, YOU LOSE CONTRACT. "
    ]
    
    print("\n" + "="*60)
    print("ASL Gloss → English Translation")
    print("="*60 + "\n")
    
    for gloss in test_glosses:
        translation = translate(gloss)
        print(f"Gloss:       {gloss}")
        print(f"Translation: {translation}")
        print("-"*60)


if __name__ == "__main__":
    main()