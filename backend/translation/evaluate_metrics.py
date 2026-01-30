'''
Precision 
- Among all the positive predictions that we made how many are actually positive 
i.e True Positives / (True Positives + False Positives)

Recall 
- Among all the positive datapoints how many did we predict positive 
i.e True Positives / (True Positives + False Negatives)

F1 Score 
- combines both precision and recall to produce a balanced score 
i.e 2*precision*recall/(precision + recall)

BLEU 
- how many words from the predicted sentence are also present in the ground truth sentence?

ROUGE
- includes both precision and recall terms by calculating the F1 score
'''

import logging
import torch
import evaluate
from transformers import T5Tokenizer, T5ForConditionalGeneration

from data.dataset import get_dataset
from utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_predictions(model, tokenizer, dataset, device):
    model.eval()
    model.to(device)

    predictions = []
    references = []

    for example in dataset:
        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )

        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        label_ids = [x for x in example["labels"] if x != -100]
        ref = tokenizer.decode(label_ids, skip_special_tokens=True)

        predictions.append(pred)
        references.append([ref])  # BLEU expects list of references

    return predictions, references


def main():
    config = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading tokenizer and model...")
    tokenizer = T5Tokenizer.from_pretrained(config.training.output_dir)
    model = T5ForConditionalGeneration.from_pretrained(config.training.output_dir)

    logger.info("Loading test dataset...")
    datasets = get_dataset(tokenizer, config)
    test_dataset = datasets["test"]

    logger.info("Running inference on test set...")
    preds, refs = generate_predictions(model, tokenizer, test_dataset, device)

    # Load metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    # meteor = evaluate.load("meteor")

    # BLEU expects tokenized references
    bleu_score = bleu.compute(
        predictions=[p.split() for p in preds],
        references=[[r.split()] for r in refs]
    )

    rouge_score = rouge.compute(
        predictions=preds,
        references=refs
    )

    # meteor_score = meteor.compute(
    #     predictions=preds,
    #     references=refs
    # )

    logger.info("\n================ Evaluation Results ================")
    logger.info(f"BLEU:   {bleu_score['bleu']:.4f}")
    logger.info(f"ROUGE-1 F1: {rouge_score['rouge1'].mid.fmeasure:.4f}")
    logger.info(f"ROUGE-2 F1: {rouge_score['rouge2'].mid.fmeasure:.4f}")
    logger.info(f"ROUGE-L F1: {rouge_score['rougeL'].mid.fmeasure:.4f}")
    # logger.info(f"METEOR: {meteor_score['meteor']:.4f}")
    logger.info("===================================================")


if __name__ == "__main__":
    main()
