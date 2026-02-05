import sys
import os
import logging
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    EarlyStoppingCallback,
)

# Add parent directory to path to import custom modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import get_dataset
from utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    config = Config()
    if config.model.model_name.startswith("rrrr66254"):
        from models.transformer2 import create_model, get_model_info
    else:
        from models.transformer import create_model, get_model_info
    # Override defaults if needed

    # config.data.use_flores = True
    # config.data.use_aslg = False
    # config.model.model_name = "t5-small"
    # config.training.num_epochs = 10
    # config.training.batch_size = 8
    
    config.print_config()
    
    set_seed(config.seed)
    logger.info(f"Random seed set to {config.seed}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load Tokenizer
    # used in decoding from tokens to text
    logger.info(f"Loading tokenizer: {config.model.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    logger.info("Tokenizer loaded!")
    
    # Load Datasets
    logger.info("Loading and preprocessing datasets...")
    try:
        datasets = get_dataset(tokenizer, config)
        
        logger.info("\nDataset Statistics:")
        logger.info(f"  Train samples: {len(datasets['train'])}")
        logger.info(f"  Validation samples: {len(datasets['validation'])}")
        logger.info(f"  Test samples: {len(datasets['test'])}")
        
        
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}", exc_info=True)
        raise
    
    # Load Model
    try:
        model = create_model(config.model.model_name)
        
        info = get_model_info(model)
        logger.info(f"\nModel Information:")
        logger.info(f"  Architecture: {config.model.model_name}")
        logger.info(f"  Total parameters: {info['total_params']:,}")
        logger.info(f"  Trainable parameters: {info['trainable_params']:,}")
        logger.info(f"  Model size: {info['size_mb']:.1f} MB")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise
    
    # Data Collator
    # for dynamic padding during batching, pad to max length in current batch
    # Automatically pads to same length:
    # input_ids: [[1, 2, 3], [6, 7, 0]]   Padded with 0
    # labels: [[4, 5, -100], [8, 9, 10]]  Padded with -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding='max_length',
        max_length=config.model.max_target_length,
    )
    
    # training
    training_args = TrainingArguments(

        output_dir=config.training.output_dir, # where to save model checkpoints
        logging_dir=config.training.log_dir,   # where to save logs
        
        num_train_epochs=config.training.num_epochs,
        per_device_train_batch_size=config.training.batch_size,
        per_device_eval_batch_size=config.training.batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        
        # Optimization
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        max_grad_norm=config.training.max_grad_norm,
        
        # Evaluation
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        
        # Saving
        save_strategy="steps",
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        
        # Logging
        logging_steps=config.training.logging_steps,
        logging_first_step=True,
        report_to=["tensorboard"],
        
        # Hardware
        fp16=config.training.use_fp16 and torch.cuda.is_available(), # Mixed precision, half precision if GPU supports
        dataloader_num_workers=config.training.dataloader_num_workers,
        
        # Misc
        seed=config.seed,
        remove_unused_columns=True,
        push_to_hub=False,
    )
    
    # Callbacks for early stopping
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=20,  # Stop if no improvement for 5 evals
            early_stopping_threshold=0.0
        )
    ]
    
    # Intialize trainer
    logger.info("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
    
    # Train
    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60 + "\n")
    
    try:
        # Train
        train_result = trainer.train()
        
        logger.info("\nSaving final model...")
        trainer.save_model()
        tokenizer.save_pretrained(config.training.output_dir)
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        logger.info("\nTraining metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        
    # except KeyboardInterrupt:
    #     logger.warning("\n  Training interrupted by user!")
    #     save_path = Path(config.training.output_dir) / "interrupted"
    #     trainer.save_model(save_path)
    #     tokenizer.save_pretrained(save_path)
    #     logger.info(f"Model saved to: {save_path}")
    #     return
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise
    
    # evaluate
    logger.info("\n" + "="*60)
    logger.info("Evaluating on test set...")
    logger.info("="*60)
    
    try:
        test_results = trainer.evaluate(
            datasets['test'],
            metric_key_prefix="test"
        )
        
        trainer.log_metrics("test", test_results)
        trainer.save_metrics("test", test_results)
        
        logger.info("\nTest metrics:")
        for key, value in test_results.items():
            logger.info(f"  {key}: {value}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
    
    logger.info("\n" + "="*60)
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to: {config.training.output_dir}")
    logger.info(f"Logs saved to: {config.training.log_dir}")
    logger.info("="*60 + "\n")
    
    logger.info("To view training progress:")
    logger.info(f"  tensorboard --logdir {config.training.log_dir}")

    logger.info("\n" + "=" * 60)
    logger.info("Running sample predictions")
    logger.info("=" * 60)

    sample_glosses = [
        "HELLO MY NAME J-O-H-N",
        "NICE MEET YOU",
        "HOW YOU",
        "I FINE",
        "YOUR NAME WHAT",
        "WHERE YOU LIVE",
        "I HUNGRY",
        "I THIRSTY WANT WATER",
        "TOMORROW STORE I GO",
        "YESTERDAY SCHOOL I GO",
        "LAST-WEEK MOVIE I WATCH",
        "PAST NIGHT I SLEEP EARLY",
        "COFFEE I DRINK WANT",
        "MUSIC I LISTEN ENJOY",
        "IF RAIN TOMORROW GAME CANCEL",
        "TEACHER SAY TEST DIFFICULT BUT I STUDY HARD",
        "AFTER EAT DINNER I WATCH TV",
    ]

    for gloss in sample_glosses:
        prediction = run_inference(model, tokenizer, gloss, device)
        logger.info(f"\nGloss: {gloss}")
        logger.info(f"Prediction: {prediction}")


def run_inference(model, tokenizer, gloss: str, device: str):
    model.eval()
    model.to(device)

    input_text = f"translate ASL gloss to English: {gloss}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction


if __name__ == "__main__":
    main()

# logger.info("\nRunning final evaluation (BLEU / ROUGE / METEOR)...")
# os.system("python evaluate.py")

'''
total number of optimizer steps = (number of training samples / (batch size * gradient accumulation steps)) * number of epochs
For example, with 87710 training samples, batch size 8, gradient accumulation steps 2, and 50 epochs:
steps_per_epoch = 87710 / (8 * 2) = 5481.875
total optimizer steps = 5481.875 * 50 = 274093.75 steps


'''