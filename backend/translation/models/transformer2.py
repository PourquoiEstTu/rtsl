"""
Model creation for ASL gloss-to-text translation
(Glossa-BART version)
"""

import logging
from transformers import AutoModelForSeq2SeqLM, PreTrainedModel

logger = logging.getLogger(__name__)


def create_model(
    model_name: str = "rrrr66254/Glossa-BART"
) -> PreTrainedModel:
    """
    Args:
        model_name: Hugging Face model name or local path
                    (default: rrrr66254/Glossa-BART)

    Returns:
        Initialized BART-based seq2seq model
    """
    logger.info(f"Loading Glossa-BART model: {model_name}")

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("Model loaded successfully!")
        logger.info(f"  Architecture: BART (Glossa)")
        logger.info(f"  Total parameters: {num_params:,}")
        logger.info(f"  Trainable parameters: {num_trainable:,}")
        logger.info(f"  Model size: ~{num_params * 4 / (1024**2):.1f} MB")

        return model

    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        raise


def get_model_info(model: PreTrainedModel) -> dict:
    """
    Utility function to inspect model size and configuration.
    """
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_params": num_params,
        "trainable_params": num_trainable,
        "size_mb": num_params * 4 / (1024**2),
        "config": model.config.to_dict(),
        "model_type": model.config.model_type,
    }
