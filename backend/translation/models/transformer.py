import logging
from transformers import T5ForConditionalGeneration, PreTrainedModel

logger = logging.getLogger(__name__)


def create_model(model_name: str = "t5-small") -> PreTrainedModel:
    """
    Args: model_name: Name of pretrained T5 model (Options: t5-small, t5-base, t5-large)  
    Returns: Initialized T5 model
    """
    logger.info(f"Loading model: {model_name}")
    
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model loaded successfully!")
        logger.info(f"  Total parameters: {num_params:,}")
        logger.info(f"  Trainable parameters: {num_trainable:,}")
        logger.info(f"  Model size: ~{num_params * 4 / (1024**2):.1f} MB")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        raise


def get_model_info(model: PreTrainedModel) -> dict:
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': num_params,
        'trainable_params': num_trainable,
        'size_mb': num_params * 4 / (1024**2),
        'config': model.config.to_dict()
    }