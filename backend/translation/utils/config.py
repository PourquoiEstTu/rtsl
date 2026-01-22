from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ModelConfig:
    model_name: str = "t5-small"  # t5-small, t5-base, t5-large
    max_input_length: int = 128
    max_target_length: int = 128
    
    # Generation parameters
    num_beams: int = 4
    early_stopping: bool = True
    no_repeat_ngram_size: int = 2
    

@dataclass
class DataConfig:
    use_flores: bool = True  # 2M-Flores-ASL (small, high-quality)
    use_aslg: bool = False   # ASLG-PC12 (large, synthetic)
    
    # Validation split
    val_split_size: float = 0.05  # 5% of train for validation
    

@dataclass
class TrainingConfig:
    # Training
    num_epochs: int = 10
    batch_size: int = 8  # Reduced from 16 for better stability
    learning_rate: float = 3e-4  # Higher for small datasets
    weight_decay: float = 0.01
    warmup_steps: int = 100  # Reduced for small dataset
    max_grad_norm: float = 1.0  # Gradient clipping
    
    # Evaluation & Logging
    eval_strategy: str = "steps"
    eval_steps: int = 100  # More frequent evaluation
    save_steps: int = 100
    logging_steps: int = 50
    
    # Saving
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Hardware
    use_fp16: bool = True  # Mixed precision
    gradient_accumulation_steps: int = 2  # Effective batch size = 16
    dataloader_num_workers: int = 4
    
    # Paths
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    log_dir: str = "./logs"
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for dir_path in [self.output_dir, self.cache_dir, self.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.data.use_flores and not self.data.use_aslg:
            raise ValueError("Must enable at least one dataset (use_flores or use_aslg)")
        
        if self.training.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
    
    def print_config(self):
        """Pretty print configuration"""
        print("\n" + "="*60)
        print("Configuration")
        print("="*60)
        print(f"\nModel:")
        print(f"  Name: {self.model.model_name}")
        print(f"  Max input length: {self.model.max_input_length}")
        print(f"  Max target length: {self.model.max_target_length}")
        
        print(f"\nData:")
        print(f"  Use Flores: {self.data.use_flores}")
        print(f"  Use ASLG: {self.data.use_aslg}")
        
        print(f"\nTraining:")
        print(f"  Epochs: {self.training.num_epochs}")
        print(f"  Batch size: {self.training.batch_size}")
        print(f"  Gradient accumulation: {self.training.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.training.batch_size * self.training.gradient_accumulation_steps}")
        print(f"  Learning rate: {self.training.learning_rate}")
        print(f"  Output dir: {self.training.output_dir}")
        print("="*60 + "\n")


def get_config(
    model_name: str = "t5-small",
    use_flores: bool = True,
    use_aslg: bool = False,
    num_epochs: int = 10,
    batch_size: int = 8,
) -> Config:
    """Get a configuration with custom parameters"""
    config = Config()
    config.model.model_name = model_name
    config.data.use_flores = use_flores
    config.data.use_aslg = use_aslg
    config.training.num_epochs = num_epochs
    config.training.batch_size = batch_size
    return config