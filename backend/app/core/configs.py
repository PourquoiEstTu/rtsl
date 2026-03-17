import configparser
from dataclasses import dataclass, field
from pathlib import Path

class Config:
    def __init__(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)

        # training
        train_config = config['TRAIN']
        self.batch_size = int(train_config['BATCH_SIZE'])
        self.max_epochs = int(train_config['MAX_EPOCHS'])
        self.log_interval = int(train_config['LOG_INTERVAL'])
        self.num_samples = int(train_config['NUM_SAMPLES'])
        self.drop_p = float(train_config['DROP_P'])

        # optimizer
        opt_config = config['OPTIMIZER']
        self.init_lr = float(opt_config['INIT_LR'])
        self.adam_eps = float(opt_config['ADAM_EPS'])
        self.adam_weight_decay = float(opt_config['ADAM_WEIGHT_DECAY'])

        # GCN
        gcn_config = config['GCN']
        self.hidden_size = int(gcn_config['HIDDEN_SIZE'])
        self.num_stages = int(gcn_config['NUM_STAGES'])

    def __str__(self):
        return 'bs={}_ns={}_drop={}_lr={}_eps={}_wd={}'.format(
            self.batch_size, self.num_samples, self.drop_p, self.init_lr, self.adam_eps, self.adam_weight_decay
        )

@dataclass
class ModelConfig:
    # model_name: str = "t5-small"  # t5-small, t5-base, t5-large
    model_name: str = "rrrr66254/Glossa-BART"  # Glossa-BART for ASL gloss-to-text
    max_input_length: int = 128
    max_target_length: int = 128
    
    # Generation parameters
    num_beams: int = 4
    early_stopping: bool = True
    no_repeat_ngram_size: int = 2

@dataclass
class DataConfig:
    use_flores: bool = True  # 2M-Flores-ASL (small, high-quality)
    use_aslg: bool = False  # ASLG-PC12 (large, synthetic)
    
    # Validation split
    val_split_size: float = 0.05  # 5% of train for validation

@dataclass
class TrainingConfig:
    # Training
    num_epochs: int = 3
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
class Gloss_to_Sentence_Config:
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
        
if __name__ == '__main__':
    # Updated path - use local config file
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, 'code/TGCN/configs/local_test.ini')
    print(str(Config(config_path)))