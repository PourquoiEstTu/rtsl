from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

from datasets import load_dataset, DatasetDict, concatenate_datasets
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


@dataclass
class DatasetOptions:
    use_aslg: bool = True
    use_flores: bool = False


class ASLGlossDataset:
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_input_length: int = 128,
        max_target_length: int = 128,
        options: DatasetOptions = None,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.options = options or DatasetOptions()

    def load_aslg(self) -> DatasetDict:
        """
        Load ASLG-PC12 dataset
        """
        logger.info("Loading ASLG-PC12 dataset...")
        ds = load_dataset("achrafothman/aslg_pc12")
        
        # Rename 'text' to 'sentence' for consistency with Flores dataset
        ds = ds.rename_column("text", "sentence")
        
        # Keep only 2 needed columns
        cols_to_keep = ["gloss", "sentence"]
        ds["train"] = ds["train"].remove_columns(
            [c for c in ds["train"].column_names if c not in cols_to_keep]
        )
        
        # Split train into train/val/test (80/10/10)
        split_1 = ds["train"].train_test_split(test_size=0.1, seed=42)
        train_ds = split_1["train"]
        temp_ds = split_1["test"]
        split_2 = temp_ds.train_test_split(test_size=0.5, seed=42)

        ds = DatasetDict({
            "train": train_ds,
            "validation": split_2["train"],
            "test": split_2["test"],
        })
        logger.info(f"ASLG-PC12 loaded: train={len(ds['train'])}, val={len(ds['validation'])}, test={len(ds['test'])}")
        return ds

    def load_flores(self) -> DatasetDict:
        """
        Load 2M-Flores-ASL dataset
        """
        logger.info("Loading 2M-Flores-ASL dataset...")
        ds = load_dataset("facebook/2M-Flores-ASL")
        
        for split in ds.keys():
            cols_to_remove = [c for c in ds[split].column_names if c not in ['gloss', 'sentence']]
            if cols_to_remove:
                ds[split] = ds[split].remove_columns(cols_to_remove)
        
        # The dataset structure is as follows:

        # DatasetDict({
        #     train: Dataset({
        #         features: ['gloss', 'sentence'],
        #         num_rows: 947
        #     }),
        #     validation: Dataset({
        #         features: ['gloss', 'sentence'],
        #         num_rows: 50
        #     }),
        #     test: Dataset({
        #         features: ['gloss', 'sentence'],
        #         num_rows: 1012
        #     })
        # })

        # Flores has 'dev' and 'devtest', create train/val/test
        train_val = ds['dev'].train_test_split(test_size=0.05, seed=42)
        
        result = DatasetDict({
            'train': train_val['train'],       # ~947 samples
            'validation': train_val['test'],   # ~50 samples  
            'test': ds['devtest']              # ~1012 samples
        })
        
        logger.info(f"Flores loaded: train={len(result['train'])}, val={len(result['validation'])}, test={len(result['test'])}")
        return result

    def load_datasets(self) -> DatasetDict:
        datasets = []
        
        if self.options.use_flores:
            datasets.append(self.load_flores())
        
        if self.options.use_aslg:
            datasets.append(self.load_aslg())
        
        if not datasets:
            raise ValueError("No dataset loaded. Must use at least one dataset (use_flores or use_aslg)")
        
        if len(datasets) == 1:
            return datasets[0]

        # Combine datasets
        combined = DatasetDict({
            split: concatenate_datasets([ds[split] for ds in datasets])
            for split in ['train', 'validation', 'test']
        })
        
        logger.info(f"Combined: train={len(combined['train'])}, val={len(combined['validation'])}, test={len(combined['test'])}")
        return combined

    def preprocess(self, examples: Dict) -> Dict:
        """
        Tokenize inputs (glosses) and targets (sentences)
        """
        # prefix for T5 "translate ASL gloss to English: "
        inputs = [f"translate ASL gloss to English: {g}" for g in examples["gloss"]]
        
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_input_length,
            truncation=True,
            # padding="max_length",
        )
        # what the tokenizer returns
        # [1359, 71, 3, 12, ..., 0, 0, 0]  padded to max length (128)
        # [translate, ASL, gloss, to, English, :, ...]
        # print('tokenized model_inputs:', model_inputs)
        
        # Tokenize targets
        labels = self.tokenizer(
            examples["sentence"],
            max_length=self.max_target_length,
            truncation=True,
            # padding="max_length",
        )
        
        # Replace padding with -100 (PyTorch ignores this in loss calculation)
        pad_id = self.tokenizer.pad_token_id
        model_inputs["labels"] = [
            [tok if tok != pad_id else -100 for tok in seq]
            for seq in labels["input_ids"]
        ]
        
        print('Keys:', model_inputs.keys())
        print('Type:', type(model_inputs))
        print('Number of samples:', len(model_inputs['input_ids']))

        # print('First input_ids:', model_inputs['input_ids'][0])
        # print('First attention_mask:', model_inputs['attention_mask'][0])
        # print('First labels:', model_inputs['labels'][0])
        
        # {
        #     'input_ids': [
        #         [13959, 71, 3, 12, 1566, 10, 5428, ..., 0, 0, 0],  # Shape: [128]
        #         # ... more samples in batch
        #     ],
        #     'attention_mask': [
        #         [1, 1, 1, 1, 1, 1, 1, ..., 0, 0, 0],  # 1 for real tokens, 0 for padding
        #         # ... more samples in batch
        #     ],
        #     'labels': [
        #         [1537, 13, 70, 5799, 43, ..., -100, -100, -100],  # Shape: [128]
        #         # ... more samples in batch
        #     ]
        # }
        return model_inputs
    
    def prepare(self) -> DatasetDict:
        """Load and tokenize datasets"""
        datasets = self.load_datasets()
        
        logger.info("Tokenizing datasets...")
        tokenized = datasets.map(
            self.preprocess,
            batched=True,
            remove_columns=datasets["train"].column_names, # remove raw columns 'gloss', 'sentence'
            desc="Tokenizing",
        )

        # {
        #     'gloss': 'MANY IX++ WRITER FROM #THE #ONION...',
        #     'sentence': 'Many of their writers have gone on...',
        #     'input_ids': [13959, 71, 3, 12, 1566, 10, 5428, ...],
        #     'attention_mask': [1, 1, 1, 1, 1, 1, 1, ...],
        #     'labels': [1537, 13, 70, 5799, 43, 1622, ...]
        # }
        # -> after remove_columns ->
        # {
        #     'input_ids': [13959, 71, 3, 12, 1566, 10, 5428, ...],
        #     'attention_mask': [1, 1, 1, 1, 1, 1, 1, ...],
        #     'labels': [1537, 13, 70, 5799, 43, 1622, ...]
        # }
        
        logger.info("Tokenization complete!")
        return tokenized


def get_dataset(tokenizer: PreTrainedTokenizerBase, config) -> DatasetDict:
    """
    Convenience function to get preprocessed datasets
    
    Args:
        tokenizer: Tokenizer for the model
        config: Configuration object with model and training settings
        
    Returns:
        Tokenized DatasetDict ready for training
    """
    options = DatasetOptions(
        use_aslg=config.data.use_aslg,
        use_flores=config.data.use_flores,
    )
    
    handler = ASLGlossDataset(
        tokenizer=tokenizer,
        max_input_length=config.model.max_input_length,
        max_target_length=config.model.max_target_length,
        options=options,
    )
    
    return handler.prepare()