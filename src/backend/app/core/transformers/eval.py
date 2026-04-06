"""
Evaluate ASL Gloss-to-English Translation Model
Runs evaluation on CONVERSATIONAL_ASL dataset with all metrics
"""

from translator import translator
from conversational_asl import CONVERSATIONAL_ASL
from typing import List, Tuple, Dict
import numpy as np
from datetime import datetime
import json


class ASLEvaluator:
    def __init__(self, translator):
        """Initialize evaluator with a translator instance"""
        self.translator = translator
        
        # Lazy load metrics
        self._bleu_metric = None
        self._rouge_metric = None
        self._meteor_metric = None
        self._semantic_model = None
    
    def _load_metrics(self):
        """Lazy load evaluation metrics"""
        if self._bleu_metric is None:
            from evaluate import load
            print("Loading evaluation metrics...")
            self._bleu_metric = load("bleu")
            self._rouge_metric = load("rouge")
            try:
                self._meteor_metric = load("meteor")
            except:
                print("⚠️  METEOR metric not available")
                self._meteor_metric = None
            print("✅ Metrics loaded!")
    
    def _load_semantic_model(self):
        """Lazy load sentence transformer for semantic similarity"""
        if self._semantic_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print("Loading semantic similarity model...")
                self._semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("✅ Semantic model loaded!")
            except ImportError:
                print("⚠️  Semantic similarity not available")
                print("    Install with: pip install sentence-transformers")
                self._semantic_model = None
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two sentences"""
        self._load_semantic_model()
        
        if self._semantic_model is None:
            return 0.0
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        embeddings = self._semantic_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def calculate_wer(self, prediction: str, reference: str) -> float:
        """Calculate Word Error Rate (Levenshtein distance at word level)"""
        pred_words = prediction.lower().split()
        ref_words = reference.lower().split()
        
        # Dynamic programming for edit distance
        d = [[0] * (len(ref_words) + 1) for _ in range(len(pred_words) + 1)]
        
        for i in range(len(pred_words) + 1):
            d[i][0] = i
        for j in range(len(ref_words) + 1):
            d[0][j] = j
        
        for i in range(1, len(pred_words) + 1):
            for j in range(1, len(ref_words) + 1):
                if pred_words[i-1] == ref_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = min(
                        d[i-1][j] + 1,    # deletion
                        d[i][j-1] + 1,    # insertion
                        d[i-1][j-1] + 1   # substitution
                    )
        
        return d[len(pred_words)][len(ref_words)] / max(len(ref_words), 1)
    
    def evaluate(
        self,
        test_data: List[Tuple[str, str]],
        show_examples: int = 10
    ) -> Dict:
        """
        Evaluate model on test data
        
        Args:
            test_data: List of (gloss, reference) tuples
            show_examples: Number of example predictions to show
            
        Returns:
            Dictionary with all metrics
        """
        self._load_metrics()
        self._load_semantic_model()
        
        # Separate glosses and references
        glosses = [item[0] for item in test_data]
        references = [[item[1]] for item in test_data]
        
        # Generate predictions
        print(f"\nGenerating predictions for {len(glosses)} examples...")
        predictions = []
        for i, gloss in enumerate(glosses):
            pred = self.translator.translate(gloss)
            predictions.append(pred)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(glosses)} completed...")
        
        # Calculate all metrics
        print("\nCalculating metrics...")
        results = {}
        
        # BLEU Score
        bleu_results = self._bleu_metric.compute(
            predictions=predictions,
            references=references
        )
        results['bleu'] = bleu_results['bleu']
        results['bleu_1'] = bleu_results['precisions'][0]
        results['bleu_2'] = bleu_results['precisions'][1]
        results['bleu_3'] = bleu_results['precisions'][2]
        results['bleu_4'] = bleu_results['precisions'][3]
        
        # ROUGE Scores
        rouge_results = self._rouge_metric.compute(
            predictions=predictions,
            references=[r[0] for r in references]
        )
        results['rouge_1'] = rouge_results['rouge1']
        results['rouge_2'] = rouge_results['rouge2']
        results['rouge_l'] = rouge_results['rougeL']
        
        # METEOR Score (if available)
        if self._meteor_metric is not None:
            try:
                meteor_results = self._meteor_metric.compute(
                    predictions=predictions,
                    references=[r[0] for r in references]
                )
                results['meteor'] = meteor_results['meteor']
            except:
                results['meteor'] = None
        
        # Semantic Similarity (if available)
        if self._semantic_model is not None:
            similarities = []
            for pred, ref in zip(predictions, [r[0] for r in references]):
                sim = self.semantic_similarity(pred, ref)
                similarities.append(sim)
            
            results['semantic_similarity_mean'] = float(np.mean(similarities))
            results['semantic_similarity_std'] = float(np.std(similarities))
            results['semantic_similarity_min'] = float(np.min(similarities))
            results['semantic_similarity_max'] = float(np.max(similarities))
            results['semantic_similarities'] = similarities
        
        # Exact Match Accuracy
        exact_matches = sum(
            1 for p, r in zip(predictions, [ref[0] for ref in references])
            if p.lower().strip() == r.lower().strip()
        )
        results['exact_match'] = exact_matches / len(predictions)
        
        # Word Error Rate
        wer_scores = []
        for pred, ref in zip(predictions, [r[0] for r in references]):
            wer = self.calculate_wer(pred, ref)
            wer_scores.append(wer)
        results['wer_mean'] = float(np.mean(wer_scores))
        results['wer_std'] = float(np.std(wer_scores))
        
        # Store metadata
        results['total_examples'] = len(test_data)
        results['model_name'] = self.translator.model_name
        results['timestamp'] = datetime.now().isoformat()
        results['predictions'] = predictions
        results['references'] = [r[0] for r in references]
        results['glosses'] = glosses
        
        # Print results
        self._print_results(results, test_data, predictions, show_examples)
        
        # Save to JSON
        self._save_results(results)
        
        return results
    
    def _print_results(
        self,
        results: Dict,
        test_data: List[Tuple],
        predictions: List[str],
        show_examples: int
    ):
        """Print formatted evaluation results"""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        print(f"Model: {results['model_name']}")
        print(f"Dataset: CONVERSATIONAL_ASL")
        print(f"Total examples: {results['total_examples']}\n")
        
        # BLEU Scores
        print("BLEU Scores:")
        print(f"  Overall BLEU: {results['bleu']:.4f} ({results['bleu']*100:.2f}%)")
        print(f"  BLEU-1:       {results['bleu_1']:.4f}")
        print(f"  BLEU-2:       {results['bleu_2']:.4f}")
        print(f"  BLEU-3:       {results['bleu_3']:.4f}")
        print(f"  BLEU-4:       {results['bleu_4']:.4f}")
        
        # ROUGE Scores
        print("\nROUGE Scores:")
        print(f"  ROUGE-1: {results['rouge_1']:.4f}")
        print(f"  ROUGE-2: {results['rouge_2']:.4f}")
        print(f"  ROUGE-L: {results['rouge_l']:.4f}")
        
        # METEOR (if available)
        if results.get('meteor') is not None:
            print(f"\nMETEOR Score: {results['meteor']:.4f}")
        
        # Semantic Similarity
        if 'semantic_similarity_mean' in results:
            print("\nSemantic Similarity:")
            print(f"  Mean:   {results['semantic_similarity_mean']:.4f}")
            print(f"  Std:    {results['semantic_similarity_std']:.4f}")
            print(f"  Min:    {results['semantic_similarity_min']:.4f}")
            print(f"  Max:    {results['semantic_similarity_max']:.4f}")
        
        # Other Metrics
        print("\nOther Metrics:")
        print(f"  Exact Match:  {results['exact_match']:.4f} ({results['exact_match']*100:.2f}%)")
        print(f"  WER (mean):   {results['wer_mean']:.4f}")
        print(f"  WER (std):    {results['wer_std']:.4f}")
        
        # Quality Interpretation
        print("\n" + "-"*70)
        print("Quality Assessment:")
        if results['bleu'] > 0.5:
            print("  🟢 Excellent translation quality (BLEU > 0.5)")
        elif results['bleu'] > 0.3:
            print("  🟡 Good translation quality (BLEU > 0.3)")
        elif results['bleu'] > 0.15:
            print("  🟠 Fair translation quality (BLEU > 0.15)")
        else:
            print("  🔴 Poor translation quality (BLEU < 0.15)")
        
        if 'semantic_similarity_mean' in results:
            if results['semantic_similarity_mean'] > 0.8:
                print("  🟢 High semantic similarity (> 0.8)")
            elif results['semantic_similarity_mean'] > 0.6:
                print("  🟡 Moderate semantic similarity (> 0.6)")
            else:
                print("  🟠 Low semantic similarity (< 0.6)")
        
        print("="*70)
        
        # Show example predictions
        if show_examples > 0:
            print(f"\nSample Predictions (showing first {show_examples}):")
            print("-"*70)
            
            for i in range(min(show_examples, len(test_data))):
                gloss = test_data[i][0]
                reference = test_data[i][1]
                prediction = predictions[i]
                
                # Show similarity if available
                if 'semantic_similarities' in results:
                    sim = results['semantic_similarities'][i]
                    sim_str = f" (sim: {sim:.3f})"
                else:
                    sim_str = ""
                
                print(f"\n{i+1}.")
                print(f"  Gloss:     {gloss}")
                print(f"  Reference: {reference}")
                print(f"  Predicted: {prediction}{sim_str}")
    
    def _save_results(self, results: Dict):
        """Save results to JSON file"""
        output = {
            "timestamp": results['timestamp'],
            "model": results['model_name'],
            "dataset": "CONVERSATIONAL_ASL",
            "total_examples": results['total_examples'],
            "metrics": {
                "bleu": results['bleu'],
                "bleu_1": results['bleu_1'],
                "bleu_2": results['bleu_2'],
                "bleu_3": results['bleu_3'],
                "bleu_4": results['bleu_4'],
                "rouge_1": results['rouge_1'],
                "rouge_2": results['rouge_2'],
                "rouge_l": results['rouge_l'],
                "exact_match": results['exact_match'],
                "wer_mean": results['wer_mean'],
                "wer_std": results['wer_std'],
            },
            "examples": [
                {
                    "gloss": g,
                    "reference": r,
                    "prediction": p
                }
                for g, r, p in zip(
                    results['glosses'],
                    results['references'],
                    results['predictions']
                )
            ]
        }
        
        # Add optional metrics
        if results.get('meteor') is not None:
            output['metrics']['meteor'] = results['meteor']
        
        if 'semantic_similarity_mean' in results:
            output['metrics']['semantic_similarity'] = {
                'mean': results['semantic_similarity_mean'],
                'std': results['semantic_similarity_std'],
                'min': results['semantic_similarity_min'],
                'max': results['semantic_similarity_max'],
            }
        
        filename = "evaluation_results.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Results saved to {filename}")


def main():
    """Run evaluation on CONVERSATIONAL_ASL dataset"""
    print("="*70)
    print("ASL GLOSS → ENGLISH TRANSLATION EVALUATION")
    print("="*70)
    print(f"Dataset: CONVERSATIONAL_ASL ({len(CONVERSATIONAL_ASL)} examples)")
    
    # Initialize evaluator
    evaluator = ASLEvaluator(translator)
    
    # Run evaluation
    results = evaluator.evaluate(
        test_data=CONVERSATIONAL_ASL,
        show_examples=10
    )
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"\nKey Metrics:")
    print(f"  BLEU:                {results['bleu']:.4f}")
    print(f"  ROUGE-L:             {results['rouge_l']:.4f}")
    if 'semantic_similarity_mean' in results:
        print(f"  Semantic Similarity: {results['semantic_similarity_mean']:.4f}")
    print(f"  Exact Match:         {results['exact_match']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()

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
BLEU-1: unigram precision/ individual words match (e.g. "I" vs "I")
BLEU-2: bigram precision/ 2-word phrases match (e.g. "I am" vs "I am")
BLEU-3: trigram precision/ 3-word phrases match (e.g. "I am happy" vs "I am happy")

ROUGE
- includes both precision and recall terms by calculating the F1 score
'''