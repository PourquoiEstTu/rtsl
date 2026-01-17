from sentence_transformers import SentenceTransformer
from jiwer import wer

# returns close to 1 if two sentences are semantically similar
def sentence_semantic_comparison(label:str, predicted:str):
    """ Compares two strings and returns a float number from 0 to 1 where 1 means the sentences have identical meaning through transformers.
    Inputs: Label and predicted sentence.
    Output: Value between 0 to 1 representing how similar the meanings are using cosine similarity.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    label_embedding = model.encode(label)
    predicted_embedding = model.encode(predicted)
    similarity = model.similarity(label_embedding, predicted_embedding)
    return similarity.item()

# returns close to 0 if two sentences are syntactically similar
def sentence_syntactic_comparison(label:str, predicted:str):
    """ Compares two strings and returns the word error rate between the label and prediction. 
    Inputs: Label and predicted sentence.
    Output: Float that roughly represents the percentage of words incorrect in the label (can be greater than 1 for extremely bad predictions).
    """
    return wer(label, predicted)