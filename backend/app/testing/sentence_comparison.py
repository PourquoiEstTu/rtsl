from sentence_transformers import SentenceTransformer

def sentence_comparison(sentence1:str, sentence2:str):
    """ Compares two strings and returns a float number from 0 to 1 where 1 means the sentences have identical meaning through transformers.
    Inputs: Two strings.
    Output: Value between 0 to 1 representing how similar the sentences are.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding1 = model.encode(sentence1)
    embedding2 = model.encode(sentence2)
    similarity = model.similarity(embedding1, embedding2)
    return similarity.item()