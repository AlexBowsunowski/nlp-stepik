import numpy as np
from typing import List 


def update_w2v_weights(center_embeddings: List[float],
                       context_embeddings: List[float],
                       center_word: int,
                       context_word: int,
                       label: int,
                       learning_rate: float):
    """
    center_embeddings - VocabSize x EmbSize
    context_embeddings - VocabSize x EmbSize
    center_word - int - identifier of center word
    context_word - int - identifier of context word
    label - 1 if context_word is real, 0 if it is negative
    learning_rate - float > 0 - size of gradient step
    """
    
    w = np.array(center_embeddings[center_word])
    d = np.array(context_embeddings[context_word])
    sigmoid = 1/(np.exp(-sum(w*d)) + 1)
    diff_sigmoid_w = d * (sigmoid - label)
    diff_sigmoid_d = w * (sigmoid - label)
    center = center_embeddings[center_word] - diff_sigmoid_w*learning_rate
    context = context_embeddings[context_word] - diff_sigmoid_d*learning_rate
    center_embeddings[center_word] = center
    context_embeddings[context_word] = context


