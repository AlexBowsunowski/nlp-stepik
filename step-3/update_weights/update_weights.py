import numpy as np


def update_w2v_weights(center_embeddings, context_embeddings, center_word, context_word, label, learning_rate):
    """
    center_embeddings - VocabSize x EmbSize
    context_embeddings - VocabSize x EmbSize
    center_word - int - identifier of center word
    context_word - int - identifier of context word
    label - 1 if context_word is real, 0 if it is negative
    learning_rate - float > 0 - size of gradient step
    """
    w = center_embeddings[center_word]
    d = context_embeddings[context_word]

    sigma = 1/(1+np.exp(-w@d))
    grad_w = (sigma-label) * d
    grad_d = (sigma-label) * w

    w -= learning_rate * grad_w
    d -= learning_rate * grad_d



