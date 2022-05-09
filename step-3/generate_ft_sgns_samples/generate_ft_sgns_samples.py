import numpy as np
from typing import List 

def generate_ft_sgns_samples(text: np.array,
                              window_size: int,
                              vocab_size: int,
                              ns_rate: int,
                              token2subwords: List[int] 
                              ) -> List:
    """
    text - list of integer numbers - ids of tokens in text
    window_size - odd integer - width of window
    vocab_size - positive integer - number of tokens in vocabulary
    ns_rate - positive integer - number of negative tokens to sample per one positive sample



    returns list of training samples (CenterWord, CtxWord, Label)
    """
    half_window = window_size // 2
    samples = []

    for center in range(len(text)):
        left_border = max(center-half_window, 0)
        right_border = min(center+half_window+1, len(text))
        center_n_gramm = [text[center]] + token2subwords[text[center]]
        center_n_gramm = list(set(center_n_gramm))
        positive_samples = [[center_n_gramm, text[pos], 1] 
                          for pos in range(left_border, right_border)
                          if pos != center]
        negative_samples = [[center_n_gramm, neg, 0] for neg in np.random.choice(vocab_size,
                          ns_rate*len(positive_samples))]
        samples += positive_samples + negative_samples
    return samples


if __name__ == "__main__":
    
    # First set of arguments
    text = [1, 2, 0, 1, 4, 0, 4, 1, 5, 4, 5, 4, 5, 1]
    window_size = 3
    vocab_size = 6
    ns_rate = 2
    token2subwords = [[17], [10, 12], [20, 20], [7, 13], [], [7, 11]]
    result = generate_ft_sgns_samples(text, window_size, vocab_size, ns_rate, token2subwords)
    
    print(result)