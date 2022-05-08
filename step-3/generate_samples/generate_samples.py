import numpy as np


def generate_w2v_sgns_samples(text: np.array,
                              window_size: int,
                              vocab_size: int,
                              ns_rate: int
                              ) -> np.array:
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
        positive_samples = [[text[center], text[pos], 1] 
                          for pos in range(left_border, right_border)
                          if pos != center]
        negative_samples = [[text[center], neg, 0] for neg in np.random.choice(vocab_size,
                          ns_rate*len(positive_samples))]
        samples += positive_samples + negative_samples
    return np.array(samples)


if __name__ == "__main__":
    
    # First set of arguments
    text = np.array([1, 0, 1, 0, 0, 5, 0, 3, 5, 5, 3, 0, 5, 0, 5, 2, 0, 1, 3])
    window_size = 3
    vocab_size = 6
    ns_rate = 1
    
    result = generate_w2v_sgns_samples(text, window_size, vocab_size, ns_rate)
    assert sum(result[:, 2]) == 36 and len(result) - sum(result[:, 2]) == 36

    