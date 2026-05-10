import numpy as np

def softmax(logits):

    # handle overflow
    logits = logits - np.max(logits)

    # denom calc
    summ_denom = np.sum(np.exp(logits))

    return np.exp(logits)/summ_denom

def cross_entropy(probs, y_true):
    return (-1.0)*np.log(probs[y_true])

# testing area

# setup logits
logits = np.array([2.0, 1.0, 0.1])
print(f"Logits: {logits}")

# calc probs
probs = softmax(logits)
print(f"Probabilities: {probs}")

# cross entropy loss
ce_loss = cross_entropy(probs, 0)
print(f"Cross Entropy loss for idx 0: {ce_loss}")