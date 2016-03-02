import numpy as np

def correlation_matrix(input_data, targets):
    return [np.corrcoef(series, targets)[0,1] for series in transpose_list(input_data)]

def pearson_matrix(input_data, targets):
    C = []; P = []
    for feature in input_data.T:
        c, p = stats.pearsonr(feature, targets)
        C.append(c)
        P.append(p)
    return np.array(C), np.array(P)

def calculate_P(x):
    return sum(x)/(len(x) + 0.0)

def findCoOccurrences(x, y):
    return [int(int(xi) & int(yi)) for xi, yi in zip(x,y)]

def conditional_P(x, y):
    """ Returns the conditional probability P(X|Y) using n(X&Y)/n(Y).
    """
    return 0 if sum(y) == 0 else sum(findCoOccurrences(x, y))/(sum(y) + 0.0)

def calc_entropy_term(x):
    """ Calculates individual terms of the entropy sum.
    """
    return 0 if x == 0 else -x*np.log2(x)

def entropy(probabilities):
    """ Calculates the total entropy over a group of probabilities.
    """
    return sum([hi(p) for p in x])

def information_gain(t, c):
    """ Computes the information gain of a series of observations t and their
    respective classes C.
    """
    information_gain.idx += 1
    not_t = [int(not ti) for ti in t];
    P_t = calculate_P(t); P_not_t = calculate_P(not_t)
    ig = 0
    for ci in set(c):
        C_i = [int(ci == cj) for cj in c]
        ig += calc_entropy_term(calculate_P(C_i)) - \
              P_t*calc_entropy_term(conditional_P(C_i, t)) - \
              P_not_t*calc_entropy_term(conditional_P(C_i, not_t))
    return ig
information_gain.idx = 0

def IG_matrix(input_data, targets):
    return [information_gain(series, targets) for series in [list(x) for x in zip(*input_data)]]

def logloss(predictions, targets):
    e = 10.0**(-15)
    return -np.sum(targets*np.log(np.clip(predictions, e, 1.0-e)))