from ml_utils import IG_matrix
import numpy as np

# Test Data
y = [1, 2, 2, 3, 3, 3]


S = [
    [1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [1, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 0],
]

inputs = [list(x) for x in zip(*S)]

S_np = np.array(S)

def coOccurranceTest():
    findCoOccurrences()

def IG_test():
    print "Information Gain Testing."
    SCORES = []

    print (y,)
    SCORES = IG_matrix(inputs, y)

    for score in reversed(sorted(zip(S, SCORES), key=lambda x: x[1])):
        print score

    SCORES_np = IG_matrix(S_np.T, y)

    for score in reversed(sorted(zip(S, SCORES_np), key=lambda x: x[1])):
        print score

if __name__ == '__main__':
    IG_test()