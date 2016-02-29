import numpy as np

def computeNumericGradient(H, J):
    numgrad = np.zeros(H.shape)
    perturb = np.zeros(H.shape)
    e = 1e-4
    for p in range(H.size):
        perturb[0, p] = e
        loss1 = J(H - perturb)
        loss2 = J(H + perturb)

        numgrad[0, p] = (loss2-loss1)/(2*e)
        perturb[0, p] = 0

    return numgrad

def append_predictions(inputs, predictions):
    in_list_T = zip(*(list(inputs)))
    in_list_T.append(list(predictions))
    return np.array(in_list_T).T
