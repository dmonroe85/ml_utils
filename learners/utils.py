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


def multiclass_prediction(input_data, pipeline_list, targets=np.array([]), verbose=0):
    predictions = [];
    for jdx, pipeline in enumerate(pipeline_list):
        targets_j = targets[:, jdx]

        if targets.size:
            pipeline.fit(input_data, targets_j)

        if targets.size and verbose >= 1:
            print jdx
            if verbose >= 2:
                print pipeline.score(input_data, targets_j)

        y = pipeline.predict_proba(input_data)
        # print y[y > 0.0].size
        predictions.append(y)
    return np.array(predictions).T