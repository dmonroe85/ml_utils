import numpy as np
from functools import partial
from scipy import optimize as opt

# Collaborating Filtering
def cofi_getH(X, Theta):
    x_ = np.reshape(X, (1, X.size))
    t_ = np.reshape(Theta, (1, Theta.size))
    H = np.hstack((x_, t_))
    return H

def cofi_getXT(H, Y):
    sY = Y.shape
    nM, nU = sY
    nF = H.size/np.sum(sY) # Number of Features
    sX = nM*nF
    try:
        if H.shape[0] == 1:
            X = np.reshape(H[:, :sX], (nM, nF))
            T = np.reshape(H[:, sX:], (nU, nF))
        else:
            X = np.reshape(H[:sX], (nM, nF))
            T = np.reshape(H[sX:], (nU, nF))
    except IndexError as e:
        print H.shape
        print nM, nU, nF, sX
        raise IndexError(e)
    return X, T

def cofi_J(X, Theta, Y, R, L=0):
    cost = 0.5*np.sum(((X.dot(Theta.T) - Y)*R)**2) + \
           0.5*L*(np.sum(Theta**2) + np.sum(X**2))
    # print cost
    return cost

def cofi_dJ(X, Theta, Y, R, L=0):
    E = (X.dot(Theta.T) - Y)*R
    dX = E.dot(Theta) + L*X
    dTheta = E.T.dot(X) + L*Theta
    H = cofi_getH(dX, dTheta)
    return H

def cofi_Cost(H, Y, R, L=0):
    X, T = cofi_getXT(H, Y)
    return cofi_J(X, T, Y, R, L)

def cofi_Grad(H, Y, R, L=0):
    X, T = cofi_getXT(H, Y)
    return np.ndarray.flatten(cofi_dJ(X, T, Y, R, L))

def cofi_denormalize(Y_norm, Y_mean, Y_std):
    m, _ = Y_norm.shape
    Y = np.zeros(Y_norm.shape)
    for idx in range(m):
        Y[idx, :] = Y_norm[idx, :]*Y_std[idx] + Y_mean[idx]
    return Y

def cofi_normalize(Y, R):
    m, _ = Y.shape
    Y_mean = np.zeros((m, 1))
    Y_std  = np.zeros((m, 1))
    Y_norm = np.zeros(Y.shape)
    for idx in range(m):
        r_idx = R[idx, :] == 1
        Y_mean[idx] = np.mean(Y[idx, r_idx])
        Y_std[idx]  = np.std( Y[idx, r_idx])
        Y_norm[idx, r_idx] = (Y[idx, r_idx] - Y_mean[idx])/Y_std[idx]
    return Y_norm, Y_mean, Y_std

def cofi_ls(Y, R, L=0, F=10, X=None, T=None, iterations=100):
    Ynorm, Ymean, Ystd = cofi_normalize(Y, R)

    N = Y.shape[1]
    M = Y.shape[0]
    if X == None:
        X = np.random.normal(size=(N, F))

    if T == None:
        T = np.random.normal(size=(M, F))

    opts = {'maxiter': 100, 'disp': True}

    initialH = cofi_getH(X, T)
    J_ = partial(cofi_Cost, Y=Ynorm, R=R, L=L)
    dJ_ = partial(cofi_Grad, Y=Ynorm, R=R, L=L)

    H_opt = opt.fmin_cg(J_, initialH, fprime=dJ_, maxiter=iterations)
    X_opt, T_opt = cofi_getXT(H_opt, Ynorm)
    XT_opt = X_opt.dot(T_opt.T)

    P = cofi_denormalize(XT_opt, Ymean, Ystd)
    return X_opt, T_opt, P