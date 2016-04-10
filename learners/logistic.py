import numpy as np
import math
from scipy.special import expit


def logistic_J(T, X, Y, L):
    m = length(Y)
    h = expit(X.T.dot(T))
