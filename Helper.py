import numpy as np
from numpy.linalg import norm

def norm_matrix(A, p, q):
    '''
    Calculate norm A
    :param A: matrix to calculate norm
    :param p: first order
    :param q: second order
    :return: A_p,q
    '''
    if q == 0:
        return np.sum(norm(A, ord=p, axis=1) > 0)
    return np.sum(norm(A, ord=p, axis=1)**q)**(1.0/q)

def norm_matrix_to(A, p, q):
    '''
    Calculate norm A: (||A||p,q)^q
    :param A: matrix to calculate norm
    :param p: first order
    :param q: second order
    :return: A_p,q
    '''

    return np.sum(norm(A, ord=p, axis=1)**q)

def norm_matrix_inf_q(A, q):
    '''
    Calculate norm A: (||A||inf,q)
    :param A:
    :param q:
    :return:
    '''
    return np.sum(np.max(np.abs(A))**q)**(1.0/q)
