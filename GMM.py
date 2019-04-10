
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.stats import norm
from sklearn.datasets.samples_generator import make_blobs

#seed data for reconstruction
np.random.seed(0)

#create data
X,Y = make_blobs(cluster_std=1.5,random_state=20,n_samples=500,centers=3)
# Stratch dataset to get ellipsoid data
X = np.dot(X,np.random.RandomState(0).randn(2,2))

# plt.scatter(X[:, 0], X[:, 1], c='grey', s=30)
# plt.axis('equal')
# plt.show()


import scipy

def E_step(X, pi, mu, sigma):
    """
    Performs E-step on GMM model
    Each input is numpy array:
    X: (N x d), data points
    pi: (C), mixture component weights
    mu: (C x d), mixture component means
    sigma: (C x d x d), mixture component covariance matrices

    Returns:
    gamma: (N x C), probabilities of clusters for objects
    """
    N = X.shape[0]  # number of objects
    C = pi.shape[0]  # number of clusters
    d = mu.shape[1]  # dimension of each object
    gamma = np.zeros((N, C))  # distribution q(T)
    for i in range(C):
        gamma[:, i] = pi[i] * scipy.stats.multivariate_normal(mean=mu[i], cov=sigma[i]).pdf(X)
    for n in range(N):
        z = np.sum(gamma[n, :])
        gamma[n, :] = gamma[n, :] / (z)

    return gamma

#gamma = E_step(X, pi0, mu0, sigma0)

def M_step(X, gamma):
    """
    Performs M-step on GMM model
    Each input is numpy array:
    X: (N x d), data points
    gamma: (N x C), distribution q(T)

    Returns:
    pi: (C)
    mu: (C x d)
    sigma: (C x d x d)
    """
    N = X.shape[0]  # number of objects
    C = gamma.shape[1]  # number of clusters
    d = X.shape[1]  # dimension of each object


    mu = []
    sigma = []
    pi = []
    reg_cov = 1e-6 * np.identity(len(X[0]))
    for i in range(C):
        r_x = np.sum(X * gamma[:, i].reshape(len(X), 1), axis=0)
        r_i = sum(gamma[:, i])
        mu.append(r_x / r_i)
        s = ((1 / r_i) * np.dot((np.array(gamma[:, i]).reshape(len(X), 1) * (X - mu[i])).T, (X - mu[i]))) + reg_cov
        sigma.append(s)
        pi.append(r_i / np.sum(gamma))
    return np.array(pi), np.array(mu), np.array(sigma)

#pi, mu, sigma = M_step(X, gamma)


def compute_vlb(X, pi, mu, sigma, gamma):
    """
    Each input is numpy array:
    X: (N x d), data points
    gamma: (N x C), distribution q(T)
    pi: (C)
    mu: (C x d)
    sigma: (C x d x d)

    Returns value of variational lower bound
    """
    N = X.shape[0]  # number of objects
    C = gamma.shape[1]  # number of clusters
    d = X.shape[1]  # dimension of each object

    L1 = []
    L2 = []

    for i in range(X.shape[0]):
        for j in range(C):
            Norm=scipy.stats.multivariate_normal(mean=mu[j], cov=sigma[j]).pdf(X)
            if np.logical_and(np.isnan(Norm[i])==False,np.isinf(Norm[i])==False):
                if Norm[i] >0:
                    L1.append(gamma[i, j] * (np.log(pi[j]) + np.log(Norm[i])))
                if gamma[i, j]>0:
                    L2.append((gamma[i, j] * np.log(gamma[i, j])))
    L1 = np.array(L1)
    L2 = np.array(L2)
    L1 = L1[~np.isnan(L1)]
    L2 = L2[~np.isnan(L2)]
    loss = np.sum(L1) - np.sum(L2)

    return loss

#loss = compute_vlb(X, pi, mu, sigma, gamma)


def create_mean():
    mean=np.random.randint(min(X[:, 0]), max(X[:, 0]), size=(3, len(X[0])))
    return mean


def create_sigma():
    I = np.diag((1, 1))
    sigma_1 = [I, I, I]
    sigma_1 = np.array(sigma_1)
    return sigma_1

def create_pi():
    a = []
    pi_1 = []
    for i in range(3):
        a.append(random.randint(1, 10))
    z = sum(a)
    for x in range(3):
        pi_1.append(a[x] / z)
    return np.array(pi_1)

def Best_config(loses,pis0,mus0,sigmas0):
    best_config=[]
    for key, value in loses.items():
        max_val = max(list(loses.values()))
        if value == max_val:
            return loses[key],pis0[key],mus0[key],sigmas0[key]


def train_EM(X, C, rtol=1e-3, max_iter=100, restarts=30):
    '''
    Starts with random initialization *restarts* times
    Runs optimization until saturation with *rtol* reached
    or *max_iter* iterations were made.

    X: (N, d), data points
    C: int, number of clusters
    '''
    N = X.shape[0]  # number of objects
    d = X.shape[1]  # dimension of each object
    best_loss = None
    best_pi = None
    best_mu = None
    best_sigma = None
    pis = {}
    mus = {}
    sigmas = {}
    losses={}

    for _ in range(restarts):
        try:
            ### creating parameters
            mu=create_mean()
            sigma=create_sigma()
            pi=create_pi()

            ### The Functions
            Stop_point = False
            loss=1
            for ind in range(max_iter):
                if Stop_point ==False:
                    if np.logical_and(np.isnan(mu).any() == False, np.isinf(mu).any() == False):
                        if np.logical_and(np.isnan(sigma).any() == False, np.isinf(sigma).any() == False):
                            gamma = E_step(X, pi, mu, sigma)
                            pi, mu, sigma = M_step(X, gamma)
                            if ind>0 and rtol >= np.abs((compute_vlb(X, pi, mu, sigma, gamma)-loss) / loss):
                                Stop_point =True
                            else:
                                loss = compute_vlb(X, pi, mu, sigma, gamma)
                else:
                    break
            losses[_]=loss
            pis[_]=pi
            mus[_]=mu
            sigmas[_]=sigma

        except Exception:
                print("array must not contain infs or NaNs")
                pass
    best_loss, best_pi, best_mu, best_sigma=Best_config(losses,pis,mus,sigmas)

    return best_loss, best_pi, best_mu, best_sigma

best_loss, best_pi, best_mu, best_sigma = train_EM(X, 3)

gamma = E_step(X, best_pi, best_mu, best_sigma)
labels = gamma.argmax(axis=1)
colors = np.array([(31, 119, 180), (255, 127, 14), (44, 160, 44)]) / 255.
plt.scatter(X[:, 0], X[:, 1], c=colors[labels], s=30)
plt.axis('equal')
plt.show()
