from __future__ import print_function, division
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def generate_spsd_matrix(d, random_seed=None):
    """Reproducible random generation of symmetric
    positive semi-definite matrices.

    Parameters
    ----------
    d : integer
        Number of dimensions.

    random_seed : integer
        Random seed number.

    Returns
    ----------
    A : ndarray, shape (n,n)
        Symmetric positive definite matrix.

    """

    if random_seed is not None:
        np.random.seed(random_seed)
    A = np.random.randn(d, d)
    return np.dot(A.T, A)


def IsPositiveSemiDefinite(m):
    eigen = np.linalg.eigh(m)
    return np.all(eigen[0] > 0)


def GenerateGaussianSamples(mean, covariance, numberOfSamples):
    mean = np.array(mean)
    dimensionality = mean.shape[0]
    x = np.random.randn(numberOfSamples, dimensionality)
    # Eigen decomposition of symmetric matrix
    D, U = np.linalg.eigh(covariance)
    D = np.sqrt(np.diag(D))
    Y = mean + np.dot(x, D).dot(U.T)
    return Y, x


def generate_gaussian_data(n_samples, n_features=None, mu=None, cov=None, random_seed=None):
    """Generates a multivariate gaussian dataset.

    Parameters
    ----------

    n_samples : integer
        Number of samples.

    n_features : integer
        Number of dimensions (features).

    mu : array, optional (default random), shape (n_features,)
        Mean vector of normal distribution.

    cov : array, optional (default random), shape (n_features,n_features)
        Covariance matrix of normal distribution.

    random_seed : integer
        Random seed.

    Returns
    -------

    x : array, shape (n_samples, n_features)
        Data matrix arranged in rows (i.e.
        columns correspond to features and
        rows to observations).

    mu : array, shape (n_features,)
        Mean vector of normal distribution.

    cov : array, shape (n_features,n_features)
        Covariance matrix of normal distribution.

    Raises
    ------

    ValueError when the shapes of mu and C are not compatible
    with n_features.

    """

    if random_seed is not None:
        np.random.seed(random_seed)

    if mu is None:
        mu = np.random.randn(n_features, )
    else:
        if n_features is None:
            n_features = mu.shape[0]
        else:
            if mu.shape[0] != n_features:
                raise ValueError("Shape mismatch between mean and number of features.")

    if cov is None:
        cov = generate_spsd_matrix(n_features, random_seed=random_seed)
    else:
        if (cov.shape[0] != n_features) or (cov.shape[1] != n_features):
            raise ValueError("Shape mismatch between covariance and number of features.")

    x = np.random.multivariate_normal(mu, cov, n_samples)
    return (x, mu, cov)


dimension = 2
# mean = np.random.rand(dimension)
mean = np.zeros(dimension)
numberOfSamples = 1000
covarianceMatrix = np.matrix([[1, 0.3], [0.3, 2]])
Y, x = GenerateGaussianSamples(mean, covarianceMatrix, numberOfSamples)
print(mean)

# make a historgram
N = np.random.multivariate_normal(size=numberOfSamples, mean=mean, cov=covarianceMatrix)
K, mu, c = generate_gaussian_data(numberOfSamples, dimension, mean, covarianceMatrix)

fig, ax = plt.subplots(dimension + 1, 3, figsize=(10, 10))
for d in range(0, dimension):
    sns.distplot(Y[:, d], ax=ax[d][0], kde=True, color='orange')
    sns.distplot(N[:, d], ax=ax[d][1], kde=True, color='red')
    sns.distplot(K[:, d], ax=ax[d][2], kde=True, color='green')

sns.distplot(np.average(Y, axis=1), ax=ax[dimension][0], kde=True, color='orange')
sns.distplot(np.average(N, axis=1), ax=ax[dimension][1], kde=True, color='red')
sns.distplot(np.average(K, axis=1), ax=ax[dimension][2], kde=True, color='green')

empiricalCovariance = np.cov(Y, rowvar=False)
print("Initial covariance matrix is")
print(covarianceMatrix)
print("Emprirical Covariance from sampled data is")
print(empiricalCovariance)
plt.show()

# Build the data frame
y = np.array(Y[:,0]).flatten()
k = np.array(K[:,0]).flatten()


p = pd.DataFrame(columns=['Y','K'])
p['Y'] = pd.Series(y)
p['K'] = pd.Series(k)

sns.jointplot(x='Y',y='K',data=p,kind="reg")