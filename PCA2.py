from __future__ import print_function, division
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, explained_variance_score

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



x_3d, mu_true, C_true = generate_gaussian_data(n_samples=1000, n_features=3, random_seed=10)
print("Dataset consists of {} samples and {} features.".format(x_3d.shape[0], x_3d.shape[1]))
print("True mean:\n{}".format(mu_true))
print("True covariance matrix:\n{}".format(C_true))


empMean = np.mean(x_3d,axis=0)
empCov = np.cov(x_3d,rowvar=False)
print("Empirical mean:\n{}".format(empMean))
print("Empirical covariance matrix:\n{}".format(empCov))

print("Estimated correlation matrix:\n{}".format(np.corrcoef(x_3d, rowvar=False)))

g = sns.jointplot(x_3d[:,0],x_3d[:,1], size=4,  ratio=5, joint_kws={"s" : 1}, color='red')
g.set_axis_labels("Dim 1", "Dim 2")
plt.show()

#Initialise a pca object and "fit" it using the dataset x_3d. Print the three PC directions. Store the PC scores for x_3d in an array called pc_scores.

#Hint: according to the documentation the components are stored row-wise, i.e. the components_ array has shape [n_components, n_features].
# You should print the PC directions column-wise, i.e. take the transpose of the components_ array.

'''
Use any implementation of PCA of your choice to project the dataset x_3d onto 1, 2, and 3 dimensions.
This projection should just take the form

X red =XW 1:k
Xred=XW1:k
where X
X
 is the centered data matrix and W
W
 is the matrix containing the PC directions as columns. The approximation of the data by the k
k
 principle components is
X recon =X red W T 1:k +μ
Xrecon=XredW1:kT+μ
.

Plot the mean-squared-error and explained variance of the approximation as a function of the number of components used.
Label axes appropriately. You should make use of the sklearn mean_squared_error() and explained_variance_score() metrics.
For explained_variance_score(), you should set the multioutput parameter to variance_weighted.
'''

centeredData = x_3d - empMean
centeredCov = np.cov(centeredData, rowvar=False)
#v,s,u  = np.linalg.svd(a=centeredData,full_matrices=1,compute_uv=1)

pca3 = PCA()
pca3.fit(centeredData)
W3 = pca3.components_
#x3Reduced = np.dot(centeredData,W3)
#x3Reconstructed = np.dot(x3Reduced,W3.T)+empMean
print("PCA Components for 3 dimensions {0}:".format(W3))
print("Explained variance is : {0}".format(pca3.explained_variance_))
print("Explained variance ratio is : {0}".format(pca3.explained_variance_ratio_))


ev = []
mse = []
num_components = range(1,4)

for k in num_components:
    x_red = np.dot(centeredData, W3[:,:k])
    x_recon = np.dot(x_red, W3[:,:k].T) + empMean
    mse.append(mean_squared_error(x_3d, x_recon))
    ev.append(explained_variance_score(x_3d, x_recon, multioutput='variance_weighted'))

pca2 = PCA()
pca2.fit(centeredData)
W2 = pca2.components_
#x3Reduced = np.dot(centeredData,W3)
#x3Reconstructed = np.dot(x3Reduced,W3.T)+empMean
print("PCA Components for 3 dimensions {0}:".format(W2))
print("Explained variance is : {0}".format(pca2.explained_variance_))
print("Explained variance ratio is : {0}".format(pca2.explained_variance_ratio_))


ev2 = []
mse2 = []
num_components2 = range(1,3)

# some non whitespace changes

for k in num_components2:
    x_red = np.dot(centeredData, W2[:,:k])
    x_recon = np.dot(x_red, W3[:,:k].T) + empMean
    mse2.append(mean_squared_error(x_3d, x_recon))
    ev2.append(explained_variance_score(x_3d, x_recon, multioutput='variance_weighted'))


f, (ax1, ax2) = plt.subplots(2, 2, sharex=True)

ax1[0].scatter(num_components,ev,color='red', s=30)
ax1[0].plot(num_components, ev, linestyle='--', color='red')
ax1[0].set_ylabel('explained variance')


ax1[1].scatter(num_components2,ev2,color='blue', s=30)
ax1[1].plot(num_components2, ev2, linestyle='--', color='blue')


ax2[0].scatter(num_components,mse,color='red', s=30)
ax2[0].plot(num_components, mse, linestyle='--', color='red')
ax2[0].set_ylabel('mean squared error')


ax2[1].scatter(num_components2,mse2,color='blue', s=30)
ax2[1].plot(num_components2, mse2, linestyle='--', color='blue')

f.text(0.5, 0.04, 'Number of components', ha='center')

ax1[0].set_xticks(num_components)
ax1[1].set_xticks(num_components2)
ax2[0].set_xticks(num_components)
ax2[1].set_xticks(num_components2)

plt.show()

