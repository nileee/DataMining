from __future__ import division, print_function  # Imports from __future__ since we're running Python 2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # Import the PCA module
import matplotlib.lines as mlines
from sklearn.decomposition import KernelPCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE

def scatter_3d_label(X_3d, y, fig=None, s=2, alpha=0.5, lw=2):
    """Visualuse a 3D embedding with corresponding labels.

    X_3d : ndarray, shape (n_samples,3)
        Low-dimensional feature representation.

    y : ndarray, shape (n_samples,)
        Labels corresponding to the entries in X_3d.

    s : float
        Marker size for scatter plot.

    alpha : float
        Transparency for scatter plot.

    lw : float
        Linewidth for scatter plot.
    """
    from mpl_toolkits.mplot3d import Axes3D
    if fig is None:
        fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    targets = np.unique(y)
    colors = sns.color_palette(n_colors=targets.size)
    for color, target in zip(colors, targets):
        ax.scatter(X_3d[y == target, 0], X_3d[y == target, 1], X_3d[y == target, 2],
                   color=color, s=s, label=target, alpha=alpha, lw=lw)
    return (fig, ax)

def scatter_2d_label(X_2d, y, s=2, alpha=0.5, lw=2):
    """Visualuse a 2D embedding with corresponding labels.

    X_2d : ndarray, shape (n_samples,2)
        Low-dimensional feature representation.

    y : ndarray, shape (n_samples,)
        Labels corresponding to the entries in X_2d.

    s : float
        Marker size for scatter plot.

    alpha : float
        Transparency for scatter plot.

    lw : float
        Linewidth for scatter plot.
    """
    targets = np.unique(y)
    colors = sns.color_palette(n_colors=targets.size)
    for color, target in zip(colors, targets):
        plt.scatter(X_2d[y == target, 0], X_2d[y == target, 1], color=color, label=target, s=s, alpha=alpha, lw=lw)


# Your code goes here
def kde_2d_label(X_2d, y):
    """Kernel density estimate in a 2D embedding with corresponding labels.

    X_2d : ndarray, shape (n_samples,2)
        Low-dimensional feature representation.

    y : ndarray, shape (n_samples,)
        Labels correspodning to the entries in X_2d.
    """
    targets = np.unique(y)
    colors = sns.color_palette(n_colors=targets.size)
    for color, target in zip(colors, targets):
        sns.kdeplot(X_2d[y == target, 0], X_2d[y == target, 1], cmap=sns.dark_palette(color, as_cmap=True))



plt.style.use('ggplot')

labels_path = os.path.join(os.getcwd(), 'datasets', 'landsat', 'landsat_classes.csv')
landsat_labels = pd.read_csv(labels_path, delimiter=',', index_col=0)
landsat_labels
landsat_labels_dict = landsat_labels.to_dict()["Class"]

train_path = os.path.join(os.getcwd(), 'datasets', 'landsat', 'landsat_train.csv')
test_path = os.path.join(os.getcwd(), 'datasets', 'landsat', 'landsat_test.csv')
landsat_train = pd.read_csv(train_path, delimiter=',')
landsat_test = pd.read_csv(test_path, delimiter=',')

# Replace label numbers with their names
landsat_train.replace({'label' : landsat_labels_dict}, inplace=True)
landsat_train.sample(n=5, random_state=10)

x = landsat_train.drop('label', axis=1).values # Input features
y = landsat_train['label'].values # Labels
print('Dimensionality of X: {}\nDimensionality of y: {}'.format(x.shape, y.shape))




pixelMeans = []

for p in range(0,36):
    m = np.average(x[:,p], axis=0)
    sd = np.std(x[:,p], axis=0)
    t = (m,sd)
    pixelMeans.append(t)

for i in range(0,4):
    print("Mean for Pixel1_{0} is : {1}".format(i,pixelMeans[i][0]))
    print("SD for Pixel1_{0} is : {1}".format(i, pixelMeans[i][1]))


#  Data Standardisation
stan = StandardScaler()
stan.fit(x)
xStandard = stan.transform(x)

pixelMeansStandard = []

for p in range(0,36):
    m = np.average(xStandard[:,p], axis=0)
    sd = np.std(xStandard[:,p], axis=0)
    t = (m,sd)
    pixelMeansStandard.append(t)

for i in range(0,4):
    print("Mean for Pixel1_{0} is : {1}".format(i,pixelMeansStandard[i][0]))
    print("SD for Pixel1_{0} is : {1}".format(i, pixelMeansStandard[i][1]))


# In order to see random projections, just multiply the data with a weight vector of lower dimensions

np.random.seed(seed=20)
randomWeight = np.random.randn(xStandard.shape[1],2)
randomProjection = np.dot(xStandard,randomWeight)

plt.figure(figsize=(8,5)) # Initialise a figure instance with defined size
scatter_2d_label(randomProjection, y)
plt.legend(loc='center left', bbox_to_anchor=[1.01, 0.5], scatterpoints=3) # Add a legend outside the plot at specified point
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

X_pca_2d = PCA(n_components=2).fit_transform(xStandard) # Initialise a PCA instance, fit it by using X_sc and then transform X_sc
plt.figure(figsize=(8,5))
scatter_2d_label(X_pca_2d, y)
plt.title('Labelled data in 2-D PCA space')
plt.xlabel('Principal component score 1')
plt.ylabel('Principal component score 2')
plt.legend(loc='best', scatterpoints=3) # Ask matplotlib to place the legend where it thinks best
plt.show()

cats = np.unique(y)
color_palette = sns.color_palette(n_colors=cats.size)
patches = [mlines.Line2D([], [], color=color_palette[ii], label=cat) for ii, cat in enumerate(cats)]

plt.figure(figsize=(8,5))
kde_2d_label(X_pca_2d, y)
plt.title('Kernel density estimation in PCA space')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.legend(patches, cats, loc='center left', scatterpoints=3, bbox_to_anchor=[1.01, 0.5]) # Use the custom legend entries
plt.show()

targets = np.unique(y)
colors = sns.color_palette(n_colors=targets.size)
plt.figure(figsize=(25,20))
for color, target in zip(colors, targets):
    sns.kdeplot(X_pca_2d[y == target, 0], X_pca_2d[y == target, 1], cmap=sns.dark_palette
                (color, as_cmap=True),shade=True, shade_lowest=False)
plt.title('Kernel density estimation in PCA space')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.legend(patches, cats, loc='center left', scatterpoints=3, bbox_to_anchor=[1.01, 0.5]) # Use the custom legend entries
plt.show()

targets = np.unique(y)
colors = sns.color_palette(n_colors=targets.size)
plt.figure(figsize=(25,20))
for color, target in zip(colors, targets):
    sns.kdeplot(randomProjection[y == target, 0], randomProjection[y == target, 1], cmap=sns.dark_palette
                (color, as_cmap=True),shade=True, shade_lowest=False)
plt.title('Kernel density estimation for Random Projections')
plt.xlabel('Random Projection Dimension 1')
plt.ylabel('Random Projection Dimension 2')
plt.legend(patches, cats, loc='center left', scatterpoints=3, bbox_to_anchor=[1.01, 0.5]) # Use the custom legend entries
plt.show()

kernelpoly = KernelPCA(n_components=2, kernel="poly", degree=2)
kernelpoly.fit(xStandard)
xTranformedPoly = kernelpoly.transform(xStandard)
print(kernelpoly.n_components)

targets = np.unique(y)
colors = sns.color_palette(n_colors=targets.size)
plt.figure(figsize=(15,10))
for color, target in zip(colors, targets):
    sns.kdeplot(xTranformedPoly[y == target, 0], xTranformedPoly[y == target, 1], cmap=sns.dark_palette
                (color, as_cmap=True),shade=True, shade_lowest=False)
plt.title('Kernel density estimation for Random Projections')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend(patches, cats, loc='center left', scatterpoints=3, bbox_to_anchor=[1.01, 0.5]) # Use the custom legend entries
plt.show()

kernelpoly = KernelPCA(n_components=2, kernel="rbf")
kernelpoly.fit(xStandard)
xTranformedPoly = kernelpoly.transform(x)
print(kernelpoly.n_components)

targets = np.unique(y)
colors = sns.color_palette(n_colors=targets.size)
plt.figure(figsize=(15,10))
for color, target in zip(colors, targets):
    sns.kdeplot(xTranformedPoly[y == target, 0], xTranformedPoly[y == target, 1], cmap=sns.dark_palette
                (color, as_cmap=True),shade=True, shade_lowest=False)
plt.title('Kernel density estimation for Random Projections')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend(patches, cats, loc='center left', scatterpoints=3, bbox_to_anchor=[1.01, 0.5]) # Use the custom legend entries
plt.show()

mds_3_comp = MDS(n_components=3, metric=True, n_init=1, max_iter=100, random_state=10)
X_mds_3d = mds_3_comp.fit_transform(xStandard)

fig = plt.figure(figsize=(8,6))
_, ax = scatter_3d_label(X_mds_3d, y, fig=fig)
ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')
ax.set_zlabel('Dim 3')
ax.set_title('Labelled data with MDS (3 components)')
ax.legend(loc='center left', bbox_to_anchor=[1.01, 0.5], scatterpoints=3)
ax.azim = 400
ax.elev = 5
fig.tight_layout()
plt.show()

sns.set(font_scale=1.5) # Set default font size
fig, ax = plt.subplots(3,2,figsize=(12,14))
for ii, perplexity in enumerate([150,200,220,250,280,300]):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=10)
    X_tsne_2d = tsne.fit_transform(xStandard)
    plt.subplot(3,2,ii+1)
    scatter_2d_label(X_tsne_2d, y)
    plt.title('Perplexity: {}, KL-score: {}'.format(perplexity, tsne.kl_divergence_))
    plt.xlabel('Component 1')
    plt.ylabel('Component 2 ')
plt.legend(loc='center left', bbox_to_anchor=[1.01, 1.5], scatterpoints=3)
fig.tight_layout()
plt.show()