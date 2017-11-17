# Import required packages
from __future__ import division, print_function # Imports from __future__ since we're running Python 2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
random_state = 10 # Ensure reproducible results
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

labels_path = os.path.join(os.getcwd(), 'datasets', 'landsat', 'landsat_classes.csv')
landsat_labels = pd.read_csv(labels_path, delimiter=',', index_col=0)
landsat_labels_dict = landsat_labels.to_dict()["Class"]



# Plot confusion matrix by using seaborn heatmap function
def plot_confusion_matrix(cm, normalize=False, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix.

    If normalize is set to True, the rows of the confusion matrix are normalized so that they sum up to 1.

    """
    if normalize is True:
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
        vmin, vmax = 0., 1.
        fmt = '.2f'
    else:
        vmin, vmax = None, None
        fmt = 'd'
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=vmin, vmax=vmax,
                    annot=True, annot_kws={"fontsize": 9}, fmt=fmt)
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def ApplyLabelNames(frameSeries):
    labelValue = frameSeries.loc['label']
    labelName = landsat_labels_dict.get(labelValue)
    return labelName


train_path = os.path.join(os.getcwd(), 'datasets', 'landsat', 'landsat_train.csv')
test_path = os.path.join(os.getcwd(), 'datasets', 'landsat', 'landsat_test.csv')
landsat_train_full = pd.read_csv(train_path, delimiter=',')
landsat_test = pd.read_csv(test_path, delimiter=',')

sLength = len(landsat_train_full['label'])
landsat_train_full['LabelName'] = pd.Series(np.random.randn(sLength), index=landsat_train_full.index)
landsat_train_full['LabelName'] = landsat_train_full.apply(ApplyLabelNames, axis=1)

sLength = len(landsat_test['label'])
landsat_test['LabelName'] = pd.Series(np.random.randn(sLength), index=landsat_test.index)
landsat_test['LabelName'] = landsat_test.apply(ApplyLabelNames, axis=1)

uniqueLabels = landsat_train_full.label.unique()

xtrainfull = landsat_train_full.iloc[:,0:36].values
print("Shape of xtrainfull is :{0}".format(xtrainfull.shape))
ytrainfull = landsat_train_full['LabelName']
print("Shape of ytrainfull is :{0}".format(ytrainfull.shape))

xtest = landsat_test.iloc[:,0:36].values
print("Shape of xtest is :{0}".format(xtest.shape))
ytest = ytrain = landsat_test['LabelName']
print("Shape of ytest is :{0}".format(ytest.shape))


xtrain, xvalidation, ytrain, yvalidation = train_test_split(xtrainfull, ytrainfull, test_size=0.33, random_state=random_state)


print("Shape of xtrain is :{0}".format(xtrain.shape))
print("Shape of xvalidation is :{0}".format(xvalidation.shape))
print("Shape of ytrain is :{0}".format(ytrain.shape))
print("Shape of yvalidation is :{0}".format(yvalidation.shape))


#Standardise the training, validation and testing set
# testing set is not the same as validation. testing set is used to guage the generalisability of the model.
# validation is to make sure that there is no over fitting of the model
# best technique is to use some sort of cross validation.

standardScalar = StandardScaler()
xTrainStandard = standardScalar.fit_transform(xtrain)
xValidationStandard = standardScalar.fit_transform(xvalidation)
xTestStandard = standardScalar.fit_transform(xtest)


# USE GAUSSIAN NAIVE BAYES AS CLASSIFICATION

gnb = GaussianNB().fit(xTrainStandard,ytrain)

# Check the classification accuracy

# By using the predict() method and accuracy_score metric
gnb_prediction = gnb.predict(xValidationStandard)
gnb_accuracy = accuracy_score(yvalidation, gnb_prediction) # The accuracy_score() function takes as inputs
                                                 # the true labels and the predicted ones

# By using the score() method
gnb_accuracy_alt = gnb.score(xValidationStandard, yvalidation) # The score() method takes as inputs
                                              # the test input features and the associated (true) labels

# Print results
print("GNB classification accuracy on validation set (by using the accuracy_score() function): {:.3f}"
      .format(gnb_accuracy))
print("GNB classification accuracy on validation set (by using the model's score() method): {:.3f}"
      .format(gnb_accuracy_alt))

yValidationPrediction = gnb_prediction
confusionMatrixValidation = confusion_matrix(yvalidation,yValidationPrediction)

plot_confusion_matrix(confusionMatrixValidation,classes=gnb.classes_)


yTestPrediction = gnb.predict(xTestStandard)
gnbTestAccuracy = accuracy_score(ytest, yTestPrediction) # The accuracy_score() function takes as inputs
                                                 # the true labels and the predicted ones

# By using the score() method
gnbTestAccuracyScoreMethod = gnb.score(xTestStandard, ytest) # The score() method takes as inputs
                                              # the test input features and the associated (true) labels





confusionMatrixTest = confusion_matrix(ytest,yTestPrediction)


fig, ax = plt.subplots(1,2)
plt.subplot(1,2,1)
sns.heatmap(confusionMatrixTest, vmin=0., vmax=1.)
plt.title("Unnormalised Test Confusion Matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.subplot(1,2,2)
confusionMatrixTestNorm = confusionMatrixTest / confusionMatrixTest.sum(axis=1)[:, np.newaxis]
sns.heatmap(confusionMatrixTestNorm, vmin=0., vmax=1.)
plt.title("Normalised Test Confusion Matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# This is the code in the solution
# Notice how it shows everything and is perfectly positioned. This was the reason for you not getting A marks.
# Your code goes here
fig = plt.figure(figsize=(9,5))
ax1 = fig.add_subplot(121)
plot_confusion_matrix(confusionMatrixTest, normalize=False, classes=gnb.classes_) # un-normalized
ax2 = fig.add_subplot(122)
plot_confusion_matrix(confusionMatrixTest, normalize=True, classes=gnb.classes_, title='Normalised confusion matrix') # normalized
ax2.get_yaxis().set_visible(False)
fig.tight_layout()
plt.show()

# log-loss
# in order to calculate log loss, we have to recalculate predictions based on log-likelihood

