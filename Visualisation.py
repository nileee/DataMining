from __future__ import division, print_function  # Imports from __future__ since we're running Python 2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

labels_path = os.path.join(os.getcwd(), 'datasets', 'landsat', 'landsat_classes.csv')
landsat_labels = pd.read_csv(labels_path, delimiter=',', index_col=0)
landsat_labels
dict = landsat_labels.to_dict()["Class"]


def ApplyLabelNames(frameSeries):
    labelValue = frameSeries.loc['label']
    labelName = dict.get(labelValue)
    return labelName


train_path = os.path.join(os.getcwd(), 'datasets', 'landsat', 'landsat_train.csv')
test_path = os.path.join(os.getcwd(), 'datasets', 'landsat', 'landsat_test.csv')
landsat_train = pd.read_csv(train_path, delimiter=',')
landsat_test = pd.read_csv(test_path, delimiter=',')

landsat_train.head(n=5)

# Inspect 7 random entries in the test dataset. Set the random_state parameter to a number of your choice (i.e. 10) to ensure reproducible results.

state = np.random.RandomState(1234)
sampledData = landsat_train.sample(n=7, random_state=state)
sampledData['DummyColoumn'] = 'some name'
# sampledData['DummyColoumn'] = sampledData.iloc[:,[8,10,15]].apply(ChangeDummyColoumn, axis=0)

keyColoumns = 'label'

trainingAggregates = pd.DataFrame({'LabelCount': landsat_train.groupby([keyColoumns]).size()}).reset_index();
trainingAggregates['LabelName'] = trainingAggregates.apply(ApplyLabelNames, axis=1)

pixels = np.arange(1, 10)  # Pixel values (1-9)
bands = np.arange(1, 5)  # Spectral band values (1-4)
labels = np.sort(
    landsat_train.label.unique())  # Get the labels in the dataset, by looking at possible values of "label" attribute
'''
ax = trainingAggregates[['LabelCount']].plot(kind='bar', title ="Pixel Types", legend=True, fontsize=12)
ax.set_xlabel("Label", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
labels = trainingAggregates.LabelName.values[:]
ax = plt.gca()
ax.set_xticklabels(labels, rotation=0)
#plt.show()

# QUESTION 2
#The pandas info() method provides basic information (i.e. number of rows/columns, variable data types) about a DataFrame.

#Display the basic information about the landsat_train dataframe. How many attributes/samples are there in the dataset?

print(landsat_train.info())


fig, ax = plt.subplots(9,4, figsize=(17,17)) # Figure with 9 rows and 4 columns

for ii, pixel in enumerate(pixels):
    for jj, band in enumerate(bands):
        variable_name = 'pixel_' + str(pixel) + '_' + str(band) # Get the variable name of interest
        variableToDist = landsat_train[variable_name]
        sns.distplot(variableToDist, ax=ax[ii][jj], kde=True) # Use a single feature at a time
        ax[ii][jj].xaxis.label.set_visible(False)
[ax[0][ii].set_title("Band {}".format(band)) for ii, band in enumerate(bands)] # Set band titles for top plots
[ax[ii][0].set_ylabel("Pixel {}".format(pixel)) for ii, pixel in enumerate(pixels)] # Set pixel titles for left-most plots
fig.tight_layout()
#plt.show()


fig, ax = plt.subplots(1,4, figsize=(17,3))
for ii, band in enumerate(bands):
    variable_names = ['pixel_' + str(pixel) + '_' + str(band) for pixel in pixels] # All pixels for the specified band
    sns.distplot(landsat_train[variable_names].values.reshape(-1,), ax=ax[ii], kde=True, bins=25) # Reshape into 1D array
    ax[ii].set_title('Band {}'.format(band)) # Subplot titles
ax[0].set_ylabel('Pooled pixels') # ylabel for left-most subplot
fig.tight_layout()
#plt.show()


labels = np.sort(landsat_train.label.unique()) # Get the labels in the dataset, by looking at possible values of "label" attribute
fig, ax = plt.subplots(labels.size,4, figsize=(17,14))
for ii, label in enumerate(labels):
    for jj, band in enumerate(bands):
        variable_names = ['pixel_' + str(pixel) + '_' + str(band) for pixel in pixels] # Pool pixels together
        sns.distplot(landsat_train[landsat_train["label"]==label][variable_names].values.reshape(-1,), ax=ax[ii][jj], bins=25) # Filter by label
[ax[0][ii].set_title("Band {}".format(band)) for ii, band in enumerate(bands)] # Set band titles on top plots
[ax[ii][0].set_ylabel("{}".format(dict[label])) for ii, label in enumerate(labels)] # Set label titles in left-most plots
fig.tight_layout()




Produce a figure with four subplots (one for each spectral band) and within each subplot use the kdeplot() function
to show the kernel density estimate for the pooled pixel intensity values, separately for each class.
Pay special attention in setting the legend(s) in your figure correctly.




for i, band in enumerate(bands):
    for j,pixel in enumerate(pixels):
        for li, label in enumerate(labels):
            variableName = ['pixel_' + str(pixel) + '_' + str(band) for pixel in pixels]
            variableData = landsat_train[landsat_train["label"]==label][variableName]
            sns.kdeplot(variableData, ax=[0][i],shade=True)

plt.show()


'''
'''




'''
colour = ['green', 'red', 'orange', 'black', 'blue', 'gray']

'''
fig, ax = plt.subplots(2,2, figsize=(17,14))
d1 = np.array([1,1,1,1,1,1,22,2,2,2,2,2,2,2,2,2,2,33,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,9,9,10])
d2 = np.array([13,13,13,13,1,1,2,2,2,2,12,12,12,12,12,12,2,13,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,9,9,10])
#d1 = d1.astype(int)
#d2 = d2.astype(int)

d3 = np.array([1,1,1,2,2,2,2,2,2,2,2,33,3,4,4,4,4,4,4,4,4,4,4,4,4,4,5,6,6,6,6,7,7,7,7,8,8,8,9,9,10])
d4 = np.array([13,13,13,13,1,1,2,2,2,12,12,12,12,2,13,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,6,6,6,6,8,9,9,10])

sns.kdeplot(d1,ax=ax[0][0],color=colour[5])
sns.kdeplot(d3,ax=ax[0][1],color = colour[1])
sns.kdeplot(d4,ax=ax[1][0],color=colour[2])
sns.kdeplot(d2,ax=ax[1][1], color = colour[3])
plt.show()

fig, ax = plt.subplots(1,bands.size, figsize=(17,4))

for b,band in enumerate(bands):
    pixelVariables = ['pixel_' + str(pixel) + '_' + str(band) for pixel in pixels]
    for l,label in enumerate(labels):
        trainingDataLabelIntesity = landsat_train[landsat_train['label']==label][pixelVariables].values.reshape(-1)
        sns.kdeplot(trainingDataLabelIntesity,ax=ax[b],label=dict[label])
[ax[b].set_title("Band {}".format(band)) for b, band in enumerate(bands)] # Set band titles
ax[0].legend(loc='upper center', bbox_to_anchor=(1.6, 1.25),ncol=6) # Put legend outside plot
[ax[ii].legend_.remove() for ii in np.arange(1,4)] # Remove all legends except the first one
plt.show()
'''
sLength = len(landsat_train['label'])
landsat_train['LabelName'] = pd.Series(np.random.randn(sLength), index=landsat_train.index)
landsat_train['LabelName'] = landsat_train.apply(ApplyLabelNames, axis=1)

landsat_train = landsat_train.assign(LabelName=pd.Series(np.random.randn(sLength)).values)

for band in bands:
    variable_names = ['pixel_' + str(pixel) + '_' + str(band) for pixel in pixels]
    landsat_train['avg_' + str(band)] = landsat_train[variable_names].mean(axis=1)

labelBandAverages = pd.DataFrame(columns={'avg_1', 'avg_2', 'avg_3', 'avg_4', 'label', 'labelname'})
a = [];

for label in labels:
    variableNames = ['avg_1', 'avg_2', 'avg_3', 'avg_4']
    series = landsat_train[landsat_train['label'] == label][variableNames]
    averageSeries = series.mean(axis=0)
    labelName = dict[label]
    a = np.append(a, averageSeries)
    a = np.append(a, int(label))
    a = np.append(a, labelName)

a.resize(len(labels), 6)
p = pd.DataFrame(a, columns=['avg_1', 'avg_2', 'avg_3', 'avg_4', 'label', 'name'])

print(landsat_train.describe())

# Your code goes here
sns.pairplot(landsat_train, vars=["avg_1", "avg_2", "avg_3", "avg_4"], plot_kws={'s': 6},
             hue='LabelName', diag_kind='kde')  # Set variables of interest, marker size and bins for histograms
plt.show()
'''''''''
for band in bands:
    for label in labels:
        pixelVariables = ['pixel_' + str(pixel) + '_' + str(band) for pixel in pixels]
        series = landsat_train[landsat_train['label']==label][pixelVariables]
        series2 = landsat_train[pixelVariables]
        averageValLabels = series.mean(axis=1)
        averageValSansLabels = series2.mean(axis=1)
        print(averageValSansLabels)
        #landsat_train['avg_' + str(band)] = landsat_train[variable_names].mean(axis=1)
landsat_train.head(5) # Show the first 5 observations in the updated dataframe
'''
