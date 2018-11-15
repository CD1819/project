import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

#Funcoes auxiliares
#Data -> Information
def balancingData(data):
    data['classes'] = [0 if line=='neg' else 1 for line in data.classes]
    #class neg is the predominant one
    df_majority = data[data.classes==0]
    df_minority = data[data.classes==1]
    
    number_samples = len(df_majority)
    
    df_minority_resampled = resample(df_minority, replace = True, n_samples=number_samples)
    
    balancedset =  pd.concat([df_majority, df_minority_resampled])
    return balancedset

#Filling missing values
def changeNaNvalues(data, value):
    training = data
    if(value == 0):
        for column in training:
            training[column].fillna(0, inplace=True)
        return training
    elif(value == 'max'):
        for column in training:
            training[column].fillna(training[column].max(), inplace=True)
        return training
    elif(value == 'min'):
        for column in training:
            training[column].fillna(training[column].min(), inplace=True)
        return training
    elif(value == 'mean'):
        for column in training:
            training[column].fillna(training[column].mean(), inplace=True)
        return training
    elif(value == 'interpolate'):
        return training.interpolate(axis=0)

def get_binary_relations(data):
    pairs = {}
    ind = 1
    for index, row in data.iterrows():
        data_listed = list(data.columns.values)
        size = len(data_listed)
        for i in range(size):
            j=i+1
            while j < size:
                aux = (row[data_listed[i]],row[data_listed[j]])
                if(aux in pairs):
                    pairs[aux] +=1
                else:
                    pairs[aux] = 1
                j+=1
        print(ind)
        ind+=1
    return pairs

#================================== MAIN CODE ================================================



#Carregamento e Processamento de dados
aps_failure_test_set = pd.read_csv('aps_failure_test_set_classes.csv', na_values='na')
aps_failure_training_set = pd.read_csv('aps_failure_training_set_classes.csv', na_values='na')

training_set = balancingData(aps_failure_training_set)

aps_failure_test_set['classes'] = aps_failure_test_set['classes'] = [0 if line=='neg' else 1 for line in aps_failure_test_set.classes]

tsY = aps_failure_test_set['classes']
test_set = aps_failure_test_set.loc[:, aps_failure_test_set.columns != 'classes']
trY = training_set['classes']
training_set = training_set.loc[:, aps_failure_training_set.columns != 'classes']

dataset = pd.concat([training_set,test_set])

#tsX = changeNaNvalues(test_set, 0)
#trX1 = changeNaNvalues(dataset, 'min')
#tsX = changeNaNvalues(test_set, 'min')
#trX2 = changeNaNvalues(dataset, 'max')
#tsX = changeNaNvalues(test_set, 'max')
#trX3 = changeNaNvalues(dataset, 'mean')
#tsX = changeNaNvalues(test_set, 'mean')
#trX4 = changeNaNvalues(training_set, 'interpolate')
#tsX = changeNaNvalues(test_set, 'interpolate')

#KMeans
#np.set_printoptions(threshold=np.nan)
kmeans = KMeans(n_clusters=5)
kmeans.fit(trX3)

plt.scatter(trX3[:, 96], trX3[:,70], s = 100, c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 96], kmeans.cluster_centers_[:, 70], s = 300, c = 'red',label = 'Centroids')
plt.title('Aps Clusters and Centroids')
plt.xlabel('ck_000')
plt.ylabel('bj_000')
plt.legend()

plt.show()
#all_predictions = model.predict(trX)
