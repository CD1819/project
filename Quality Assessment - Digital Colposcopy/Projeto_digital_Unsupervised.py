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
import matplotlib.patches as mpatches

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


<<<<<<< HEAD
#Carregamento e Processamento de dados

#================================== GREEN ====================================================
data_set_green = pd.read_csv('green.csv', na_values='na')
data_set_hinselmann = pd.read_csv('hinselmann.csv', na_values='na')
data_set_schiller = pd.read_csv('schiller.csv', na_values='na')

#binary_relations = get_binary_relations(data_set_green)
=======
>>>>>>> 2bc969e7cb1d85f041e6affc3eb8af225bb1d90e

#Carregamento e Processamento de dados
data_set_schiller = pd.read_csv('schiller.csv', na_values='na')
X_schiller = data_set_schiller.iloc[:,:-1]

#KMeans
#np.set_printoptions(threshold=np.nan)
kmeans = KMeans(n_clusters=5)
kmeans.fit(X_schiller)
kmeans.cluster_centers_
distance = kmeans.fit_transform(X_schiller)
labels = kmeans.labels_

plt.scatter(X_schiller['walls_artifacts_area'], X_schiller['area_h_max_diff'], s = 100, c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 5], kmeans.cluster_centers_[:, 14], s = 300, c = 'red',label = 'Centroids')
plt.title('Schiller Clusters and Centroids')
plt.xlabel('walls_artifacts_area')
plt.ylabel('area_h_max_diff')
plt.legend()

<<<<<<< HEAD
res2 = list()
n_cluster = range(2,20)
for n in n_cluster:
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(data_set_hinselmann)
    res2.append(np.average(np.min(cdist(data_set_hinselmann, kmeans.cluster_centers_, 'euclidean'), axis=1)))

res3 = list()
n_cluster = range(2,20)
for n in n_cluster:
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(data_set_schiller)
    res3.append(np.average(np.min(cdist(data_set_schiller, kmeans.cluster_centers_, 'euclidean'), axis=1)))

green_patch = mpatches.Patch(color='green', label ='Green')
red_patch = mpatches.Patch(color='red', label ='Hinselmann')
blue_patch = mpatches.Patch(color='blue', label ='Schiller')
plt.plot(n_cluster, res,'g', n_cluster, res2, 'r', n_cluster, res3, 'b')
plt.title('elbow curve')
plt.legend(handles=[green_patch, red_patch, blue_patch])
plt.show()

 #KMeans
#np.set_printoptions(threshold=np.nan)
#model = KMeans(n_clusters=5)
#model.fit(data_set_green)
#all_predictions = model.predict(data_set_green)
#print(all_predictions)
 
# # AgglomerativeClustering
# dendrogram = sch.dendrogram(sch.linkage(tsX, method='ward'))
# hc = AgglomerativeClustering(n_clusters=170, affinity = 'euclidean', linkage = 'ward')
# y_hc = hc.fit_predict(tsX)
# print(y_hc)
# 
=======
plt.show()
>>>>>>> 2bc969e7cb1d85f041e6affc3eb8af225bb1d90e
