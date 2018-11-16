import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

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

#================================== GREEN ====================================================
data_set_green = pd.read_csv('green.csv', na_values='na')
data_set_hinselmann = pd.read_csv('hinselmann.csv', na_values='na')
data_set_schiller = pd.read_csv('schiller.csv', na_values='na')

X_gree = data_set_green.iloc[:,:-1]
X_hinselmann = data_set_hinselmann.iloc[:,:-1]
X_schiller = data_set_schiller.iloc[:,:-1]

#binary_relations = get_binary_relations(data_set_green)

kmeans = KMeans(n_clusters=5)
plt.scatter(X_schiller['walls_artifacts_area'], X_schiller['area_h_max_diff'], s = 100, c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 5], kmeans.cluster_centers_[:, 14], s = 300, c = 'red',label = 'Centroids')
plt.title('Schiller Clusters and Centroids')
plt.xlabel('walls_artifacts_area')
plt.ylabel('area_h_max_diff')
plt.legend()

res1 = list()
n_cluster = range(2,20)
for n in n_cluster:
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(data_set_green)
    res1.append(np.average(np.min(cdist(data_set_green, kmeans.cluster_centers_, 'euclidean'), axis=1)))

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
plt.plot(n_cluster, res1,'g', n_cluster, res2, 'r', n_cluster, res3, 'b')
plt.title('elbow curve')
plt.legend(handles=[green_patch, red_patch, blue_patch])
plt.show()

#KMeans
#kmeans.fit(data_set_green)
#all_predictions = model.predict(data_set_green)
#print(all_predictions)

#kmeans.fit(X_schiller)
#kmeans.cluster_centers_
#distance = kmeans.fit_transform(X_schiller)
#labels = kmeans.labels_

#kmeans.fit(X_schiller)
#kmeans.cluster_centers_
#distance = kmeans.fit_transform(X_schiller)
#labels = kmeans.labels_


# # AgglomerativeClustering
# dendrogram = sch.dendrogram(sch.linkage(tsX, method='ward'))
# hc = AgglomerativeClustering(n_clusters=170, affinity = 'euclidean', linkage = 'ward')
# y_hc = hc.fit_predict(tsX)
# print(y_hc)
# 
