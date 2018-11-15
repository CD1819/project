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
def balancingData(data,objective):
    var = Counter(data[objective])
    print(var)
    if(var[1] > var[0]):
        df_majority = data[data.consensus==1]
        df_minority = data[data.consensus==0]
    elif(var[0] > var[1]):
        df_majority = data[data.consensus==1]
        df_minority = data[data.consensus==0]
    else:
        return data
    number_samples = len(df_majority)
    df_minority_resampled = resample(df_minority, replace = True, n_samples=number_samples)
    balancedData =  pd.concat([df_majority, df_minority_resampled])
    var = Counter(balancedData[objective])
    print(var)
    return balancedData

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

def accuracy(Confusion_Matrix):
    total = Confusion_Matrix[0][0] + Confusion_Matrix[0][1] + Confusion_Matrix[1][0] + Confusion_Matrix[1][1]
    tp_tn = Confusion_Matrix[0][0] + Confusion_Matrix[1][1]
    return tp_tn/total

def error_rate(Confusion_Matrix):
    total = Confusion_Matrix[0][0] + Confusion_Matrix[0][1] + Confusion_Matrix[1][0] + Confusion_Matrix[1][1]
    fp_fn = Confusion_Matrix[0][1] + Confusion_Matrix[1][0]
    return fp_fn/total

def precision(Confusion_Matrix):
    tp = Confusion_Matrix[0][0]
    tp_fp = tp + Confusion_Matrix[0][1]
    return tp/tp_fp

def t_p_rate(Confusion_Matrix): #Sensitivity or Recall
    tp = Confusion_Matrix[0][0]
    tp_fn = tp + Confusion_Matrix[1][1]
    return tp/tp_fn

def specificity (Confusion_Matrix):
    tn = Confusion_Matrix[1][1]
    tp_fn = tn + Confusion_Matrix[1][0]
    return tn/tp_fn

def f_p_rate (Confusion_Matrix):
    fp = Confusion_Matrix[0][1]
    tn_fp = Confusion_Matrix[1][1] + fp
    return fp/tn_fp

def printRocChart(tsY, pred):
    fpr, tpr, threshold = roc_curve(tsY, pred, pos_label=None)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def printMeasures(Confusion_Matrix):
    accuracy_measure = accuracy(Confusion_Matrix)
    error_rate_measure = error_rate(Confusion_Matrix)
    precision_measure = precision(Confusion_Matrix)
    specificity_measure = specificity(Confusion_Matrix)
    FP_rate_measure = f_p_rate(Confusion_Matrix)
    TP_rate_measure = t_p_rate(Confusion_Matrix)
    print("Confusion matrix:\n", Confusion_Matrix)
    print("Accuracy:", accuracy_measure)
    print("Error rate:", error_rate_measure)
    print("Precision:", precision_measure)
    print("Specifity:", specificity_measure )
    print("FP rate:", FP_rate_measure)
    print("TP rate:", TP_rate_measure, "\n")
    
    return accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure 


#Funcoes de aprendizagem

#-----K-nearest neighbors (Instance-based Learning)-----
def KNNClassifier(trX, trY, tsX, tsY):
    knn = KNeighborsClassifier(n_neighbors=5)

    model_KNN = knn.fit(trX, trY)
    predY_KNN = model_KNN.predict(tsX)
    cnf_matrix_KNN = confusion_matrix(tsY, predY_KNN)

    print("KNN Results:")
    accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = printMeasures(cnf_matrix_KNN)
    
    printRocChart(tsY,predY_KNN)
    
    return accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure

#-----Naive Bayes-----
def GNBClassifier(trX, trY, tsX, tsY):
    clf = GaussianNB()

    model_GNB = clf.fit(trX, trY)
    predY_GNB = model_GNB.predict(tsX)
    cnf_matrix_GNB = confusion_matrix(tsY, predY_GNB)
    
    print("Naive Bayes Results:")
    accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = printMeasures(cnf_matrix_GNB)
    
    printRocChart(tsY,predY_GNB)
    
    return accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure

#-----CART (Decision Trees)-----

def CARTClassifier(trX, trY, tsX, tsY, data, data_name):
    cart = DecisionTreeClassifier()

    model_CART = cart.fit(trX, trY)
    predY_CART = model_CART.predict(tsX)
    cnf_matrix_CART = confusion_matrix(tsY, predY_CART)
    
    print("CART Results:")
    accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = printMeasures(cnf_matrix_CART)
    
    printRocChart(tsY,predY_CART)
    #dot_data = tree.export_graphviz(model_CART, out_file=None,  feature_names=data.axes[1][1:],  
    #class_names=data.axes[1][0],  filled=True, rounded=True, special_characters=True) 
    #graph = graphviz.Source(dot_data)  
    #graph.render(data_name,view=True) 
    
    return accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure

#-----Random Forest-----
def RFClassifier(trX, trY, tsX, tsY):
    rf = RandomForestClassifier()

    model_RF = rf.fit(trX, trY)
    predY_RF = model_RF.predict(tsX)
    cnf_matrix_RF = confusion_matrix(tsY, predY_RF)
    
    print("Random Forest Results:")
    accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = printMeasures(cnf_matrix_RF)
    
    printRocChart(tsY,predY_RF)
    
    return accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure

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

OBJECTIVE = 'consensus'

#Carregamento e Processamento de dados

#================================== GREEN ====================================================
data_set_green = pd.read_csv('green.csv', na_values='na')
data_set_hinselmann = pd.read_csv('hinselmann.csv', na_values='na')
data_set_schiller = pd.read_csv('schiller.csv', na_values='na')

#binary_relations = get_binary_relations(data_set_green)

res = list()
n_cluster = range(2,20)
for n in n_cluster:
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(data_set_green)
    res.append(np.average(np.min(cdist(data_set_green, kmeans.cluster_centers_, 'euclidean'), axis=1)))

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