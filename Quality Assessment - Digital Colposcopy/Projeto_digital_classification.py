import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
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

def CARTClassifier(trX, trY, tsX, tsY, data):
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
    #graph.render('dtree_render',view=True) 
    
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





#================================== MAIN CODE ================================================

OBJECTIVE = 'consensus'

#Carregamento e Processamento de dados

#================================== GREEN ====================================================
data_set_green = pd.read_csv('green.csv', na_values='na')
X_green = data_set_green.iloc[:,:-1]
Y_green = data_set_green[OBJECTIVE]
trX_green, tsX_green, trY_green, tsY_green = train_test_split(X_green, Y_green, train_size=0.7, stratify=Y_green)
training_data_green = pd.concat([trX_green,trY_green],axis=1)
training_data_green = balancingData(training_data_green,OBJECTIVE)
trX_green = training_data_green.iloc[:,:-1]
trY_green = training_data_green[OBJECTIVE]

print("\n\n\n\===== GREEN RESULTS =====\n\n\n")

#-----K-nearest neighbors (Instance-based Learning)-----
accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = KNNClassifier(trX_green, trY_green, tsX_green, tsY_green)


#-----Naive Bayes-----
accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = GNBClassifier(trX_green, trY_green, tsX_green, tsY_green)


#-----CART (Decision Trees)-----
accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = CARTClassifier(trX_green, trY_green, tsX_green, tsY_green, training_data_green)


#-----Random Forest-----
accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = RFClassifier(trX_green, trY_green, tsX_green, tsY_green)

#================================== HINSELMAN ================================================
data_set_hinselmann = pd.read_csv('hinselmann.csv', na_values='na')
X_hinselmann = data_set_hinselmann.iloc[:,:-1]
Y_hinselmann = data_set_hinselmann[OBJECTIVE]
trX_hinselmann, tsX_hinselmann, trY_hinselmann, tsY_hinselmann = train_test_split(X_hinselmann, Y_hinselmann, train_size=0.7, stratify=Y_hinselmann)
training_data_hinselmann = pd.concat([trX_hinselmann,trY_hinselmann],axis=1)
training_data_hinselmann = balancingData(training_data_hinselmann,OBJECTIVE)
trX_hinselmann = training_data_hinselmann.iloc[:,:-1]
trY_hinselmann = training_data_hinselmann[OBJECTIVE]

print("\n\n\n===== HINSELMANN RESULTS =====\n\n\n")

#-----K-nearest neighbors (Instance-based Learning)-----
accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = KNNClassifier(trX_hinselmann, trY_hinselmann, tsX_hinselmann, tsY_hinselmann)


#-----Naive Bayes-----
accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = GNBClassifier(trX_hinselmann, trY_hinselmann, tsX_hinselmann, tsY_hinselmann)


#-----CART (Decision Trees)-----
accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = CARTClassifier(trX_hinselmann, trY_hinselmann, tsX_hinselmann, tsY_hinselmann, training_data_hinselmann)


#-----Random Forest-----
accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = RFClassifier(trX_hinselmann, trY_hinselmann, tsX_hinselmann, tsY_hinselmann)

#================================== SCHILLER =================================================
data_set_schiller = pd.read_csv('schiller.csv', na_values='na')
X_schiller = data_set_schiller.iloc[:,:-1]
Y_schiller = data_set_schiller[OBJECTIVE]
trX_schiller, tsX_schiller, trY_schiller, tsY_schiller = train_test_split(X_schiller, Y_schiller, train_size=0.7, stratify=Y_schiller)
training_data_schiller = pd.concat([trX_schiller,trY_schiller],axis=1)
training_data_schiller = balancingData(training_data_schiller,OBJECTIVE)
trX_schiller = training_data_schiller.iloc[:,:-1]
trY_schiller = training_data_schiller[OBJECTIVE]

print("\n\n\n===== SCHILLER RESULTS =====\n\n\n")

#-----K-nearest neighbors (Instance-based Learning)-----
accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = KNNClassifier(trX_schiller, trY_schiller, tsX_schiller, tsY_schiller)


#-----Naive Bayes-----
accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = GNBClassifier(trX_schiller, trY_schiller, tsX_schiller, tsY_schiller)


#-----CART (Decision Trees)-----
accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = CARTClassifier(trX_schiller, trY_schiller, tsX_schiller, tsY_schiller, training_data_schiller)


#-----Random Forest-----
accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = RFClassifier(trX_schiller, trY_schiller, tsX_schiller, tsY_schiller)
