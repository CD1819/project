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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample

#Funcoes auxiliares
#Data -> Information | Group -> Class | name -> Name of the most proiminent type of the class
def balancingData(data):
    data['classes'] = [0 if line=='neg' else 1 for line in data.classes]
    #class neg is the predominant one
    df_majority = data[data.classes==1]
    df_minority = data[data.classes==0]
    
    number_samples = len(df_majority)
    
    df_minority_resampled = resample(df_minority, replace = True, n_samples=number_samples)
    
    balancedset =  pd.concat([df_majority, df_minority_resampled])
    return balancedset
    
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
def KNNClassifier():
    knn = KNeighborsClassifier(n_neighbors=5)

    model_KNN = knn.fit(trX, trY)
    predY_KNN = model_KNN.predict(tsX)
    cnf_matrix_KNN = confusion_matrix(tsY, predY_KNN)

    print("KNN Results:")
    accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = printMeasures(cnf_matrix_KNN)
    
    printRocChart(tsY,predY_KNN)
    
    return accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure

#-----Naive Bayes-----
def GNBClassifier():
    clf = GaussianNB()

    model_GNB = clf.fit(trX, trY)
    predY_GNB = model_GNB.predict(tsX)
    cnf_matrix_GNB = confusion_matrix(tsY, predY_GNB)
    
    print("Naive Bayes Results:")
    accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = printMeasures(cnf_matrix_GNB)
    
    printRocChart(tsY,predY_GNB)
    
    return accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure

#-----CART (Decision Trees)-----

def CARTClassifier():
    cart = DecisionTreeClassifier()

    model_CART = cart.fit(trX, trY)
    predY_CART = model_CART.predict(tsX)
    cnf_matrix_CART = confusion_matrix(tsY, predY_CART)
    
    print("CART Results:")
    accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = printMeasures(cnf_matrix_CART)
    
    printRocChart(tsY,predY_CART)
    
    return accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure

#-----Random Forest-----
    
def RFClassifier():
    rf = RandomForestClassifier()

    model_RF = rf.fit(trX, trY)
    predY_RF = model_RF.predict(tsX)
    cnf_matrix_RF = confusion_matrix(tsY, predY_RF)
    
    print("Random Forest Results:")
    accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = printMeasures(cnf_matrix_RF)
    
    printRocChart(tsY,predY_RF)
    
    return accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure
    

accuracies = []
error_rates = []
precisions = []
specificities = []
FP_rates = []
TP_rates = []

#Carregamento e Processamento de dados
aps_failure_test_set = pd.read_csv('aps_failure_test_set_classes.csv', na_values='na')
aps_failure_training_set = pd.read_csv('aps_failure_training_set_classes.csv', na_values='na')

#training_set is DataFrame
training_set = balancingData(aps_failure_training_set)

training_set = training_set.fillna(40)
aps_failure_test_set = aps_failure_test_set.fillna(0)
aps_failure_test_set['classes'] = aps_failure_test_set['classes'] = [0 if line=='neg' else 1 for line in aps_failure_test_set.classes]

tsY = aps_failure_test_set['classes']
tsX = aps_failure_test_set.loc[:, aps_failure_test_set.columns != 'classes']
trY = training_set['classes']
trX = training_set.loc[:, aps_failure_training_set.columns != 'classes']


#training_set['ab_000'].replace(0, np.nan)

print
print(aps_failure_test_set.head(30))
#print(training_set.describe())

#Separacao dos grupos de teste e treino
#trX, tsX, trY, tsY = train_test_split(X, Y, train_size=0.7, stratify=Y)

#Funcoes de aprendizagem

#-----K-nearest neighbors (Instance-based Learning)-----
accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = KNNClassifier()
accuracies.append(accuracy_measure)
error_rates.append(error_rate_measure)
precisions.append(precision_measure)
specificities.append(specificity_measure)
FP_rates.append(FP_rate_measure)
TP_rates.append(TP_rate_measure)

#-----Naive Bayes-----
accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = GNBClassifier()
accuracies.append(accuracy_measure)
error_rates.append(error_rate_measure)
precisions.append(precision_measure)
specificities.append(specificity_measure)
FP_rates.append(FP_rate_measure)
TP_rates.append(TP_rate_measure)


#-----CART (Decision Trees)-----
accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = CARTClassifier()
accuracies.append(accuracy_measure)
error_rates.append(error_rate_measure)
precisions.append(precision_measure)
specificities.append(specificity_measure)
FP_rates.append(FP_rate_measure)
TP_rates.append(TP_rate_measure)

#-----Random Forest-----
accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = RFClassifier()
accuracies.append(accuracy_measure)
error_rates.append(error_rate_measure)
precisions.append(precision_measure)
specificities.append(specificity_measure)
FP_rates.append(FP_rate_measure)
TP_rates.append(TP_rate_measure)
