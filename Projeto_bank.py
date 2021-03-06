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
def preprocessData(df):
    label_encoder = LabelEncoder()
    dummy_encoder = OneHotEncoder()
    pdf = pd.DataFrame()
    for att in df.columns:
        if df[att].dtype == np.float64 or df[att].dtype == np.int64:
            pdf = pd.concat([pdf, df[att]], axis=1)
        else:
            df[att] = label_encoder.fit_transform(df[att])
            # Fitting One Hot Encoding on train data
            temp = dummy_encoder.fit_transform(df[att].values.reshape(-1,1)).toarray()
            # Changing encoded features into a dataframe with new column names
            temp = pd.DataFrame(temp,
                                columns=[(att + "_" + str(i)) for i in df[att].value_counts().index])
            # In side by side concatenation index values should be same
            # Setting the index values similar to the data frame
            temp = temp.set_index(df.index.values)
            # adding the new One Hot Encoded varibales to the dataframe
            pdf = pd.concat([pdf, temp], axis=1)
    return pdf


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

#Carregamento e Processamento de dados
dataset = pd.read_csv('bank.csv');

dataset['pep'] = [1 if line=='YES' else 0 for line in dataset.pep]
 
df_majority = dataset[dataset.pep==0]
df_minority = dataset[dataset.pep==1]

df_minority_resampled = resample(df_minority, replace = True, n_samples= 326)

dataset =  pd.concat([df_majority, df_minority_resampled])

print(dataset)

X = dataset.iloc[:,:-1]
Y = dataset['pep']

preprocess_X = preprocessData(X)

#Separacao dos grupos de teste e treino
trX, tsX, trY, tsY = train_test_split(preprocess_X, Y, train_size=0.7, stratify=Y)

#Funcoes de aprendizagem

#-----K-nearest neighbors (Instance-based Learning)-----
knn = KNeighborsClassifier(n_neighbors=5)

model_KNN = knn.fit(trX, trY)
predY_KNN = model_KNN.predict(tsX)
cnf_matrix_KNN = confusion_matrix(tsY, predY_KNN)

print("KNN Results:")
print("Confusion matrix:\n", cnf_matrix_KNN)
print("Accuracy:", accuracy(cnf_matrix_KNN))
print("Error rate:", error_rate(cnf_matrix_KNN))
print("Precision:", precision(cnf_matrix_KNN))
print("Specifity:", specificity(cnf_matrix_KNN))
print("FP rate:", f_p_rate(cnf_matrix_KNN))
print("TP rate:", t_p_rate(cnf_matrix_KNN), "\n")
printRocChart(tsY,predY_KNN)

#-----Naive Bayes-----
clf = GaussianNB()

model_GNB = clf.fit(trX, trY)
predY_GNB = model_GNB.predict(tsX)
cnf_matrix_GNB = confusion_matrix(tsY, predY_GNB)

print("Naive Bayes Results:")
print("Confusion matrix:\n", cnf_matrix_GNB)
print("Accuracy:", accuracy(cnf_matrix_GNB))
print("Error rate:", error_rate(cnf_matrix_GNB))
print("Precision:", precision(cnf_matrix_GNB))
print("Specifity:", specificity(cnf_matrix_GNB))
print("FP rate:", f_p_rate(cnf_matrix_GNB))
print("TP rate:", t_p_rate(cnf_matrix_GNB), "\n")
printRocChart(tsY,predY_GNB)

#-----CART (Decision Trees)-----
cart = DecisionTreeClassifier()

model_CART = cart.fit(trX, trY)
predY_CART = model_CART.predict(tsX)
cnf_matrix_CART = confusion_matrix(tsY, predY_CART)

print("CART Results:")
print("Confusion matrix:\n",cnf_matrix_CART)
print("Accuracy:", accuracy(cnf_matrix_CART))
print("Error rate:", error_rate(cnf_matrix_CART))
print("Precision:", precision(cnf_matrix_CART))
print("Specifity:", specificity(cnf_matrix_CART))
print("FP rate:", f_p_rate(cnf_matrix_CART))
print("TP rate:", t_p_rate(cnf_matrix_CART), "\n")
printRocChart(tsY,predY_CART)

#-----Random Forest-----
rf = RandomForestClassifier()

model_RF = rf.fit(trX, trY)
predY_RF = model_RF.predict(tsX)
cnf_matrix_RF = confusion_matrix(tsY, predY_RF)

print("Random Forest Results:")
print("Confusion matrix:\n", cnf_matrix_RF)
print("Accuracy:", accuracy(cnf_matrix_RF))
print("Error rate:", error_rate(cnf_matrix_RF))
print("Precision:", precision(cnf_matrix_RF))
print("Specifity:", specificity(cnf_matrix_RF))
print("FP rate:", f_p_rate(cnf_matrix_RF))
print("TP rate:", t_p_rate(cnf_matrix_RF), "\n")      
printRocChart(tsY,predY_RF)

