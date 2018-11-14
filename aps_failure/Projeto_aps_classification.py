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
import matplotlib.pyplot as plt


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

def CARTClassifier(trX, trY, tsX, tsY, data, max_Depth):
    cart = DecisionTreeClassifier(max_depth=max_Depth)

    model_CART = cart.fit(trX, trY)
    predY_CART = model_CART.predict(tsX)
    cnf_matrix_CART = confusion_matrix(tsY, predY_CART)
    
    print("CART Results:")
    accuracy_measure, error_rate_measure, precision_measure, specificity_measure, FP_rate_measure, TP_rate_measure = printMeasures(cnf_matrix_CART)
    
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

ACCURACY = [0.923625, 0.923625, 0.9428125, 0.9526875, 0.9515, 0.959875, 0.9640625, 0.967375, 0.970875, 0.975625, 0.975, 0.9763125, 0.9780625, 0.979625, 0.9803125, 0.9813125, 0.9819375, 0.982375, 0.982625, 0.9828125, 0.9831875, 0.9826875, 0.9834375, 0.9833125, 0.983125, 0.9835, 0.983625, 0.9843125, 0.984125, 0.9840625, 0.983875, 0.9841875, 0.984, 0.98475, 0.983625, 0.9840625, 0.9850625, 0.9848125, 0.9850625, 0.9853125, 0.98525, 0.9855, 0.9848125, 0.9854375, 0.9856875, 0.9858125, 0.9858125, 0.98625, 0.9864375, 0.98625]
ERRORS = [0.076375, 0.076375, 0.0571875, 0.0473125, 0.0485, 0.040125, 0.0359375, 0.032625, 0.029125, 0.024375, 0.025, 0.0236875, 0.0219375, 0.020375, 0.0196875, 0.0186875, 0.0180625, 0.017625, 0.017375, 0.0171875, 0.0168125, 0.0173125, 0.0165625, 0.0166875, 0.016875, 0.0165, 0.016375, 0.0156875, 0.015875, 0.0159375, 0.016125, 0.0158125, 0.016, 0.01525, 0.016375, 0.0159375, 0.0149375, 0.0151875, 0.0149375, 0.0146875, 0.01475, 0.0145, 0.0151875, 0.0145625, 0.0143125, 0.0141875, 0.0141875, 0.01375, 0.0135625, 0.01375]
PRECISION = [0.922752, 0.922752, 0.942528, 0.952512, 0.95168, 0.960832, 0.965504, 0.969472, 0.973376, 0.978496, 0.978112, 0.979904, 0.981952, 0.98368, 0.98464, 0.985664, 0.986496, 0.987392, 0.987904, 0.98816, 0.988736, 0.988608, 0.989184, 0.989056, 0.989312, 0.989632, 0.989888, 0.990848, 0.990784, 0.990592, 0.990528, 0.990848, 0.990784, 0.991552, 0.99072, 0.991104, 0.992256, 0.991936, 0.99264, 0.992704, 0.993024, 0.99296, 0.99264, 0.99328, 0.993152, 0.993728, 0.993344, 0.993792, 0.994624, 0.99424]
SPECIFICITY = [0.96, 0.96, 0.9546666666666667, 0.96, 0.944, 0.92, 0.904, 0.88, 0.8666666666666667, 0.856, 0.8453333333333334, 0.8266666666666667, 0.816, 0.8106666666666666, 0.8, 0.8, 0.792, 0.7733333333333333, 0.7626666666666667, 0.76, 0.752, 0.736, 0.744, 0.744, 0.7253333333333334, 0.728, 0.7226666666666667, 0.712, 0.7066666666666667, 0.712, 0.7066666666666667, 0.7066666666666667, 0.7013333333333334, 0.7013333333333334, 0.688, 0.6906666666666667, 0.6853333333333333, 0.688, 0.6693333333333333, 0.6773333333333333, 0.6613333333333333, 0.6746666666666666, 0.6586666666666666, 0.6586666666666666, 0.6746666666666666, 0.656, 0.672, 0.672, 0.6453333333333333, 0.6533333333333333]
FPRATES = [0.7702616464582004, 0.7702616464582004, 0.714968152866242, 0.6733212341197822, 0.6807935076645627, 0.6394984326018809, 0.6138952164009112, 0.5910780669144982, 0.5614035087719298, 0.5114155251141552, 0.5189681335356601, 0.5032051282051282, 0.47959183673469385, 0.4561717352415027, 0.4444444444444444, 0.42748091603053434, 0.4153543307086614, 0.40451745379876797, 0.39789473684210525, 0.39361702127659576, 0.38427947598253276, 0.3920704845814978, 0.37723214285714285, 0.38, 0.3804100227790433, 0.3724137931034483, 0.3682983682983683, 0.348780487804878, 0.35207823960880197, 0.35507246376811596, 0.3583535108958838, 0.35049019607843135, 0.3538083538083538, 0.3341772151898734, 0.3598014888337469, 0.3492462311557789, 0.3201058201058201, 0.328125, 0.31420765027322406, 0.30978260869565216, 0.30532212885154064, 0.30303030303030304, 0.31767955801104975, 0.29829545454545453, 0.2972222222222222, 0.28488372093023256, 0.29213483146067415, 0.27793696275071633, 0.25766871165644173, 0.26865671641791045]
TPRATES = [0.9756394640682094, 0.9756394640682094, 0.9762678157109712, 0.9763826018500296, 0.9767472411981083, 0.977536137517906, 0.9780226904376013, 0.9786794159452126, 0.979078151152311, 0.9794362588084561, 0.9796794871794872, 0.9801549196594328, 0.9804460348904084, 0.9806048232742121, 0.9808734459674848, 0.9808929367556206, 0.9810960473553562, 0.981549815498155, 0.981808930161557, 0.9818759936406996, 0.98207361261204, 0.9824460980728869, 0.9822688274547188, 0.9822665734443526, 0.9827082008900191, 0.9826512455516014, 0.9827805311983734, 0.9830465426376278, 0.9831703289724374, 0.9830422356303589, 0.983166052598145, 0.983171397726551, 0.9832952235772358, 0.9833079461792333, 0.9836065573770492, 0.9835503334391871, 0.9836939280502506, 0.9836263248080218, 0.9840746145549141, 0.9838883602917856, 0.9842679522963714, 0.9839548452562151, 0.9843244272386875, 0.9843343692522357, 0.9839578974066324, 0.9844037278894313, 0.9840233310086858, 0.9840304182509506, 0.9846670468225306, 0.9844740177439797]


plt.plot(ACCURACY)

    
    
    
    
    
