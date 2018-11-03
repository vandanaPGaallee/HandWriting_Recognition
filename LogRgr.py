
# coding: utf-8

# In[102]:


from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
from itertools import islice
import pandas as pd


# In[103]:


M = 10
PHI = []
train = []
TrainingTarget = []
TrainingData = []
ValDataAct = []
ValData = []
TestDataAct = []
TestData = []


# # Preparing Training Data

# In[104]:


#store the 80% of the training target data which is 55699
def GenerateTrainingTarget(rawTraining,TrainingPercent = 80): 
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

#store the 80% of the training  data which is 55699
def GenerateTrainingDataMatrix(rawData, TrainingPercent = 80):
    T_len = int(math.ceil(len(rawData)*0.01*TrainingPercent)) # this is computing the column lenth 0 to 55699
    d2 = rawData[0:T_len,:]
    #print(str(TrainingPercent) + "% Training Data Generated..")
    return d2

#store the 10% of the validation data and testing data which is 6962 which is 41 * 6962
def GenerateValData(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    dataMatrix = rawData[TrainingCount+1:V_End,:]
    #print (str(ValPercent) + "% Val Data Generated..")  
    return dataMatrix

#store the 10% of the validation target and testing target which is 6962 which is 41 * 6962
def GenerateValTargetVector(rawData, ValPercent, TrainingCount): 
    valSize = int(math.ceil(len(rawData)*ValPercent*0.01))
    V_End = TrainingCount + valSize
    t =rawData[TrainingCount+1:V_End]
    #print (str(ValPercent) + "% Val Target Data Generated..")
    return t


# In[105]:


def CreateHumanSubtract(fdf, x1, x2, y1, y2):
    temp1 = fdf.iloc[:,x1:x2]
    temp2 = fdf.iloc[:,y1:y2]
    sub_df = (temp1 - temp2.values).abs()
    #sub_df = (concat_df[['f1_x','f2_x','f3_x','f4_x','f5_x','f6_x','f7_x','f8_x','f9_x']] - concat_df[['f1_y','f2_y','f3_y','f4_y','f5_y','f6_y','f7_y','f8_y','f9_y']].values).abs()
    return sub_df

def MergeHumanDataset(FeatureData, SamePairs):
    df = pd.merge(SamePairs, FeatureData,  left_on= ['img_id_A'], right_on= ['img_id'], how='left')
    f_df = pd.merge(df, FeatureData, left_on= ['img_id_B'], right_on= ['img_id'], how='left')
    return f_df

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI)) # compute the linear regression function y(x,w)
    print(Y)
    ##print ("Test Out Generated..")
    return Y

def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)): #computing the root mean squared error
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i][0]),2) # summation of squares of error
        if(int(np.around(VAL_TEST_OUT[i][0], 0)) == ValDataAct[i]): # classifying the regression output to three ranks 0,1,2 by rounding the y value to nearest even number
            counter = counter + 1
    accuracy = (float((counter*100))/float(len(VAL_TEST_OUT))) #computes the ratio of correct prediction to total input
    ##print ("Accuracy Generated..")
    ##print ("Validation E_RMS : " + str(math.sqrt(sum/len(VAL_TEST_OUT))))
    return (str(accuracy) + ',' +  str(math.sqrt(sum/len(VAL_TEST_OUT)))) #accuracy  and root mean squared error

def ProcessDataset(H_target_df, H_concat_df, TrainingPercent, ValidationPercent, TestPercent):
    global TrainingTarget
    global TrainingData
    global ValDataAct
    global ValData
    global TestDataAct
    global TestData
    RawData = H_concat_df.values
    RawData = RawData[:, ~(RawData == RawData[0,:]).all(0)]
    TrainingTarget = np.array(GenerateTrainingTarget(H_target_df.values,TrainingPercent))
    TrainingData   = GenerateTrainingDataMatrix(RawData,TrainingPercent)
    ValDataAct = np.array(GenerateValTargetVector(H_target_df.values,ValidationPercent, (len(TrainingTarget))))
    ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
    TestDataAct = np.array(GenerateValTargetVector(H_target_df.values,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
    TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))


# # Logistic Regression Model

# In[106]:


def FindActivation(X, W):
#     print(X.shape)
#     print(W.shape)
    WX = np.dot(np.transpose(W),np.transpose(X))
    a = 1 / (1 + np.exp(-WX))
    return a

def LOGRModel(iter):    
    global TrainingTarget,TrainingData, ValDataAct, ValData , TestDataAct, TestData
    W_Now        = np.random.random((TrainingData.shape[1]+1,1))
    La           = 0.03
    learningRate = 0.5
    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = []
    W_Mat        = []
    acc_tr = []
    acc_val = []
    acc_test = []
    #AddBias
    X = np.ones((TrainingData.shape[0],1))
    TrainingData = np.hstack((TrainingData,X))
    X = np.ones((ValData.shape[0],1))
    ValData = np.hstack((ValData,X))
    X = np.ones((TestData.shape[0],1))
    TestData = np.hstack((TestData,X))
    print("After Adding Bias")
    print('----------TRAINING DATA--------------')
    print(TrainingTarget.shape)
    print(TrainingData.shape)
    print('---------VALIDATION DATA---------------')
    print(ValDataAct.shape)
    print(ValData.shape)
    print('----------TESTING DATA-------------')
    print(TestDataAct.shape)
    print(TestData.shape)
    print('----------WEIGHT-------------')
    print(W_Now.shape)
    for i in range(0,iter): 
        G = FindActivation(TrainingData, W_Now)
        val = np.subtract(G, np.transpose(TrainingTarget))
        Delta_E_D = np.dot(val,TrainingData)/TrainingTarget.shape[0]
        La_Delta_E_W  = np.dot(La,W_Now) # Error regularization
        Delta_E       = np.add(np.transpose(Delta_E_D),La_Delta_E_W)  # adding regularization to gradient error
        Delta_W       = -np.dot(learningRate,Delta_E) # multipying learning rate to computed error
        W_T_Next      = W_Now + Delta_W # subtracting error from output
        W_Now         = W_T_Next # updating the weight
        #-----------------TrainingData Accuracy---------------------#
        TR_TEST_OUT   = FindActivation(TrainingData,W_T_Next) 
        Erms_TR       = GetErms(np.transpose(TR_TEST_OUT),TrainingTarget)
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))
        acc_tr.append(float(Erms_TR.split(',')[0]))
        #-----------------ValidationData Accuracy---------------------#
        VAL_TEST_OUT  = FindActivation(ValData,W_T_Next) 
        Erms_Val      = GetErms(np.transpose(VAL_TEST_OUT),ValDataAct)
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))
        acc_val.append(float(Erms_Val.split(',')[0]))
        #-----------------TestingData Accuracy---------------------#
        TEST_OUT      = FindActivation(TestData,W_T_Next) 
        Erms_Test = GetErms(np.transpose(TEST_OUT),TestDataAct)
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))
        acc_test.append(float(Erms_Test.split(',')[0]))  
    print ('\n----------Gradient Descent Solution--------------------')
    print('learning rate %s' % learningRate)
    print('Lambda %s' % La)
    print ("Accuracy Training   = " + str(np.around(max(acc_tr),5)))
    print ("Accuracy Validation = " + str(np.around(max(acc_val),5)))
    print ("Accuracy Testing    = " + str(np.around(max(acc_test),5)))
    print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
    print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))


# # Data Preprocessing

# In[107]:


def PreProcessData(filename,f2, f3, x1, x2, y1, y2, iter, trp, vp, tp):
    FeatureData = pd.read_csv(filename) #Extract the da
    SamePairs   = pd.read_csv(f2)
    DiffnPairs   = pd.read_csv(f3)
    Merge_S_df = MergeHumanDataset(FeatureData, SamePairs)
#     print(Merge_S_df.shape)
    Merge_D_df = MergeHumanDataset(FeatureData, DiffnPairs)
#     print(Merge_D_df.shape)
    Merge_D = Merge_D_df.sample(n=Merge_S_df.shape[0])
#     print(Merge_D.shape)
    temp = pd.concat([Merge_S_df, Merge_D])
    temp = temp.sample(frac=1)
#     print(temp)
#     print(temp.shape)
    H_concat_df = temp.iloc[:, np.r_[x1:x2, y1:y2]]
    H_subtract_df = CreateHumanSubtract(temp, x1, x2, y1, y2)
    H_target_df = temp['target']
#     print("H_concat")
#     print(H_concat_df)
#     print(H_concat_df.columns.values)
#     print("H_Sub")
#     print(H_subtract_df)
#     print(H_subtract_df.columns.values)
#     print("H_Target")
#     print(H_target_df)
    print("-------------------------------------------------------------")
    print("                 %s RESULTS" % filename.split(".")[0])
    print("-------------------------------------------------------------")
    print("\n------------------------ CONCAT RESULTS--------------------\n")
    ProcessDataset(H_target_df, H_concat_df, trp, vp, tp)
    LOGRModel(iter)
    print("\n------------------------ SUBTRACT RESULTS--------------------\n")
    ProcessDataset(H_target_df, H_subtract_df, trp, vp, tp)
    LOGRModel(iter)



# In[108]:


def logr_main():
    PreProcessData("HumanObserved-Features-Data.csv", "same_pairs.csv","diffn_pairs.csv", 5, 14, 16, 25, 200, 80, 10, 10)
    PreProcessData("GSC-Features.csv", "G_same_pairs.csv","G_diffn_pairs.csv", 4, 516, 517, 1029, 50, 40, 10, 10)


# In[110]:




