
# coding: utf-8

# In[101]:


from sklearn.cluster import KMeans
import numpy as np
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt
from itertools import islice
import pandas as pd


# In[102]:


M = 40
PHI = []
train = []
TrainingTarget = []
TrainingData = []
ValDataAct = []
ValData = []
TestDataAct = []
TestData = []


# # Preparing Training Data

# In[103]:


#store the 80% of the training target data which is 55699
def GenerateTrainingTarget(rawTraining,TrainingPercent): 
    TrainingLen = int(math.ceil(len(rawTraining)*(TrainingPercent*0.01)))
    t           = rawTraining[:TrainingLen]
    #print(str(TrainingPercent) + "% Training Target Generated..")
    return t

#store the 80% of the training  data which is 55699
def GenerateTrainingDataMatrix(rawData, TrainingPercent):
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

def CreateHumanTarget(Merge_S_df, Merge_D_df):
    target_df = Merge_S_df['target']
    target_df = target_df.append(Merge_D_df['target'])
    return target_df

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


# # Linear Regression Model

# In[104]:


#Generates covariance for 41 features accross the training data
def GenerateBigSigma(Data, MuMatrix,TrainingPercent):
    BigSigma    = np.zeros((len(Data[0]),len(Data[0]))) #41 * 41
    DataT       = np.transpose(Data) #65000 * 41
    TrainingLen = math.ceil(len(Data))  #55699   
    varVect     = []
    for i in range(0,len(Data[0])):
        vct = []
        for j in range(0,int(TrainingLen)):
            vct.append(Data[j][i])   #storing 55699 for each feature in vct 
        varVect.append(np.var(vct))  #computing variance for 55699 values for each feature and storing in varVect
    
    for j in range(len(Data[0])):
        BigSigma[j][j] = varVect[j] #storing the variance for 41 features across the diagnol in 41 * 41 matrix
    return BigSigma

#computes the scalar value for phi design matrix
def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow) # computes x - u where x is 41 features and muRow is the 41 mean values for each of feature
    T = np.dot(BigSigInv,np.transpose(R))  # computes dot product of 41 Inverse covariance and R 
    L = np.dot(R,T)# returns a scalar value
    return L

# computes the values of gaussian radial basis scalar value for each of the input
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv)) #computes the exponential of scalar value
    return phi_x

def GetPhiMatrix(Data, MuMatrix, BigSigma, TrainingPercent = 80):
    DataT = np.transpose(Data) # Transposing 41 * 65000  to 65000 * 41 
    TrainingLen = math.ceil(len(Data))  # 80% 55699     
    PHI = np.zeros((int(TrainingLen),len(MuMatrix))) # 55699 * 10 where 10 is number of clusters 
    BigSigInv = np.linalg.inv(BigSigma) # computes inverse of covariance matrix
    for  C in range(0,len(MuMatrix)): # 0 t0 10
        for R in range(0,int(TrainingLen)): # 0 to 55699
            PHI[R][C] = GetRadialBasisOut(Data[R], MuMatrix[C], BigSigInv) # computes the phi(x) for each cluster
    #print ("PHI Generated..")
    return PHI

def GetValTest(VAL_PHI,W):
    Y = np.dot(W,np.transpose(VAL_PHI)) # compute the linear regression function y(x,w)
    ##print ("Test Out Generated..")
    return Y

def GetErms(VAL_TEST_OUT,ValDataAct):
    sum = 0.0
    t=0
    accuracy = 0.0
    counter = 0
    val = 0.0
    for i in range (0,len(VAL_TEST_OUT)): #computing the root mean squared error
        sum = sum + math.pow((ValDataAct[i] - VAL_TEST_OUT[i]),2) # summation of squares of error
        if(int(np.around(VAL_TEST_OUT[i], 0)) == ValDataAct[i]): # classifying the regression output to three ranks 0,1,2 by rounding the y value to nearest even number
            counter+=1
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
    print('----------TRAINING DATA--------------')
    print(TrainingTarget.shape)
    print(TrainingData.shape)
    ValDataAct = np.array(GenerateValTargetVector(H_target_df.values,ValidationPercent, (len(TrainingTarget))))
    ValData    = GenerateValData(RawData,ValidationPercent, (len(TrainingTarget)))
    print('---------VALIDATION DATA---------------')
    print(ValDataAct.shape)
    print(ValData.shape)
    TestDataAct = np.array(GenerateValTargetVector(H_target_df.values,TestPercent, (len(TrainingTarget)+len(ValDataAct))))
    TestData = GenerateValData(RawData,TestPercent, (len(TrainingTarget)+len(ValDataAct)))
    print('----------TESTING DATA-------------')
    print(TestDataAct.shape)
    print(TestData.shape)

def LRModel(iter, TrainingPercent, ValidationPercent, TestPercent):    
    global TrainingTarget,TrainingData, ValDataAct, ValData , TestDataAct, TestData
    #this step is a optimazation technique to reduce the dimentionality
    kmeans = KMeans(n_clusters=M, random_state=0).fit(TrainingData) #Here we define the cluster size as 10 and random state to take random centroids initially.
    Mu = kmeans.cluster_centers_ #It takes 55699 * 41 values and reduces it to 10 clusters and returns 10 * 41 values where the each of the 41 features represent the average value in each cluster
    BigSigma     = GenerateBigSigma(TrainingData, Mu, TrainingPercent)
    TRAINING_PHI = GetPhiMatrix(TrainingData, Mu, BigSigma, TrainingPercent)
    TEST_PHI     = GetPhiMatrix(TestData, Mu, BigSigma, 100) 
    VAL_PHI      = GetPhiMatrix(ValData, Mu, BigSigma, 100)
    
    W_Now        = np.random.random((M))
    La           = 0.03
    learningRate = 0.5
    L_Erms_Val   = []
    L_Erms_TR    = []
    L_Erms_Test  = []
    W_Mat        = []
    acc_tr = []
    acc_val = []
    acc_test = []    
#     print(Mu.shape)
#     print(BigSigma.shape)
#     print(TRAINING_PHI.shape)
#     print(np.shape(W_Now))
#     print(VAL_PHI.shape)
#     print(TEST_PHI.shape)

    for i in range(0,iter): 

#         print ('---------Iteration: ' + str(i) + '--------------')
        Delta_E_D     = -np.dot((TrainingTarget[i] - np.dot(np.transpose(W_Now),TRAINING_PHI[i])),TRAINING_PHI[i]) # Computing Delta E_D which is the rate of change of error with respect to w 
        La_Delta_E_W  = np.dot(La,W_Now) # Error regularization
        Delta_E       = np.add(Delta_E_D,La_Delta_E_W)  # adding regularization to gradient error
        Delta_W       = -np.dot(learningRate,Delta_E) # multipying learning rate to computed error
        W_T_Next      = W_Now + Delta_W # subtracting error from output
        W_Now         = W_T_Next # updating the weight
        
        #-----------------TrainingData Accuracy---------------------#
        TR_TEST_OUT   = GetValTest(TRAINING_PHI,W_T_Next) 
        Erms_TR       = GetErms(TR_TEST_OUT,TrainingTarget)
        L_Erms_TR.append(float(Erms_TR.split(',')[1]))
        acc_tr.append(float(Erms_TR.split(',')[0]))
        #-----------------ValidationData Accuracy---------------------#
        VAL_TEST_OUT  = GetValTest(VAL_PHI,W_T_Next) 
        Erms_Val      = GetErms(VAL_TEST_OUT,ValDataAct)
        L_Erms_Val.append(float(Erms_Val.split(',')[1]))
        acc_val.append(float(Erms_Val.split(',')[0]))
        #-----------------TestingData Accuracy---------------------#
        TEST_OUT      = GetValTest(TEST_PHI,W_T_Next) 
        Erms_Test = GetErms(TEST_OUT,TestDataAct)
        L_Erms_Test.append(float(Erms_Test.split(',')[1]))
        acc_test.append(float(Erms_Test.split(',')[0]))  
    print ('\n----------Gradient Descent Solution--------------------')
    print('M %s' % M)
    print('learning rate %s' % learningRate)
    print('Lambda %s' % La)
    print ("Accuracy Training   = " + str(np.around(max(acc_tr),5)))
    print ("Accuracy Validation = " + str(np.around(max(acc_val),5)))
    print ("Accuracy Testing    = " + str(np.around(max(acc_test),5)))
    print ("E_rms Training   = " + str(np.around(min(L_Erms_TR),5)))
    print ("E_rms Validation = " + str(np.around(min(L_Erms_Val),5)))
    print ("E_rms Testing    = " + str(np.around(min(L_Erms_Test),5)))


# # Data Preprocessing

# In[105]:


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
    H_concat_df = temp.iloc[:, np.r_[x1:x2, y1:y2]]
    H_subtract_df = CreateHumanSubtract(temp, x1, x2, y1, y2)
    H_target_df = temp['target']    
    print("-------------------------------------------------------------")
    print("                 %s RESULTS" % filename.split(".")[0])
    print("-------------------------------------------------------------")
    print("\n------------------------ CONCAT RESULTS--------------------\n")
    ProcessDataset(H_target_df, H_concat_df, trp, vp, tp)
    LRModel(iter,trp, vp, tp)
    print("\n------------------------ SUBTRACT RESULTS--------------------\n")
    ProcessDataset(H_target_df, H_subtract_df, trp, vp, tp)
    LRModel(iter, trp, vp, tp)




# In[106]:


def lr_main():
    PreProcessData("HumanObserved-Features-Data.csv", "same_pairs.csv","diffn_pairs.csv", 5, 14, 16, 25,200, 80, 10, 10)
    PreProcessData("GSC-Features.csv", "G_same_pairs.csv","G_diffn_pairs.csv", 4, 516, 517, 1029,50, 30, 10, 10)


# In[107]:




