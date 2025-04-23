#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Package for Mask matrix optimization
import numpy as np


# In[2]:


#'fil' is identical as 'filtered_' in "GAE-AM procedure.py"
#Extracted Causal DAG
fil=np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
       [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
       [1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


# In[3]:


# Extraction of 2-hop subgraph DAG
filtered_ = fil[[10,4,7,9,6,8,1],:][:,[10,4,7,9,6,8,1]]


# In[4]:


#Setting dimensions for StrNN skeleton
inp_, h_1, h_2, h_3, h_4, out_ = 7, 20, 10, 10, 20, 7


# In[5]:


#initial mask matrix settings before optimization(before greedy factorization)
M5 = np.zeros(h_4*out_).reshape(h_4,-1)
M4 = np.zeros(h_3*h_4).reshape(h_3,-1)
M3 = np.zeros(h_2*h_3).reshape(h_2,-1)
M2 = np.zeros(h_1*h_2).reshape(h_1,-1)
M1 = np.zeros(inp_*h_1).reshape(inp_,-1)


# In[6]:


#First Factorization-Step 1) M5 computation
L = 0
stack =[]
for t in range(0,filtered_.shape[0],1):
    if sum(filtered_[t]) != 0:
        stack.append(filtered_[t])
        L += 1
stack_ = np.array(stack)

m1 = M5.shape[0]//stack_.shape[0] ;n1=M5.shape[0]%stack_.shape[0] #m1: 몫
for i in range(0, (m1-1),1):
    stack_ = np.concatenate( (stack_,np.array(stack)), axis=0)
M5 = np.concatenate( (stack_, stack_[0:n1]),axis=0)
M5


# In[7]:


#First Factorization-Step 2) A1 computation
A_original = filtered_
A1 = np.ones(inp_*h_4).reshape(inp_,-1)
for i in range(0, A1.shape[0],1):
    C = []
    for j in range(0,A_original.shape[1],1):
        if A_original[i,j] == 0: C.append(j)
    T = M5[:,C]
    R = []
    for q in range(0, T.shape[0],1):
        if np.sum(T[q,:]) != 0: R.append(q)
    A1[i,R] = 0
A1


# In[8]:


#Second Factorization-Step 1) M4 computation
L = 0
stack =[]
for t in range(0,A1.shape[0],1):
    if sum(A1[t]) != 0:
        stack.append(A1[t])
        L += 1
stack_ = np.array(stack)

m1 = M4.shape[0]//stack_.shape[0] ;n1=M4.shape[0]%stack_.shape[0] #m1: 몫
for i in range(0, (m1-1),1):
    stack_ = np.concatenate( (stack_,np.array(stack)), axis=0)
M4 = np.concatenate( (stack_, stack_[0:n1]),axis=0)
M4


# In[9]:


#Second Factorization-Step 2) A2 computation
A2 = np.ones(inp_*h_3).reshape(inp_,-1)
for i in range(0, A2.shape[0],1):
    C = []
    for j in range(0,A1.shape[1],1):
        if A1[i,j] == 0: C.append(j)
    T = M4[:,C]
    R = []
    for q in range(0, T.shape[0],1):
        if np.sum(T[q,:]) != 0: R.append(q)
    A2[i,R] = 0
A2


# In[10]:


#Third Factorization-Step 1) M3 computation
L = 0
stack =[]
for t in range(0,A2.shape[0],1):
    if sum(A2[t]) != 0:
        stack.append(A2[t])
        L += 1
stack_ = np.array(stack)

m1 = M3.shape[0]//stack_.shape[0] ;n1=M3.shape[0]%stack_.shape[0] #m1: 몫
for i in range(0, (m1-1),1):
    stack_ = np.concatenate( (stack_,np.array(stack)), axis=0)
M3 = np.concatenate( (stack_, stack_[0:n1]),axis=0)
M3


# In[11]:


#Third Factorization-Step 2) A3 computation
A3 = np.ones(inp_*h_2).reshape(inp_,-1)
for i in range(0, A3.shape[0],1):
    C = []
    for j in range(0,A2.shape[1],1):
        if A2[i,j] == 0: C.append(j)
    T = M3[:,C]
    R = []
    for q in range(0, T.shape[0],1):
        if np.sum(T[q,:]) != 0: R.append(q)
    A3[i,R] = 0
A3


# In[12]:


#Fourth Factorization-Step 1) M2 computation
L = 0
stack =[]
for t in range(0,A3.shape[0],1):
    if sum(A3[t]) != 0:
        stack.append(A3[t])
        L += 1
stack_ = np.array(stack)

m1 = M2.shape[0]//stack_.shape[0] ;n1=M2.shape[0]%stack_.shape[0] #m1: 몫
for i in range(0, (m1-1),1):
    stack_ = np.concatenate( (stack_,np.array(stack)), axis=0)
M2 = np.concatenate( (stack_, stack_[0:n1]),axis=0)
M2


# In[13]:


#Fourth Factorization-Step 2) A4 computation
A4 = np.ones(inp_*h_1).reshape(inp_,-1)
for i in range(0, A4.shape[0],1):
    C = []
    for j in range(0,A3.shape[1],1):
        if A3[i,j] == 0: C.append(j)
    T = M2[:,C]
    R = []
    for q in range(0, T.shape[0],1):
        if np.sum(T[q,:]) != 0: R.append(q)
    A4[i,R] = 0
A4


# In[14]:


#M1 == A4(Factorization complete)
M1 = A4


# In[15]:


#Checking errors in mask matrix optimization
np.where(M1@M2@M3@M4@M5> 0,1,0)-filtered_


# In[16]:


filtered_


# In[17]:


#Importing required package for StrNN fitting
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from keras.models import Model
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
from keras.initializers import HeNormal,GlorotNormal


# In[18]:


dataframe = pd.read_csv("heart_failure_clinical_records_dataset.csv")
dataframe.head(5)


# In[19]:


#Eliminating follow-up time and sex variable for identical environment
dataframe.drop(['time','sex'],axis=1,inplace=True)


# In[20]:


dt = dataframe


# In[21]:


#Random Splitting of train, test data (same as in GAE-AM by setting identical seed(random_state))
from sklearn.model_selection import train_test_split
whole = dt.iloc[:,0:11]
train, test = train_test_split(whole,test_size=0.4, random_state=123123,shuffle=True)


# In[22]:


#Normalization of train data
mn1 = MinMaxScaler()
train_ = mn1.fit_transform(train)
train_ = pd.DataFrame(train_, columns=dt.columns)


# In[23]:


#Normalization of test data
test_ = mn1.transform(test)
test_ = pd.DataFrame(test_, columns=dt.columns)


# In[24]:


#Setting Hardamard multiplication of Masks using custom kernel constraint(keras.constraints.Constraint)
class mask_1(tf.keras.constraints.Constraint):
    def __call__(self,w):
        return M1*w

class mask_2(tf.keras.constraints.Constraint):
    def __call__(self,w):
        return M2*w

class mask_3(tf.keras.constraints.Constraint):
    def __call__(self,w):
        return M3*w

class mask_4(tf.keras.constraints.Constraint):
    def __call__(self,w):
        return M4*w

class mask_5(tf.keras.constraints.Constraint):
    def __call__(self,w):
        return M5*w


# In[25]:


#Implementing only 2-hop subgraph data
train_ = train_.iloc[:,[10,4,7,9,6,8,1]]
test_ = test_.iloc[:,[10,4,7,9,6,8,1]]


# In[26]:


#Defining Weighted MSE, MAE. Unlike in GAE-AM, implements weighting in terms of t'otal' MSE
#Excluded non-connected output node values from loss computation using v.
@tf.function
def MSE_A(y_true, y_pred):
    v = np.zeros(7)
    v[0:5]=1
    v[[0,3]] = 0.4
    v[[1,2,4]] = 0.6
    val = y_true*v-y_pred*v
    val = tf.square(val)
    val = tf.reduce_sum(val, axis=1)
    return tf.reduce_mean(val)

@tf.function
def MAE_A(y_true, y_pred):
    v = np.zeros(7)
    v[0:5]=1
    v[[0,3]] = 0.4
    v[[1,2,4]] = 0.6
    val = y_true*v-y_pred*v
    val = tf.abs(val)
    val = tf.reduce_sum(val, axis=1)
    return tf.reduce_mean(val)


# In[27]:


#Build Skeleton for StrNN
e_s = EarlyStopping(monitor='loss', patience=300)
tf.keras.utils.set_random_seed(321)
k = tf.keras.initializers.GlorotUniform(seed=123)
input1 = tf.keras.layers.Input(shape=(7,))
x=tf.keras.layers.Dense(units=20, use_bias= False, activation='sigmoid',kernel_initializer=k, kernel_constraint=mask_1())(input1) 
y_ = tf.keras.layers.Dense(units=10, use_bias= False, activation='sigmoid',kernel_initializer=k, kernel_constraint = mask_2())(x) 
Encoder = tf.keras.models.Model(inputs=input1, outputs=y_)

input2 = tf.keras.layers.Input(shape=(10,))
x2=tf.keras.layers.Dense(units=10, use_bias= False, kernel_initializer=k, activation='sigmoid', kernel_constraint = mask_3())(input2)
x2=tf.keras.layers.Dense(units=20, use_bias= False, kernel_initializer=k, activation='sigmoid', kernel_constraint = mask_4())(x2)
y_2 = tf.keras.layers.Dense(units=7, use_bias= False, kernel_initializer=k, activation='sigmoid', kernel_constraint=mask_5())(x2) 
Decoder = tf.keras.models.Model(inputs=input2, outputs=y_2)


# In[28]:


#Building StrNN skeleton
class AE(Model):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Encoder
        self.decoder = Decoder
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
autoencoder = AE()


# In[29]:


#Setting Adam optimizer for StrNN
from tensorflow.keras.optimizers import RMSprop, Adam
autoencoder.compile(optimizer=Adam(learning_rate=0.005), loss=[MSE_A], metrics=[MAE_A])


# In[30]:


#Fitting StrNN, 2000 epochs, batch_size=20
hist = autoencoder.fit(train_, train_,epochs=2000, batch_size=50, shuffle=True, callbacks=[e_s])


# In[31]:


autoencoder.evaluate(train_,train_)


# In[32]:


#Checking Loss function values per epoch
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
plt.plot(hist.history['loss'], color="red",label="loss")
#plt.grid()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()


# In[33]:


#Checking Metric values per epoch
plt.plot(hist.history['MAE_A'], color="darkblue",label="MAE")
#plt.grid()
plt.xlabel('epochs')
plt.ylabel('MAE')
plt.legend()


# In[34]:


#Returning StrNN results for Train data
ed1 = autoencoder.encoder(np.array(train_))
S1 = train_.columns
dc1 = pd.DataFrame(autoencoder.decoder(ed1), columns=S1)


# In[35]:


#Cheking StrNN performance in train data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, f1_score, precision_score,roc_auc_score, matthews_corrcoef

res1 = []
for i in dc1['DEATH_EVENT']:
    if (np.round(i)<=0.5):
        res1.append(0)
    else:
        res1.append(1)

res2 = train_['DEATH_EVENT']

print(accuracy_score(res2,res1))
print(f1_score(res2,res1))
print(roc_auc_score(res2,dc1['DEATH_EVENT']))
print(matthews_corrcoef(res2, res1))


# In[36]:


#Returning StrNN values for test data
ed = autoencoder.encoder(np.array(test_))
S = test_.columns
dc = pd.DataFrame(autoencoder.decoder(ed), columns=S)


# In[37]:


#Cheking StrNN performance in train data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, f1_score, precision_score,roc_auc_score, matthews_corrcoef

res_1 = []
for i in dc['DEATH_EVENT']:
    if (np.round(i)<=0.5):
        res_1.append(0)
    else:
        res_1.append(1)

        
res_2 = test_['DEATH_EVENT']

dis = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(res_2,res_1), display_labels=[0,1])
dis.plot()
plt.grid()

print(accuracy_score(res_2,res_1))
print(f1_score(res_2,res_1))
print(roc_auc_score(res_2,dc['DEATH_EVENT']))
print(matthews_corrcoef(res_2, res_1))


# In[38]:


#Checking ROC curve in test data
from sklearn.metrics import auc, roc_curve
import seaborn as sns
sns.set_style("darkgrid")
fpr, tpr,thres = roc_curve(res_2,dc['DEATH_EVENT'])
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,label="Test: ROC area = %0.3f" %roc_auc,color="darkblue")
plt.plot(np.linspace(0,1,100),np.linspace(0,1,100), linestyle="--" ,color="black")
plt.legend()


# In[39]:


#Checking ROC curve in train data
fpr1, tpr1,thres1 = roc_curve(res2,dc1['DEATH_EVENT'])
roc_auc1 = auc(fpr1,tpr1)
plt.plot(fpr1,tpr1,label="Train:ROC area = %0.3f" %roc_auc1,color="red")
plt.plot(np.linspace(0,1,100),np.linspace(0,1,100), linestyle="--" ,color="black")
plt.legend()


# In[40]:


#Setting Dictionary of factors in 2-hop subgraph for convenience
dic = {}
for i in enumerate(dt.columns[[10,4,7,9,6,8,1]]):
    dic[i[0]] = i[1]


# In[41]:


#Visualization of 2-hop subgraph of causal DAG
import networkx as nx
tf.keras.utils.set_random_seed(111)
extracted_A = filtered_*1
G = nx.from_numpy_matrix(extracted_A, create_using=nx.DiGraph())
nx.draw(G,with_labels=True,connectionstyle="arc, rad=0.2",node_size=0.8e+3,font_size=10,node_color="lightgray",labels=dic)
plt.show()


# In[42]:


dic


# In[43]:


train_.columns


# In[44]:


#Intervention analysis for Serum Creatinine: step1) estimate weights W using KDE
from sklearn.neighbors import KernelDensity 
import numpy as np

Z = np.array(train_['serum_sodium']).reshape(-1,1)
Z1 = np.array(train_['ejection_fraction']).reshape(-1,1)
Z2 = np.array(train_['platelets']).reshape(-1,1)


kde2 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(np.hstack([Z1,Z2,Z])) 
log_density2 = kde2.score_samples(np.hstack([Z1,Z2,Z]))
density2 = np.exp(log_density2)


# In[45]:


#Intervention analysis for Serum Creatinine: step2) estimate ATE of intervention in serum creatinine
#non-influential values set as 0(no effect in StrNN results)
k = 0
int_effects = []
temp_ =[]
sig_ = []
Z1_ = Z1.reshape(-1)
Z2_ = Z2.reshape(-1)
Z = Z.reshape(-1)
for j in np.linspace(0,1,100):
    for i in range(0,len(Z),1):
        y_p = np.round(autoencoder.decoder(autoencoder.encoder(np.array([[0,Z1_[i],j,0,Z2_[i],0,0]])))) #non-influential values set as 0(no effect in StrNN results)
        k += np.array(y_p).reshape(-1)[0]*density2[i]
        temp_.append(np.array(y_p).reshape(-1)[0]*density2[i])
    int_effects.append(k/len(Z1_))
    sig_.append(np.std(temp_))
    temp_=[]
    k=0
    
a=np.max(train['serum_creatinine'])
b=np.min(train['serum_creatinine'])

plt.plot(np.linspace(0,1,100)*(a-b)+b,int_effects, label="E(Death_EVENT | do(serum_creatinine))", linestyle="-.",color="darkblue")
plt.fill_between(np.linspace(0,1,100)*(a-b)+b,y1=np.array(int_effects)+np.array(sig_),y2=np.array(int_effects)-np.array(sig_),color="darkred",alpha=0.3)
plt.xlabel("serum_creatinine")
#plt.grid()
plt.legend()
plt.show()


# In[46]:


#Intervention analysis for Ejection Fraction: step1) estimate weights W using KDE
from sklearn.neighbors import KernelDensity 
import numpy as np


Z = np.array(train_['platelets']).reshape(-1,1)
Z1 = np.array(train_['serum_creatinine']).reshape(-1,1)


kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(np.hstack([Z,Z1])) 
log_density = kde.score_samples(np.hstack([Z,Z1]))
density = np.exp(log_density)


# In[47]:


#Intervention analysis for Ejection fraction: step2) estimate ATE of intervention in ejection fraction
k = 0
int_effects = []
temp_ =[]
sig_ = []
Z_ = Z.reshape(-1)
Z1_ = Z1.reshape(-1)
for j in np.linspace(0,1,100):
    for i in range(0,len(Z),1):
        y_p = np.round(autoencoder.decoder(autoencoder.encoder(np.array([[0,j,Z1_[i],0,Z_[i],0,0]]))))
        k += np.array(y_p).reshape(-1)[0]*density[i]
        temp_.append(np.array(y_p).reshape(-1)[0]*density[i])
    int_effects.append(k/len(Z1_))
    sig_.append(np.std(temp_))
    temp_=[]
    k=0

a=np.max(train['ejection_fraction'])
b=np.min(train['ejection_fraction'])

plt.plot(np.linspace(0,1,100)*(a-b)+b,int_effects, label="E(Death_EVENT | do(ejection_fraction))", linestyle="-.",color="darkblue")
plt.fill_between(np.linspace(0,1,100)*(a-b)+b,y1=np.array(int_effects)+np.array(sig_),y2=np.array(int_effects)-np.array(sig_),color="darkred",alpha=0.3)
plt.xlabel("ejection_fraction")
#plt.grid()
plt.legend()
plt.show()


# In[48]:


#Intervention analysis for Platelets: step1) estimate weights W using KDE

from sklearn.neighbors import KernelDensity
import numpy as np


Z1 = np.array(train_['serum_sodium']).reshape(-1,1)
Z3 = np.array(train_['serum_creatinine']).reshape(-1,1)
Z2 = np.array(train_['ejection_fraction']).reshape(-1,1)
Z4 = np.array(train_['smoking']).reshape(-1,1)
I = np.array(train_['platelets']).reshape(-1,1)
I_ = np.linspace(0,1,100).reshape(-1,1) #interventional values to compute conditional distribution in (13) in article


kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(np.hstack([Z1,Z2,Z3]))
log_density = kde.score_samples(np.hstack([Z1,Z2,Z3]))
density = np.exp(log_density)

kde2 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(np.hstack([Z4,I]))
kde3 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(I)


# In[49]:


#Intervention analysis for Platelets: step2) estimate ATE of intervention in platelets
k = 0
int_effects = []
temp_ =[]
sig_ = []
Z1_ = Z1.reshape(-1)
Z2_ = Z2.reshape(-1)
Z3_ = Z3.reshape(-1)
for j in np.linspace(0,1,100):
    for i in range(0,len(Z1),1):
        y_p = np.round(autoencoder.decoder(autoencoder.encoder(np.array([[0,Z2_[i],Z3_[i],0,j,0,0]]))))
        
        log_density2 = kde2.score_samples(np.hstack([[Z4[i]],[[j]]]))
        density2 = np.exp(log_density2) 
        log_density3 = kde3.score_samples([[j]])
        density3 = np.exp(log_density3)
        cond_density = density2/density3
        
        k += np.array(y_p).reshape(-1)[0]*(density[i]*cond_density[0]  )
        temp_.append(np.array(y_p).reshape(-1)[0]*(density[i]*cond_density[0]))
    int_effects.append(k/len(Z1))
    sig_.append(np.std(temp_))
    temp_=[]
    k=0

a=np.max(train['platelets'])
b=np.min(train['platelets'])

plt.plot(np.linspace(0,1,100)*(a-b)+b,int_effects, label="E(DEATH_EVENT=1 | do(platelets))", linestyle="-.",color="darkblue")
plt.fill_between(np.linspace(0,1,100)*(a-b)+b,y1=np.array(int_effects)+np.array(sig_),y2=np.array(int_effects)-np.array(sig_),color="darkred",alpha=0.3)
plt.xlabel("platelets")
#plt.grid()
plt.legend()
plt.show()


# In[ ]:




