#!/usr/bin/env python
# coding: utf-8

# In[47]:


#%% Packages
import torch
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import torch.nn.functional as F


# In[48]:


City_df = pd.read_csv('city_hour.csv', delimiter=",")
List_cities=City_df.City.unique()


# In[49]:


Cities_dfs=[]
for city in List_cities:
    Cities_dfs.append(City_df.loc[City_df["City"] == city])          


# In[50]:


for city in Cities_dfs:    
    print(city.City.unique())
    print(city.count()/len(city))        


# In[51]:


for i in range(len(Cities_dfs)):
    Cities_dfs[i]=Cities_dfs[i].set_index('Datetime')
    Cities_dfs[i]=Cities_dfs[i].drop(columns=['City','AQI', 'AQI_Bucket','Benzene','Toluene','Xylene'])
    Cities_dfs[i]=Cities_dfs[i].loc[:, Cities_dfs[i].isnull().mean() <.4]   # Porcentaje de nulls mayor a 40% se elimina 
    date_time_inicial = datetime.datetime.strptime(Cities_dfs[i].index[0], '%Y-%m-%d %H:%M:%S') #fechas para el intervalo
#     print(date_time_inicial.date())
    date_time_final = datetime.datetime.strptime(Cities_dfs[i].index[-1], '%Y-%m-%d %H:%M:%S')
#     print(date_time_final.date())
    dateInicial=datetime.datetime.strftime(date_time_inicial.date() + datetime.timedelta(days=1),'%Y-%m-%d')
    dateFinal=datetime.datetime.strftime(date_time_final.date() + datetime.timedelta(days=-1),'%Y-%m-%d')
    mask = (Cities_dfs[i].index >= dateInicial + ' 00:00:00') & (Cities_dfs[i].index <= dateFinal +' 23:00:00') #Mascara para seleccionar entre las fechas
    Cities_dfs[i]=Cities_dfs[i][mask]
   


# In[52]:


Cities_dfs[0].head(24)


# In[53]:


for i in range(len(Cities_dfs)):
    if len(Cities_dfs[i]) % 48 != 0 :        
        Cities_dfs[i].drop(Cities_dfs[i].tail(24).index,inplace=True)


# In[54]:


from sklearn.impute import SimpleImputer


# In[55]:


aux_array= np.array([], dtype=np.int64).reshape(0,6)


# In[56]:


for i in range(0,len(Cities_dfs[0]),48):
    aux_df = Cities_dfs[0].iloc[i:i+48]
    if (aux_df.isnull().mean()>.1).sum() == 0: # Mas de % de nan se contabliza para que sea ideal debe ser ==0
        print(aux_df)
        aux_df.fillna(np.NaN)
        imp=SimpleImputer(missing_values=np.NaN)
        aux_df=pd.DataFrame(imp.fit_transform(aux_df))
        aux_df.to_numpy()
        aux_array=np.vstack([aux_array, aux_df])   
 


# In[57]:


np.savetxt("foo.csv", aux_array, delimiter=",")


# In[58]:


aux_array


# In[59]:


# dataset_file = 'Spam.csv'
use_gpu = False  # set it to True to use GPU and False to use CPU
if use_gpu:
    torch.cuda.set_device(0)


# In[60]:


#%% System Parameters
# 1. Mini batch size
mb_size = 128
# 2. Missing rate
p_miss = 0.2
# 3. Hint rate
p_hint = 0.9
# 4. Loss Hyperparameters
alpha = 10
# 5. Train Rate
train_rate = 0.8

#%% Data

# Data generation
# Data = np.loadtxt(dataset_file, delimiter=",",skiprows=1)
Data = aux_array

# Parameters
No = len(Data)
Dim = len(Data[0,:])

# Hidden state dimensions
H_Dim1 = Dim
H_Dim2 = Dim



# In[61]:


# Normalization (0 to 1)
Min_Val = np.zeros(Dim)
Max_Val = np.zeros(Dim)

for i in range(Dim):
    Min_Val[i] = np.min(Data[:,i])
    Data[:,i] = Data[:,i] - np.min(Data[:,i])
    Max_Val[i] = np.max(Data[:,i])
    Data[:,i] = Data[:,i] / (np.max(Data[:,i]) + 1e-6)    

#%% Missing introducing
p_miss_vec = p_miss * np.ones((Dim,1)) 

   
Missing = np.zeros((No,Dim))

for i in range(Dim):
    A = np.random.uniform(0., 1., size = [len(Data),])
#     A = Data[:,i]
    B = A > p_miss_vec[i]
    Missing[:,i] = 1.*B
    

#%% Train Test Division    
   
idx = np.random.permutation(No)

Train_No = int(No * train_rate)
Test_No = No - Train_No
    
# Train / Test Features
trainX = Data[idx[:Train_No],:]
testX = Data[idx[Train_No:],:]

# Train / Test Missing Indicators
trainM = Missing[idx[:Train_No],:]
testM = Missing[idx[Train_No:],:]


# In[62]:


#%% Necessary Functions

# 1. Xavier Initialization Definition
# def xavier_init(size):
#     in_dim = size[0]
#     xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
#     return tf.random_normal(shape = size, stddev = xavier_stddev)
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size = size, scale = xavier_stddev)
    
# Hint Vector Generation
def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size = [m, n])
    B = A > p
    C = 1.*B    
    return C


# In[ ]:





# In[63]:


#GAIN Architecture
#GAIN Consists of 3 Components

#Generator
#Discriminator
#Hint Mechanism


# In[ ]:





# In[64]:


#%% 1. Discriminator
if use_gpu is True:
    D_W1 = torch.tensor(xavier_init([Dim*2, H_Dim1]),requires_grad=True, device="cuda")     # Data + Hint as inputs
    D_b1 = torch.tensor(np.zeros(shape = [H_Dim1]),requires_grad=True, device="cuda")

    D_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]),requires_grad=True, device="cuda")
    D_b2 = torch.tensor(np.zeros(shape = [H_Dim2]),requires_grad=True, device="cuda")

    D_W3 = torch.tensor(xavier_init([H_Dim2, Dim]),requires_grad=True, device="cuda")
    D_b3 = torch.tensor(np.zeros(shape = [Dim]),requires_grad=True, device="cuda")       # Output is multi-variate
else:
    D_W1 = torch.tensor(xavier_init([Dim*2, H_Dim1]),requires_grad=True)     # Data + Hint as inputs
    D_b1 = torch.tensor(np.zeros(shape = [H_Dim1]),requires_grad=True)

    D_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]),requires_grad=True)
    D_b2 = torch.tensor(np.zeros(shape = [H_Dim2]),requires_grad=True)

    D_W3 = torch.tensor(xavier_init([H_Dim2, Dim]),requires_grad=True)
    D_b3 = torch.tensor(np.zeros(shape = [Dim]),requires_grad=True)       # Output is multi-variate

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

#%% 2. Generator
if use_gpu is True:
    G_W1 = torch.tensor(xavier_init([Dim*2, H_Dim1]),requires_grad=True, device="cuda")     # Data + Mask as inputs (Random Noises are in Missing Components)
    G_b1 = torch.tensor(np.zeros(shape = [H_Dim1]),requires_grad=True, device="cuda")

    G_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]),requires_grad=True, device="cuda")
    G_b2 = torch.tensor(np.zeros(shape = [H_Dim2]),requires_grad=True, device="cuda")

    G_W3 = torch.tensor(xavier_init([H_Dim2, Dim]),requires_grad=True, device="cuda")
    G_b3 = torch.tensor(np.zeros(shape = [Dim]),requires_grad=True, device="cuda")
else:
    G_W1 = torch.tensor(xavier_init([Dim*2, H_Dim1]),requires_grad=True)     # Data + Mask as inputs (Random Noises are in Missing Components)
    G_b1 = torch.tensor(np.zeros(shape = [H_Dim1]),requires_grad=True)

    G_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]),requires_grad=True)
    G_b2 = torch.tensor(np.zeros(shape = [H_Dim2]),requires_grad=True)

    G_W3 = torch.tensor(xavier_init([H_Dim2, Dim]),requires_grad=True)
    G_b3 = torch.tensor(np.zeros(shape = [Dim]),requires_grad=True)

theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]


# In[65]:


#GAIN Functions


# In[66]:


#%% 1. Generator
def generator(new_x,m):
    inputs = torch.cat(dim = 1, tensors = [new_x,m])  # Mask + Data Concatenate
    G_h1 = F.relu(torch.matmul(inputs, G_W1) + G_b1)
    G_h2 = F.relu(torch.matmul(G_h1, G_W2) + G_b2)   
    G_prob = torch.sigmoid(torch.matmul(G_h2, G_W3) + G_b3) # [0,1] normalized Output
    
    return G_prob

#%% 2. Discriminator
def discriminator(new_x, h):
    inputs = torch.cat(dim = 1, tensors = [new_x,h])  # Hint + Data Concatenate
    D_h1 = F.relu(torch.matmul(inputs, D_W1) + D_b1)  
    D_h2 = F.relu(torch.matmul(D_h1, D_W2) + D_b2)
    D_logit = torch.matmul(D_h2, D_W3) + D_b3
    D_prob = torch.sigmoid(D_logit)  # [0,1] Probability Output
    
    return D_prob

#%% 3. Other functions
# Random sample generator for Z
def sample_Z(m, n):
    return np.random.uniform(0., 0.01, size = [m, n])        

# Mini-batch generation
def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


# In[67]:


#GAIN Losses


# In[68]:


def discriminator_loss(M, New_X, H):
    # Generator
    G_sample = generator(New_X,M)
    # Combine with original data
    Hat_New_X = New_X * M + G_sample * (1-M)

    # Discriminator
    D_prob = discriminator(Hat_New_X, H)

    #%% Loss
    D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) + (1-M) * torch.log(1. - D_prob + 1e-8))
    return D_loss

def generator_loss(X, M, New_X, H):
    #%% Structure
    # Generator
    G_sample = generator(New_X,M)

    # Combine with original data
    Hat_New_X = New_X * M + G_sample * (1-M)

    # Discriminator
    D_prob = discriminator(Hat_New_X, H)

    #%% Loss
    G_loss1 = -torch.mean((1-M) * torch.log(D_prob + 1e-8))
    MSE_train_loss = torch.mean((M * New_X - M * G_sample)**2) / torch.mean(M)

    G_loss = G_loss1 + alpha * MSE_train_loss 

    #%% MSE Performance metric
    MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
    return G_loss, MSE_train_loss, MSE_test_loss
    
def test_loss(X, M, New_X):
    #%% Structure
    # Generator
    G_sample = generator(New_X,M)

    #%% MSE Performance metric
    MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
    return MSE_test_loss, G_sample


# In[69]:


#Optimizers


# In[70]:


optimizer_D = torch.optim.Adam(params=theta_D)
optimizer_G = torch.optim.Adam(params=theta_G)


# In[71]:


#Training


# In[72]:


#%% Start Iterations
for it in tqdm(range(5000)):    
    
    #%% Inputs
    mb_idx = sample_idx(Train_No, mb_size)
    X_mb = trainX[mb_idx,:]  
    
    Z_mb = sample_Z(mb_size, Dim) 
    M_mb = trainM[mb_idx,:]  
    H_mb1 = sample_M(mb_size, Dim, 1-p_hint)
    H_mb = M_mb * H_mb1
    
    New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce
    
    if use_gpu is True:
        X_mb = torch.tensor(X_mb, device="cuda")
        M_mb = torch.tensor(M_mb, device="cuda")
        H_mb = torch.tensor(H_mb, device="cuda")
        New_X_mb = torch.tensor(New_X_mb, device="cuda")
    else:
        X_mb = torch.tensor(X_mb)
        M_mb = torch.tensor(M_mb)
        H_mb = torch.tensor(H_mb)
        New_X_mb = torch.tensor(New_X_mb)
    
    optimizer_D.zero_grad()
    D_loss_curr = discriminator_loss(M=M_mb, New_X=New_X_mb, H=H_mb)
    D_loss_curr.backward()
    optimizer_D.step()
    
    optimizer_G.zero_grad()
    G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = generator_loss(X=X_mb, M=M_mb, New_X=New_X_mb, H=H_mb)
    G_loss_curr.backward()
    optimizer_G.step()    
        
    #%% Intermediate Losses
    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr.item())))
        print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr.item())))
        print()


# In[73]:


#Testing


# In[74]:


Z_mb = sample_Z(Test_No, Dim) 
M_mb = testM
X_mb = testX
        
New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce

if use_gpu is True:
    X_mb = torch.tensor(X_mb, device='cuda')
    M_mb = torch.tensor(M_mb, device='cuda')
    New_X_mb = torch.tensor(New_X_mb, device='cuda')
else:
    X_mb = torch.tensor(X_mb)
    M_mb = torch.tensor(M_mb)
    New_X_mb = torch.tensor(New_X_mb)
    
MSE_final, Sample = test_loss(X=X_mb, M=M_mb, New_X=New_X_mb)
        
print('Final Test RMSE: ' + str(np.sqrt(MSE_final.item())))


# In[75]:


imputed_data = M_mb * X_mb + (1-M_mb) * Sample
print("Imputed test data:")
# np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})

if use_gpu is True:
    print(imputed_data.cpu().detach().numpy())
else:
    print(imputed_data.detach().numpy())


# In[76]:


testM


# In[77]:


testX


# In[78]:


# imputedNp=imputed_data.detach().numpy()
# np.savetxt("Imputed.csv", imputedNp, delimiter=",")


# In[79]:


textX_df = pd.DataFrame(testX, columns = ['PM2.5','NO','NO2','NOx','CO','SO2'])


# In[80]:


imputed_data_df = pd.DataFrame(imputed_data.detach().numpy(), columns = ['PM2.5','NO','NO2','NOx','CO','SO2'])


# In[81]:


textX_df=textX_df.replace(0, np.nan)


# In[82]:


textX_df


# In[83]:


imputed_data_df


# In[84]:


#Graficas


# In[85]:


from matplotlib import pyplot as plt


# In[86]:


for col in imputed_data_df.columns:
    fig, ax = plt.subplots(figsize=(20, 6))
    imputed_data_df.iloc[1500:1600].plot(y=col, use_index=True,color='red', ax=ax)
    textX_df.iloc[1500:1600].plot(y=col, use_index=True,color='blue', ax=ax)
    ax.set_title(col)
plt.show()


# In[87]:


for col in textX_df.columns:
    fig, ax = plt.subplots(figsize=(20, 6))
    textX_df.iloc[1500:1600].plot(y=col, use_index=True,color='blue', ax=ax)
    ax.set_title(col)
plt.show()


# In[ ]:




