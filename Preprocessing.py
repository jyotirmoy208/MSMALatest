#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib notebook


# In[113]:


df = pd.read_csv("unprocessed_dataset_D1.csv",index_col = None)


# In[114]:


df.head()


# In[115]:


df['P. Habitable Class'].value_counts()


# In[116]:


df['P. Habitable Class'] = df['P. Habitable Class'].replace({'non-habitable':0,'mesoplanet':1,'psychroplanet':1,'thermoplanet':1,'hypopsychroplanet':1})


# In[117]:


df['P. Habitable Class'].value_counts()


# In[118]:


df_dropped = df.dropna(thresh=1600,axis=1)


# In[119]:


df_dropped.info()


# In[120]:


cols = df_dropped.columns.tolist()


# In[121]:


cols_initial = df.columns.tolist()


# In[122]:


cols_dropped = list(set(cols_initial).difference(cols))


# In[123]:


cols_dropped


# In[124]:


irrelevant_cols = ['Index','P. Name','P. Habitable Class','P. Eccentricity','S. No. Planets','S. No. Planets HZ','P. SPH','P. HZD','P. HZC','P. HZA','P. HZI']


# In[125]:


df_dropped2 = df_dropped.drop(labels = irrelevant_cols, axis=1)


# In[126]:


df_dropped2.info()


# In[127]:


df_dropped2['Imputed Eccentricity'].corr(df_dropped2['Imputed Eccentricity (EU)'])


# In[128]:


df_dropped2 = df_dropped2.drop(labels = ['Imputed Eccentricity (EU)'],axis=1)


# In[129]:


df_dropped2.info()


# In[130]:


pd.Series(index=df_dropped2.columns, data=np.count_nonzero(df_dropped2, axis=0))


# In[131]:


df_dropped2 = df_dropped2.drop(labels=['P. Int ESI','P. Surf ESI','P. Omega (deg)'],axis = 1)


# In[132]:


df_dropped2.info()


# In[133]:


corr = df_dropped2.corr()


# In[134]:


corr.style.background_gradient(cmap='coolwarm')


# In[135]:


high_corr_cols = ['P. SFlux Min (EU)','P. SFlux Max (EU)','P. Teq Min (K)','P. Teq Max (K)','P. Ts Min (K)','P. Ts Max (K)','P. Ts Mean (K)','P. Appar Size (deg)','P. Mean Distance (AU)','S. Hab Zone Max (AU)']


# In[136]:


df_dropped3 = df_dropped2.drop(labels = high_corr_cols, axis = 1)


# In[137]:


df_dropped3.info()


# In[138]:


df_dropped3 = df_dropped3.fillna(df_dropped3.mean())


# In[139]:


df_dropped3.info()


# In[140]:


corr = df_dropped3.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[142]:


df_dropped3.head(10)


# In[143]:


df_dropped3.describe()


# In[144]:


from sklearn.preprocessing import StandardScaler


# In[145]:


x = StandardScaler().fit_transform(df_dropped3)


# In[146]:


x.shape


# In[147]:


n_components = 10
from sklearn.decomposition import PCA
pca = PCA(n_components,random_state=100)
principalComponents = pca.fit_transform(x)
# principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2', 'principal component 3'])


# In[163]:


# plt.plot(range(n_components), pca.explained_variance_ratio_)
# plt.plot(range(n_components), np.cumsum(pca.explained_variance_ratio_))
# plt.title("Component-wise and Cumulative Explained Variance")


# In[153]:


# np.cumsum(pca.explained_variance_ratio_)


# In[154]:


final_data = pd.DataFrame(principalComponents)


# In[157]:


final_data.to_csv('processed_dataset_D1.csv',index=False,header=False)


# In[158]:


final_data_anomaly = pd.concat([final_data, df['P. Habitable Class']], axis = 1)


# In[162]:


final_data_anomaly.to_csv('processed_dataset_labelled_D1.csv',index=False,header=False)


# In[ ]:




