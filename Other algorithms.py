#!/usr/bin/env python
# coding: utf-8


import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans


df = pd.read_csv('Data/Processed Dataset.csv')
df = df.drop(columns = ['P. Name'])


wcss = []
for i in range(1,25):
    kmeans = KMeans(n_clusters=i,random_state=100)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,25),wcss)
plt.show()

kmeans = KMeans(n_clusters=12,random_state=0)
kmeans.fit(df)

labels = kmeans.labels_.tolist()
counts = pd.Series(kmeans.labels_).value_counts()
print('cluster sizes\n',counts)

small_cluster_threshold = 95
small_cluster_list = []
large_cluster_list = []
for index,value in counts.items():
    if(value<=small_cluster_threshold):
        small_cluster_list.append(index)
    else:
        large_cluster_list.append(index)

print('small cluster list',small_cluster_list)
print('large cluster list',large_cluster_list)

centroid_array = []
centroid_dict = {}
for i in large_cluster_list:
    centroid_array.append(kmeans.cluster_centers_[i])
    centroid_dict[i] = kmeans.cluster_centers_[i]

small_cluster = []
large_cluster = []
for i in range(len(kmeans.labels_)):
    if labels[i] in small_cluster_list:
        small_cluster.append(df.iloc[i,:])
    else:
        large_cluster.append(df.iloc[i,:])

print('small cluster total',len(small_cluster))
print('large cluster total',len(large_cluster))


'''calculate cblof of small cluster datapoints'''
cblof={}
for anoitr in range(len(small_cluster)):
    min=math.inf
    for clusitr in range(len(centroid_array)):

     cblof_dist=np.linalg.norm(small_cluster[anoitr]-centroid_array[clusitr])
     if cblof_dist<min:
         min=cblof_dist
     if  clusitr==len(centroid_array)-1:
         cblof[min] = small_cluster[anoitr]

'''calculate cblof of large cluster datapoints'''
for i in range(len(labels)):
    if labels[i] in large_cluster_list:
        dist = np.linalg.norm(df.iloc[i,:]-centroid_dict[labels[i]])
        cblof[dist] = df.iloc[i,:]

'''labelled dataset with anomaly'''
df_labels = pd.read_csv('pdatalabelled.csv')

df_labels.sort_values(by=['PC1','PC2'],axis=0,inplace = True)
df_labels.reset_index(drop=True, inplace=True)

cols = ['PC' + str(x) for x in range(1,11)]
cols += ['Anomaly Score']


'''output anomaly score'''
final_list = []
for key in cblof:
    curlist = cblof[key].tolist()
    curlist.append(key)
    final_list.append(curlist)
df_score = pd.DataFrame(final_list,columns=cols)
df_score.sort_values(by = ['PC1','PC2'],axis=0,inplace=True)
df_score.reset_index(drop=True,inplace=True)


df_labels_score = pd.concat([df_labels['P. Name'],df_score,df_labels['Anomaly'],df_labels['P. Habitable Class']],axis = 1)

df_labels_score.sort_values(by=['Anomaly Score'],ascending = False,inplace=True)
df_labels_score.reset_index(drop=True,inplace=True)

anomaly_ranks = []
for i in range(df_labels_score.shape[0]):
    if(df_labels_score['Anomaly'][i]==1):
        anomaly_ranks.append(i+1)

print('anomaly_ranks')
print(anomaly_ranks)
