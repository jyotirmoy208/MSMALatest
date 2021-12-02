#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utility import  Utilty
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np
from msma import MSMA
import pandas as pd
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


n_levels = 4
small_cluster_frac = 0.05


# In[3]:


def binary_tree(result,final_cluster, count,tree_dic, leaf_nodes):
    first_cluster = []
    second_cluster = []
    if count==0:
        tree_dic[count]=result
    else:
        key=max(tree_dic)+1
        tree_dic[key] = result
        if(count==n_levels):
            leaf_nodes.append(key)
    for itr in range(len(result)):
        if final_cluster[itr] == 1:
            first_cluster.append(result[itr])
        else:
            second_cluster.append(result[itr])

    if count < n_levels:
         count = count + 1
         print('first_cluster length', len(first_cluster))
         print('second_cluster length', len(second_cluster))
         obj = MSMA(2, len(first_cluster), '', 12, first_cluster)
         first_dict = obj.start_clustering()
         binary_tree(first_cluster,first_dict[max(first_dict)],count,tree_dic,leaf_nodes)
         obj = MSMA(2, len(second_cluster), '', 12, second_cluster)
         second_dict = obj.start_clustering()
         binary_tree(second_cluster,second_dict[max(second_dict)],count,tree_dic,leaf_nodes)

    else:
        return tree_dic
    
    
def get_large_cluster_centroid(small_cluster_threshold,tree_dic,leaf_nodes):
    merged_clusters = {}
    mergeditr = 0
    for key in tree_dic:
        if  key in leaf_nodes :
            if len(tree_dic[key]) > small_cluster_threshold:
               merged_clusters[mergeditr] = tree_dic[key]
               mergeditr = mergeditr + 1
    final_array = []
    count = 0
    for key in merged_clusters:
        for itr in range(len(merged_clusters[key])):
            array = merged_clusters[key]
            final_array.insert(count, array[itr])
            count = count + 1
    print('large clusters no', len(final_array))
    centroid = np.mean(final_array, axis=0)
    return centroid



def get_large_cluster_center(small_cluster_threshold,tree_dic,leaf_nodes):
    merged_clusters = {}
    centroid_array=[]
    mergeditr = 0
    for key in tree_dic:
        if key in leaf_nodes:
            if len(tree_dic[key]) > small_cluster_threshold:
                merged_clusters[mergeditr] = tree_dic[key]
                mergeditr = mergeditr + 1

    count = 0
    for key in merged_clusters:
        centroid = np.mean(merged_clusters[key], axis=0)
        centroid_array.insert(count,centroid)
        count = count + 1
    '''this is returning the cluster centers of all the large clusters'''
    return centroid_array


def get_large_clusters_anomalyscore(tree_dic,small_cluster_threshold, leaf_nodes):
    large_clusters={}
    anomaly_largeclusters={}
    for key in tree_dic:
        if key in leaf_nodes:
           if len(tree_dic[key])>small_cluster_threshold:
               large_clusters[key]=tree_dic[key]

    for key in large_clusters:
        array = large_clusters[key]
        centroid = np.mean(array, axis=0)
        for itr in range(len(large_clusters[key])):
            anomalyscore_largeclusters=np.linalg.norm(array[itr]-centroid)
            anomaly_largeclusters[anomalyscore_largeclusters]=array[itr]

    return anomaly_largeclusters


# In[4]:


util = Utilty()


'''reading data'''
result = util.read_file("processed_dataset_D1.csv",",")
obj = MSMA(2, len(result), '', 12,result)
print('dataset shape',result.shape)


'''obtaining best population sample'''
final_dict = obj.start_clustering()
print('final dict')
print(final_dict,'\n')

final_cluster=final_dict[max(final_dict)]


'''running recursive algorithm'''
print('running binary tree\n')
count = 0
leaf_nodes = []
tree_dic={}

a = binary_tree(result,final_cluster,count,tree_dic, leaf_nodes)
print('\ntree built\n\n')

print('leaf nodes')
'''finding minkey and minarray'''
minkey=0
total_cluster_length=0
minarray=tree_dic[minkey]
for key in tree_dic:
    if key in leaf_nodes:
      total_cluster_length=total_cluster_length+len(tree_dic[key])
      print('length',len(tree_dic[key]))
    if len(tree_dic[key])<len(minarray):
        minkey=key
        minarray=tree_dic[key]
print('\n\nminkey',minkey)
print('minarry len',len(minarray))


# In[5]:



small_cluster_threshold=math.floor(small_cluster_frac*len(result))

anomalyscore_dict={}
anomalyscore_from_largeclusters={}
merged_clusters={}

'''merged clusters dictionary of small clusters'''
mergeditr=0
for key in tree_dic:
    if len(tree_dic[key])<=small_cluster_threshold:
        merged_clusters[mergeditr]=tree_dic[key]
        mergeditr=mergeditr+1
print('no of small clusters',len(merged_clusters))

    
'''final array of all elements in small clusters'''
final_array=[]
count=0
for key in merged_clusters:
    # print(key)
    for itr in range(len(merged_clusters[key])):
        array=merged_clusters[key]
        final_array.insert(count,array[itr])
        count=count+1
print('small clusters merged size',len(final_array))


'''get centroids of all large clusters'''
centroid_array=get_large_cluster_center(small_cluster_threshold,tree_dic,leaf_nodes)
print('no of large clusters',len(centroid_array))


'''calculate cblof of small cluster datapoints'''
cblof={}
for anoitr in range(len(final_array)):
    min=math.inf
    for clusitr in range(len(centroid_array)):

     cblof_dist=np.linalg.norm(final_array[anoitr]-centroid_array[clusitr])
     if cblof_dist<min:
         min=cblof_dist
     if  clusitr==len(centroid_array)-1:
         cblof[min] = final_array[anoitr]


'''calculate cblof of large cluster datapoints'''
cblof_largecluster=get_large_clusters_anomalyscore(tree_dic,small_cluster_threshold,leaf_nodes)

final_merged_cblof={**cblof,**cblof_largecluster}

print('final anomaly score dict')
cnt = 0
for key in sorted(final_merged_cblof,reverse=True):
         print(key,"--", final_merged_cblof[key])
         cnt+=1
         if(cnt==5):
             break


# In[6]:


'''labelled dataset with anomaly'''
df_labels = pd.read_csv('pdatalabel.csv',header=None)
df_labels.sort_values(by=[0,1],axis=0,inplace = True)
df_labels.reset_index(drop=True, inplace=True)
df_labels.rename(columns={10:'anomaly'},inplace=True)


# In[7]:


'''output anomaly score'''
final_list = []
for key in final_merged_cblof:
    curlist = final_merged_cblof[key].tolist()
    curlist.append(key)
    final_list.append(curlist)
df_score = pd.DataFrame(final_list)
df_score.sort_values(by = [0,1],axis=0,inplace=True)
df_score.reset_index(drop=True,inplace=True)
df_score.rename(columns={10:'anomalyscore'},inplace=True)


# In[8]:


df_labels_score = pd.concat([df_score,df_labels['anomaly']],axis = 1)
df_labels_score.sort_values(by=['anomalyscore'],ascending = False,inplace=True)
df_labels_score.reset_index(drop=True,inplace=True)


# In[9]:


anomaly_ranks = []
for i in range(df_labels_score.shape[0]):
    if(df_labels_score['anomaly'][i]==1):
        anomaly_ranks.append(i+1)


# In[10]:


print(anomaly_ranks)


# In[ ]:


df_labels_score.to_csv('output.csv',index=False)


# In[ ]:




