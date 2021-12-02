import numpy as np
import pandas as pd
import math
from ast import literal_eval
from scipy.spatial.distance import pdist, euclidean
import random as rand
class Utilty:
    '''This method returns the centroid of the normal parameters. AS we consider large clusters are normal datapoints
    here we merged all the datapoints and created a single population. This function then returns the centroid of the population'''

    def get_large_cluster_centroid(self,small_cluster_threshold, tree_dic):
        merged_clusters = {}
        mergeditr = 0

        for key in tree_dic:
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
        '''merged_clustersarray=util.convert_dict_to_list(merged_clusters,len(merged_clusters),len(tree_dic[0]))'''
        print('merged_clusters', merged_clusters)
        centroid = np.mean(final_array, axis=0)
        return centroid

    def get_large_cluster_center(self,small_cluster_threshold, tree_dic):
        merged_clusters = {}
        centroid_array = []
        mergeditr = 0

        for key in tree_dic:
             if len(tree_dic[key]) > small_cluster_threshold:
                    merged_clusters[mergeditr] = tree_dic[key]
                    mergeditr = mergeditr + 1

        count = 0
        for key in merged_clusters:
            centroid = np.mean(merged_clusters[key], axis=0)
            centroid_array.insert(count, centroid)
            count = count + 1
        '''this is returning the cluster centers of all the large clusters'''

        return centroid_array

    def get_large_clusters_anomalyscore(self,tree_dic, small_cluster_threshold):

        large_clusters = {}
        anomaly_largeclusters = {}
        for key in tree_dic:

                if len(tree_dic[key]) >= small_cluster_threshold:
                    large_clusters[key] = tree_dic[key]

        for key in large_clusters:
            array = large_clusters[key]
            centroid = np.mean(array, axis=0)
            for itr in range(len(large_clusters[key])):
                anomalyscore_largeclusters = np.linalg.norm(array[itr] - centroid)
                if anomalyscore_largeclusters in anomaly_largeclusters.keys():
                    print("true")
                    anomaly_largeclusters[(anomalyscore_largeclusters+rand.random())] = array[itr]

                anomaly_largeclusters[anomalyscore_largeclusters] = array[itr]

        return anomaly_largeclusters
    def read_file(self,location,delimiter):
        result = np.loadtxt(open(location, "r"),
                            delimiter=delimiter)
        '''result=pd.read_csv(location,sep=",", engine='python')

        result=result.values
        result=literal_eval(result)'''
        return result

    def DaviesBouldin(self,X, labels):
        n_cluster = 2
        cluster_k = [X[labels == k] for k in range(n_cluster)]
        centroids = [np.mean(k, axis=0) for k in cluster_k]
        variances = [np.mean([euclidean(p, centroids[i]) for p in k]) for i, k in enumerate(cluster_k)]
        db = []

        for i in range(n_cluster):
            for j in range(n_cluster):
                if j != i:
                    db.append((variances[i] + variances[j]) / euclidean(centroids[i], centroids[j]))

        return (np.max(db) / n_cluster)

    def sortDictionary(self,informationDict):
        sorted_dict = {}
        for itr in range(len(informationDict)):
            max = -math.inf;
            parent = "";
            for key in informationDict:
                if key > max:
                    max = key
                    parent = key
            if parent != "":
                sorted_dict[parent] = informationDict[max]
                informationDict.pop(parent)

        return sorted_dict
    '''in case of davis bouldin lower value is better seperation so lower value should take the first spot
    in the dictionary'''
    def sortDictionaryForDavis(self,informationDict):
        sorted_dict = {}
        for itr in range(len(informationDict)):
            min = math.inf;
            parent = "";
            for key in informationDict:
                if key < min:
                    min = key
                    parent = key
            if parent != "":
                sorted_dict[parent] = informationDict[min]
                informationDict.pop(parent)

        return sorted_dict


    def merge_dicts(self,x, y):
        z = x.copy()
        z.update(y)
        return z

    def convert_dict_to_list(self,dict,version_number,datapoint_number):
        # list=np.ndarray(shape=(60,60),dtype=int)
        # list=[np.float(0.0)]*208
        # list=[list]*60

        population = np.ones((version_number, datapoint_number))
        # sorteddict = sorted(dict.items(), key=lambda s: s[0])
        count=0
        for key in dict:

            population[count]=dict[key]
            count=count+1
            # list.pop(key)
            # list.insert(key,dict[key])

        return population

    def checkEqual(self,iterator):
        return len(set(iterator)) <= 1