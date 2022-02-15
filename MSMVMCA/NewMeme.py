import numpy as np

import math
import Entropy as Entrophy
import random as rand
from sklearn.cluster import KMeans
from Kmediods import kMedoids,convertLabelsToList
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score

from sklearn.metrics import confusion_matrix
from scipy.special  import comb
from sklearn.metrics import jaccard_similarity_score

def logIt(n):
    x=np.exp(n)/(1+np.exp(n))**2
    return x
def fintness(wieghtMatrix,result):
    transresult = np.matrix(np.transpose(result))
    multresult = np.matrix(wieghtMatrix) * np.matrix(result)
    diagelements = multresult.diagonal()
    #print("attribute",transresult[0])
    #print(multresult)
    #print(diagelements)
    diagarray = np.squeeze(np.asarray(diagelements))
    informationlist = {}
    count = 0
    for data in diagarray:
        feature=transresult[count]
        featureTranspose=np.transpose(feature)
        multoffeature=feature*featureTranspose
        informationlist[count] = np.log(logIt(data)*multoffeature)
        # informationlist.insert(count,logIt(data))
        count = count + 1
    #print(informationlist)
    return informationlist
def calculateInformation(wieghtMatrix,result):
    finaldict={}
    informationlist=fintness(wieghtMatrix,result)
    variancelist=[]
    varcount=0
    beakflag=False
    #transresult=np.transpose(result)
    #multresult=np.matrix(wieghtMatrix)*np.matrix(transresult)
    #diagelements=multresult.diagonal()
    #print(multresult)
    #print(diagelements)
    #diagarray=np.squeeze(np.asarray(diagelements))
    #informationlist={}
    #count=0;
    #for data in diagarray:
        #informationlist[count]=logIt(data);
        #informationlist.insert(count,logIt(data))
        #count=count+1
   # print(informationlist)
    #parents=[]
    sortedDict= sortDictionar(informationlist)

    for k in range(1):

      for outerstage in range(50):

        elitismRate = math.floor(len(sortedDict) / 2)
        #print(sortedDict)
        counter = 0
        parenDict = {}
        survivorDict = {}
        survivorweightdict={}
        for key in sortedDict:
            if counter < elitismRate:
                 parenDict[key] = sortedDict[key]
            else:
                survivorDict[key] = sortedDict[key]
            counter = counter + 1
        for key in survivorDict:
            survivorweightdict[key]=list(wieghtMatrix[key])
        childDict=getBreeding(parenDict,wieghtMatrix)
        mutantDict=getMutation(childDict)
        if len(childDict)==0 | len(mutantDict)==0 | beakflag==True:
            break
        #print('mutant',len(mutantDict))
        newpopulation=mergeDicts(mutantDict,survivorweightdict)
        #print('merged',newpopulation)
        newpopluationlist=convertDictToList(newpopulation)
        informationlist = fintness(newpopluationlist, result)
        varinfolist = convertDictToList(informationlist)
        variancelist.insert(varcount,np.std(varinfolist))
        varcount=varcount+1
        sortedDict = sortDictionar(informationlist)
        newpopcounter=0
        updateWeightMatrix(newpopulation,wieghtMatrix)



        for innerstage in range(45):
            elitismRate = math.floor(len(sortedDict) / 2)
            counter = 0
            parenDict = {}
            survivorDict = {}
            survivorweightdict={}
            for key in sortedDict:
                if counter < elitismRate:
                    parenDict[key] = sortedDict[key]
                else:
                    survivorDict[key] = sortedDict[key]
                counter = counter + 1
            for key in survivorDict:
                survivorweightdict[key] = list(wieghtMatrix[key])
            childDict = getBreedingInner(parenDict, wieghtMatrix)
            mutantDict = getMutationInner(childDict)
            #print('mutant', len(mutantDict))
            newpopulation = mergeDicts(mutantDict, survivorweightdict)
            newpopluationlist = convertDictToList(newpopulation)
            informationlist = fintness(newpopluationlist, result)
            varinfolist = convertDictToList(informationlist)
            variancelist.insert(varcount, np.std(varinfolist))
            varcount = varcount + 1
            sortedDict = sortDictionar(informationlist)
            # verify convergence if the last two standard deviation value of fitness list is less than .005
            # we assume the the population has convered so stop the iteration otherwise continue
            if (len(variancelist)) > 2:
                stdlength = len(variancelist)
                lastStd = variancelist[stdlength - 1]
                sndLastStd = variancelist[stdlength - 2]
                thrdLastStd = variancelist[stdlength - 3]
                if abs(lastStd - sndLastStd) < 0.005 and abs(sndLastStd - thrdLastStd) < 0.005:
                    beakflag=True
                    break
            updateWeightMatrix(newpopulation, wieghtMatrix)


            #print('sorted inner dict',sortedDict)
    #for count in range(math.floor(len(informationlist)/2)):
     #   parents.insert(max(informationlist.items(), key=operator.itemgetter(1))[0])
    print(variancelist)
    print(len(variancelist))
    return sortedDict
def updateWeightMatrix(newpopluationlist,wieghtMatrix):
    for i in range(0, wieghtMatrix.shape[0]):
        for key in newpopluationlist:
            if i==key:
                wieghtMatrix[i]=newpopluationlist[key]
                break

def convertDictToList(dict):
    #list=np.ndarray(shape=(60,60),dtype=int)
    #list=[np.float(0.0)]*208
    #list=[list]*60
    list=[np.zeros(60,int)]*len(dict)
   # sorteddict = sorted(dict.items(), key=lambda s: s[0])
    for key in dict:
        list.pop(key)
        list.insert(key,dict[key])
       # list.pop(key)
       # list.insert(key,dict[key])

    return list

def mergeDicts(x,y):
    z=x.copy()
    z.update(y)
    return z
def getMutation(childDict):


    for key in childDict:
        mutantcount=0;
        singlechild=childDict[key]
        mutantThreshold = math.floor(len(singlechild) * .25)
        mutantList = []
        for key1 in range(len(singlechild)):
            if mutantcount<mutantThreshold:
              mutantList.insert(key1,rand.randint(0,1))
            else:
                mutantList.insert(key1,singlechild[key1])
            mutantcount=mutantcount+1
        childDict[key]=mutantList
    return childDict

def getMutationInner(childDict):


    for key in childDict:
        mutantcount=0;
        singlechild=childDict[key]
        mutantThreshold = math.floor(len(singlechild) * .25*3)
        mutantList = []
        for key1 in range(len(singlechild)):
            if mutantcount<mutantThreshold:
                mutantList.insert(key1, singlechild[key1])
            else:
                mutantList.insert(key1, rand.randint(0, 1))

            mutantcount=mutantcount+1
        childDict[key]=mutantList
    return childDict
def getBreeding(parenDict,weightGenePool):

    childrenDict={}
    father={}
    mother={}
    iterator=0;
    for key in parenDict:
        if iterator==0:
            father[key]=weightGenePool[key]
        elif iterator%2==0:
            father[key]=weightGenePool[key]
        else:
            mother[key]=weightGenePool[key]
        iterator=iterator+1;
    crossoverThreshold=math.floor(len(weightGenePool)/4)
    for key1 in list(father):
        if key1 in father:
            fathchromo=father[key1]
            father.pop(key1)
            for key2 in list(mother):
                if key2 in mother:
                    motherchromo=mother[key2]
                    mother.pop(key2)
                    breedingCount=0
                    firstchildDict = []
                    secondchildDict = []
                    for key in range(len(fathchromo)):
                      if breedingCount<crossoverThreshold:
                        firstchildDict.insert(key,fathchromo[key])
                        secondchildDict.insert(key,motherchromo[key])
                      else:
                        firstchildDict.insert(key,motherchromo[key])
                        secondchildDict.insert(key,fathchromo[key])
                      breedingCount=breedingCount+1

                childrenDict[key1]=firstchildDict
                childrenDict[key2] = secondchildDict
                break

    return childrenDict
def getBreedingInner(parenDict,weightGenePool):

    childrenDict={}
    father={}
    mother={}
    iterator=0;
    for key in parenDict:
        if iterator==0:
            father[key]=weightGenePool[key]
        elif iterator%2==0:
            father[key]=weightGenePool[key]
        else:
            mother[key]=weightGenePool[key]
        iterator=iterator+1;
    crossoverThreshold=math.floor(len(weightGenePool)/4)
    for key1 in list(father):
        if key1 in father:
            fathchromo=father[key1]
            father.pop(key1)
            for key2 in list(mother):
                if key2 in mother:
                    motherchromo=mother[key2]
                    mother.pop(key2)
                    breedingCount=0
                    firstchildDict = []
                    secondchildDict = []
                    for key in range(len(fathchromo)):
                      if breedingCount<crossoverThreshold:
                        firstchildDict.insert(key,motherchromo[key])
                        secondchildDict.insert(key,fathchromo[key])
                      else:
                        firstchildDict.insert(key,fathchromo[key])
                        secondchildDict.insert(key,motherchromo[key])
                      breedingCount=breedingCount+1

                childrenDict[key1]=firstchildDict
                childrenDict[key2] = secondchildDict
                break

    return childrenDict
def sortDictionar(informationDict):
    parentDict={}
    for itr in range(len(informationDict)):
        max = -math.inf;
        parent="";
        for key in informationDict:
            if informationDict[key]>max:
                max=informationDict[key]
                parent=key
        if parent!="" :
            parentDict[parent]=max
            informationDict.pop(parent)
    #copy allthe key having 0 value
    for key in informationDict:
        parentDict[key]=informationDict[key]
    return parentDict
def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)
if __name__ == '__main__':
    rand.random(0.0001,0.0010)
    result= readCsvFile()
    result=result.transpose()
    D = pairwise_distances(result, metric='euclidean')
    M, C = kMedoids(D, 2)
    kmediodlabels=convertLabelsToList(M,C)
    print("lables k mediod",kmediodlabels)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(result)
    print(kmeans.labels_)
    test_labels = np.zeros(208,int)
    test_labels[0:97] = np.int(1)
    print("kmeans", confusion_matrix(test_labels, kmeans.labels_))
    randkmean=rand_index_score(test_labels,kmeans.labels_)
    jacardkmean=jaccard_similarity_score(test_labels,kmeans.labels_)
    siltkmean =silhouette_score(D,kmeans.labels_,"precomputed")
    print("kemans rand",randkmean)
    print("jacardkmean",jacardkmean)
    print("silthkmean", siltkmean)
    entrophygen=Entrophy.computeEntophy(test_labels,kmeans.labels_)
    print('entrophy',entrophygen)

    randkmediod = rand_index_score(test_labels, kmediodlabels)
    jacardkmediod = jaccard_similarity_score(test_labels, kmediodlabels)
    print("kemediod rand", randkmediod)
    print("jacardkemediod", jacardkmediod)
    siltkmediod = silhouette_score(D, kmediodlabels, "precomputed")
    entrophykmed = Entrophy.computeEntophy(test_labels, kmediodlabels)
    print("silthkmediod", siltkmediod)
    print('entrophy kemediod', entrophykmed)




    result1 = np.loadtxt(open("C:\personal\PhD\Dataset\\sonar-data-set\\sonar.all-data.csv", "r"), delimiter=",")
    kmeans1 = KMeans(n_clusters=2, random_state=0).fit(result1)
    print("kmeans1", confusion_matrix(test_labels, kmeans1.labels_))
    randkmean1 = rand_index_score(test_labels, kmeans1.labels_)
    jacardkmean1 = jaccard_similarity_score(test_labels, kmeans1.labels_)
    siltkmean1 = silhouette_score(D, kmeans1.labels_, "precomputed")
    print("kemans rand1", randkmean1)
    print("kmeans jacard1",jacardkmean1)

    entrophy1 = Entrophy.computeEntophy(test_labels, kmeans1.labels_)
    print('entrophy1', entrophy1)
    D1 = pairwise_distances(result1, metric='euclidean')
    siltkmean1 = silhouette_score(D1, kmeans1.labels_, "precomputed")
    print("silthkmean1", siltkmean1)
    M1, C1 = kMedoids(D1, 2)
    kmediodlabels1 = convertLabelsToList(M1, C1)


    randkmediod1 = rand_index_score(test_labels, kmediodlabels1)
    jacardkmediod1 = jaccard_similarity_score(test_labels, kmediodlabels1)
    siltkmediod1 = silhouette_score(D1, kmediodlabels1, "precomputed")
    print("kemediod rand", randkmediod1)
    print("jacardkemediod", jacardkmediod1)
    print("siltkmediod1", siltkmediod1)
    entrophykmed1 = Entrophy.computeEntophy(test_labels, kmediodlabels1)
    print('entrophy kemediod', entrophykmed1)