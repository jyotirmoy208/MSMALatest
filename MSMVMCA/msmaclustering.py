import numpy as np
from utility import  Utilty
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score
import random as rand
import math
import dunnscore
import NewMeme as newmeme
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import jaccard_similarity_score
from Kmediods import kMedoids,convertLabelsToList
class MSMA:
    def __init__(self,cluster_number,datapoint_number,data_location,version_number,result):
          self.cluster_number=cluster_number
          self.datapoint_number=datapoint_number
          self.data_location=data_location
          self.version_number=version_number
          self.result=result
    '''This function creates the initial population in a matrix form where every row is a
    version and the number of column is nothing but the number of data points and the value indicate the cluster number'''
    def create_population(self):
        n=math.floor(214/6);
        '''creating matrix of 1s'''
        population=np.ones((self.version_number,self.datapoint_number))
        for row in range(self.version_number):
            #for col in range(self.datapoint_number):
                test = np.zeros(214, int)
                x=rand.randint(0,n)
                test[0:x] = np.int(1)
                y = rand.randint(0, n)
                test[x:y+x] = np.int(2)
                z = rand.randint(0, n)
                test[y+x:y + x+z] = np.int(3)
                u = rand.randint(0, n)
                test[y + x+z:y + x + z+u] = np.int(4)
                v = rand.randint(0, n)
                test[y + x + z+u:y + x + z + u+v] = np.int(4)
                w = rand.randint(0, n)
                test[y + x + z + u+v:y + x + z + u + v+w] = np.int(5)
                population[row]=test
                #population[row][col]=rand.randint(0,5)

        ''' print(population)'''

        return population

    def calculate_silhoutte(self,location,labels):
        util=Utilty()
        self.result=util.read_file("C:\personal\PhD\Dataset\sonar-data-set\glass-classification\\glass.csv",",")
        '''print('result',result.shape[0])'''
        D = pairwise_distances(self.result, metric='euclidean')
        '''print('result', D)'''
        '''labels=np.ones(result.shape[0])
        labels[0:300]=0'''
        print('silhoutte labels', labels)
        silhoute=silhouette_score(D, labels, "precomputed")
        '''print('silhoutte labels', labels)
        print('silhoutte score',silhoute)'''
        return silhoute
    def get_crossover(self,parent_dict):
        '''below logic dividing the parents into father and mother
        0th and even elements are father and odd events are mother'''
        father = {}
        mother = {}
        iterator = 0;
        for key in parent_dict:
            if iterator == 0:
                father[key] = parent_dict[key]
                crossoverThreshold = math.floor(len(parent_dict[key]) / 4)
            elif iterator % 2 == 0:
                father[key] = parent_dict[key]
            else:
                mother[key] = parent_dict[key]
            iterator = iterator + 1;
            '''this is the number of genes will be exchanged between the father and mother'''

        '''the child will be keeping inside this dictionary after the crossover between father and mother'''
        children_dict={}
        for key1 in list(father):
            if key1 in father:
                '''first chromosome is reading from father'''
                fathchromo = father[key1]
                father.pop(key1)
                for key2 in list(mother):
                    if key2 in mother:
                        '''second chromosome is reading from mother'''
                        motherchromo = mother[key2]
                        mother.pop(key2)
                        breedingCount = 0
                        firstchildDict = []
                        secondchildDict = []
                        for key in range(len(fathchromo)):
                            if breedingCount < crossoverThreshold:
                                firstchildDict.insert(key, fathchromo[key])
                                secondchildDict.insert(key, motherchromo[key])
                            else:
                                firstchildDict.insert(key, motherchromo[key])
                                secondchildDict.insert(key, fathchromo[key])
                            breedingCount = breedingCount + 1

                    children_dict[key1] = firstchildDict
                    children_dict[key2] = secondchildDict
                    break

        return children_dict

    def get_mutation(self,childDict):

        for key in childDict:
            mutantcount = 0;
            singlechild = childDict[key]
            mutantThreshold = math.floor(len(singlechild) * .25)
            mutantList = []
            for key1 in range(len(singlechild)):
                if mutantcount < mutantThreshold:
                    mutantList.insert(key1, rand.randint(0, 5))
                else:
                    mutantList.insert(key1, singlechild[key1])
                mutantcount = mutantcount + 1
            childDict[key] = mutantList
        return childDict

    def get_silhoutte_score(self,population):
        silhoutte_dict={}
        for row in range(len(population)):
            score=self.calculate_silhoutte("",population[row])
            silhoutte_dict[score]=population[row]
        return silhoutte_dict
    def calculate_dunn(self,location,labels):
        utility=Utilty()
        D = pairwise_distances(self.result, metric='euclidean')
        '''print("labels bouldin",labels)'''
        dunn_score = davies_bouldin_score(D,labels)
        '''print("labels bouldin", labels)
        print("bouldin score", dunn_score)'''
        return dunn_score

    def get_dunn_score(self, population):
        dunnscore_dict = {}
        for row in range(len(population)):
            score = self.calculate_dunn("", population[row])
            dunnscore_dict[score] = population[row]
        return dunnscore_dict
    def delete_identical_row(self,iterator):
        util = Utilty()
        '''print('before deletion',iterator)'''
        to_delete=[]
        for i in range(len(iterator)):
           is_identical= util.checkEqual(iterator[i])
           '''print('is_identical',is_identical)'''

           if is_identical==True:
               print('is_identical is true need to delete',i)
               to_delete.append(i)
        iterator = np.delete(iterator, to_delete, axis=0)
        '''print('after deletion',iterator)'''
        return iterator

    def start_clustering(self):
        '''this dict contains the max silhoutte version of each iteration'''
        final_dict={}
        '''created the population randomly'''
        population = self.create_population()
        '''deletion chromosome if all the elements are identical'''
        print('new random created population',population)
        population=self.delete_identical_row(population)
        '''below loop callculate the silhoutte score for each version'''
        silhoutte_dict = self.get_silhoutte_score(population)
        util = Utilty()
        '''sort the population based on their score in a dictionary, where '''
        sorted_dict = util.sortDictionary(silhoutte_dict)
        '''print('random dict',sorted_dict)'''
        for outerstage in range(100):
         counter = 0
         parent_dict = {}
         survivor_dict = {}
         elitismRate = math.floor(len(sorted_dict) / 2)
         if elitismRate%2==1:
             print('elitismRate',elitismRate)
             elitismRate=elitismRate+1

         '''print('sorted dict before outer iteration',len(sorted_dict))'''
         for key in sorted_dict:
            if counter < elitismRate:
                parent_dict[key] = sorted_dict[key]
            else:
                survivor_dict[key] = sorted_dict[key]
            counter = counter + 1
         offspring_dict=self.get_crossover(parent_dict)
         mutant_dict=self.get_mutation(offspring_dict)
         if len(offspring_dict)==0 | len(mutant_dict)==0:
            print('cannot proceed further as the offspring and mutant list is empty')
         else:
            newpopulation=util.merge_dicts(mutant_dict,survivor_dict)
            '''print('merged newpopulation',len(newpopulation))'''
            newpopluation_array = util.convert_dict_to_list(newpopulation,len(newpopulation),self.datapoint_number)
            newpopluation_array=self.delete_identical_row(newpopluation_array)
            '''print('merged newpopulation array', len(newpopulation))
            print('merged newpopulation arrayfull', len(newpopulation))'''
            silhoutte_dict=self.get_silhoutte_score(newpopluation_array)

            sorted_dict = util.sortDictionary(silhoutte_dict)
            print('max val',max(sorted_dict))
            final_dict[max(sorted_dict)]=sorted_dict[max(sorted_dict)]
            for innerstage in range(40):
                parent_dict = {}
                survivor_dict = {}
                counter=0
                elitismRate = math.floor(len(sorted_dict) / 2)
                '''print('inner stage sortedict length',len(sorted_dict))'''
                if elitismRate%2==1:
                    elitismRate=elitismRate+1
                    print('elitismRate',elitismRate)
                for key in sorted_dict:
                    if counter < elitismRate:
                        parent_dict[key] = sorted_dict[key]
                    else:
                        survivor_dict[key] = sorted_dict[key]
                    counter = counter + 1
                offspring_dict = self.get_crossover(parent_dict)
                mutant_dict = self.get_mutation(offspring_dict)
                if len(offspring_dict) == 0 | len(mutant_dict) == 0:
                    print('cannot proceed further as the offspring and mutant list is empty')
                else:
                  newpopulation = util.merge_dicts(mutant_dict, survivor_dict)
                  '''print('inner stage after mergin',len(newpopulation))
                  print('inner stage after mergin full', newpopulation)'''
                  newpopluation_array = util.convert_dict_to_list(newpopulation, len(newpopulation),self.datapoint_number)
                  newpopluation_array = self.delete_identical_row(newpopluation_array)
                  '''print('inner stage after mergin array', len(newpopluation_array))
                  print('inner stage after mergin arrayfull', newpopluation_array)'''
                  dunnscore_dict=self.get_dunn_score(newpopluation_array)
                  '''print(dunnscore_dict)'''
                  sorted_dict = util.sortDictionaryForDavis(dunnscore_dict)
        silhoutte_dict = self.get_silhoutte_score(newpopluation_array)

        sorted_dict = util.sortDictionary(silhoutte_dict)

        return  final_dict




if __name__ == '__main__':
    test_labels = np.zeros(214, int)
    test_labels[0:69] = np.int(1)
    test_labels[70:145] = np.int(2)
    test_labels[146:162] = np.int(3)
    test_labels[163:175] = np.int(4)
    test_labels[176:184] = np.int(5)
    obj=MSMA(6,214,'',100,'')
    final_dict=obj.start_clustering()
    for key in final_dict:
        classes=final_dict[key]
        randval=newmeme.rand_index_score(test_labels,classes.astype(int))
        jacardkmean1 = jaccard_similarity_score(test_labels, classes.astype(int))
        print('randval',randval)
        print('jacard', jacardkmean1)
    print('final', final_dict)
    result1 = np.loadtxt(open("C:\personal\PhD\Dataset\sonar-data-set\glass-classification\\glass.csv", "r"),
                         delimiter=",")
    kmeans1 = KMeans(n_clusters=6, random_state=0).fit(result1)
   # print("kmeans without meme", confusion_matrix(test_labels, kmeans1.labels_))
    print('kmeans level',kmeans1.labels_)
    randkmean1 = newmeme.rand_index_score(test_labels, kmeans1.labels_)
    print('rand',randkmean1)
    jacardkmean1 = jaccard_similarity_score(test_labels, kmeans1.labels_)
    print('jacard',jacardkmean1)

    D1 = pairwise_distances(result1, metric='euclidean')

    M1, C1 = kMedoids(D1, 6)
    kmediodlabels1 = convertLabelsToList(M1, C1, 214)
    print('kmediod level', kmediodlabels1)
    randkmediod1 = newmeme.rand_index_score(test_labels, kmediodlabels1)
    jacardkmediod1 = jaccard_similarity_score(test_labels, kmediodlabels1)
    print('randkmediod1',randkmediod1)
    print('jacardkmediod1',jacardkmediod1)





