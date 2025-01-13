import pandas as pd
import numpy as np
import math
import sklearn
import matplotlib.pyplot as plt

import seaborn as sns # librerías para EDA visual

#Nuevo
import random
from functools import reduce
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import gensim.downloader as api

import Levenshtein
#API for text semantic text similarity
word2vec_model = api.load("glove-wiki-gigaword-50")
#Case Base comes from an Excel provided by the supervisor
# path = r'C:\Users\emman\Documents\TEC\DLIG\Case Based Reasoning\CaseBase\CleanedDATA V12-05-2021.csv'
path = r'/home/dingw/work/CleanedDATA V12-05-2021.csv'
df = pd.read_csv(path, sep=';', encoding='windows-1252')

#Each case will be represented by a structure which will have the solutions, descriptions and performance seperated
class CasoInd:
    def __init__(self, Description_CB, Solution_CB, Performance,reference):
        #Cada uno de estos atributos son listas, menos reference que es un int
        self.description = Description_CB
        self.solution = Solution_CB
        self.performance=Performance
        self.reference=reference
    def __str__(self):
        return f"{self.reference}"
#Then we create the casebase asigning everything needed

CaseBase=[]
for i in range(0,len(df)):
    Description_CB=df.loc[i,['Task', 'Case study type', 'Case study', 'Online/Off-line', 'Input for the model']]
    Solution_CB=df.loc[i,['Model Approach', 'Model Type', 'Models', 'Data Pre-processing', 'Complementary notes', 'Publication identifier,,,,,,,,,,,,,,,,,,']] #Gotta fix this bug
    Performance=df.loc[i,['Performance indicator', 'Performance', 'Publication Year']]
    case=CasoInd(Description_CB.values,Solution_CB.values,Performance.values,i)
    CaseBase.append(case)
    
    
def CompareSimilarityDesc(Sol1,Sol2,Weights):
    #Gives the similarity for two given cases
    Sim=Similarity()
    Sim1=Sim.SimTask(Sol1,Sol2)
    Sim2=Sim.SimCaseStudyType(Sol1,Sol2)
    Sim3=Sim.Levenshtein(Sol1.description[2],Sol2.description[2])#Case Study
    Sim4=Sim.On_Offline(Sol1.description[3],Sol2.description[3])
    Sim5=Sim.Levenshtein(Sol1.description[4],Sol2.description[4]) #Input for the model
    SimGlobal=Sim.GlobalSim([Sim1,Sim2,Sim3,Sim4,Sim5],Weights)    
    return SimGlobal

def CompareSimilaritySol(Sol1,Sol2,Weights):
    Sim=Similarity()
    Sim1=Sim.SimPrePro(Sol1,Sol2)#Data PreProcessing
    Sim2=Sim.Levenshtein(Sol1.solution[0],Sol2.solution[0])#Model Approach
    Sim3=Sim.Levenshtein(Sol1.solution[2],Sol2.solution[2])#Model Type
    Sim4=Sim.SimTaxon(Sol1,Sol2)#Model
    SimGlobal=Sim.GlobalSim([Sim1,Sim2,Sim3,Sim4],Weights)#Global Sim
    return SimGlobal


def SearchSimilar(UserInput,CaseBase,NumberRetrievals,Weights):
    InputCase=CasoInd(UserInput,0,0,"CasoUsuario")
    ListRetrievals=[]
    ListSim=[]
    for i in range(0,len(CaseBase)-1):
        Sim=CompareSimilarityDesc(InputCase,CaseBase[i],Weights)
        if i==0:
            ListRetrievals.append(CaseBase[i])
            ListSim.append(Sim)
        else:
            if len(ListRetrievals)<NumberRetrievals:
                    ListRetrievals.append(CaseBase[i])
                    ListSim.append(Sim)
            else:
                 for j in range(0,NumberRetrievals-1): 
                      if Sim>ListSim[j]:
                        ListRetrievals.pop(j)
                        ListSim.pop(j)
                        ListRetrievals.append(CaseBase[i])
                        ListSim.append(Sim) 
                        break
    return ListRetrievals,ListSim

def Diversity(RetrievedSolutions,weights):
    #Ill do it taking the average similarity between solutions
    listDiv=[]
    for i in range(0,len(RetrievedSolutions)-1):
        for j in range(i+1,len(RetrievedSolutions)):
            Div=1-CompareSimilaritySol(RetrievedSolutions[i],RetrievedSolutions[j],weights)
            listDiv.append(Div)
    return sum(listDiv) / len(listDiv)

############################## Wendi's research and modified CNN ####################################


#Now, for the CNN we will separate cases by their Solution and their description

class Solution:
    def __init__(self,solution,reference):
        self.solution=solution
        self.NestedSolutions=[] #Space to nest similar solutions
        self.link=reference
class Description:
    def __init__(self,description,reference):
        self.description=description
        self.NestedDescriptions=[] #Space to nest similar descriptions
        self.link=reference


def DescriptionsAndSolutions(CaseBase):
    SolutionList=[]
    DescriptionList=[]
    for i in CaseBase:
        Sol=Solution(i.solution,i.reference)
        Desc=Description(i.description,i.reference)
        SolutionList.append(Sol)
        DescriptionList.append(Desc)
    return SolutionList,DescriptionList

def ModifiedCNN(SolutionList,DescriptionList,WeightsSol,WeightsDesc):

    NestedSolutionList=[]
    NestedDescriptionList=[]
    ThresholdSol=0.7
    ThresholdDesc=0.7
    #########################################Generalization of descriptions####################################################
    i=0
    while DescriptionList:
        if i==0:
            RandDescription = random.sample(DescriptionList, 1)
            NestedDescriptionList.append(RandDescription[0])
            DescriptionList=[obj for obj in DescriptionList if obj not in RandDescription]
            i+=1
        else:
            RandDescription = random.sample(DescriptionList, 1)
            DescriptionList=[obj for obj in DescriptionList if obj not in RandDescription]
            for j in range(0,len(NestedDescriptionList)):#Compare Similarity with the rest of the descriptions
                DescriptionSimilarity=CompareSimilarityDesc(NestedDescriptionList[j],RandDescription[0],WeightsDesc)
                print(DescriptionSimilarity)
                if DescriptionSimilarity>=ThresholdDesc:#if it is similar, nest it
                    NestedDescriptionList[j].NestedDescriptions.append(RandDescription[0])
                    SolutionList[RandDescription[0].link].link=NestedDescriptionList[j].link #Se asume que SolutionList está ordenada
                    #Now the solution that this one had has to be reindexed
                    break
                if j==(len(NestedDescriptionList)-1):
                    if DescriptionSimilarity<ThresholdDesc:#if it is not similar to anybody it is a GC
                        NestedDescriptionList.append(RandDescription[0])
    

    ##########################################Generalization of solutions###################################################
    i=0
    while SolutionList:
        if i==0:
            RandSolution = random.sample(SolutionList, 1)
            NestedSolutionList.append(RandSolution[0])
            SolutionList=[obj for obj in SolutionList if obj not in RandSolution]
            i+=1
        else:
            RandSolution = random.sample(SolutionList, 1)
            SolutionList=[obj for obj in SolutionList if obj not in RandSolution]
            for j in range(0,len(NestedSolutionList)):#Compare Similarity with the rest of the solutions
                SolutionSimilarity=CompareSimilaritySol(NestedSolutionList[j],RandSolution[0],WeightsSol)
                if SolutionSimilarity>=ThresholdSol:#if it is similar, nest it
                    NestedSolutionList[j].NestedSolutions.append(RandSolution[0])
                
                    #Now the description that this one had has to be reindexed TODO
                    break
                if j==(len(NestedSolutionList)-1):
                    if SolutionSimilarity<ThresholdSol:#if it is not similar to anybody it is a GC
                        NestedSolutionList.append(RandSolution[0])

    #################################Process of reindexation and re-arraging of solutions###################################
    #First re-arrange the solutions
    """for k in range(0,len(NestedSolutionList)):
        if len(NestedSolutionList[k].NestedSolution)>=1:"""
            #The solution returned by checking the performance
            #Will be the GC and nest the other solutions to it
            #Check the descriptions that have
    return NestedDescriptionList,NestedSolutionList

def SearchSimilarModCNN(UserInput,Descriptions,Solutions,NumberRetrievals,Weights):
    InputCase=Description(UserInput,299)
    ListRetrievalsDesc=[]
    ListSim=[]
    for i in range(0,len(Descriptions)-1):
        Sim=CompareSimilarityDesc(InputCase,Descriptions[i],Weights)
        if i==0:
            ListRetrievalsDesc.append(Descriptions[i])
            ListSim.append(Sim)
        else:
            if len(ListRetrievalsDesc)<NumberRetrievals:
                    ListRetrievalsDesc.append(Descriptions[i])
                    ListSim.append(Sim)
            else:
                 for j in range(0,NumberRetrievals-1): 
                      if Sim>ListSim[j]:
                        ListRetrievalsDesc.pop(j)
                        ListSim.pop(j)
                        ListRetrievalsDesc.append(Descriptions[i])
                        ListSim.append(Sim) 
                        break
    OrderIndexList= sorted(range(len(ListSim)), key=lambda i: ListSim[i], reverse=True)
    ListRetrievalSol=[]
    for j in OrderIndexList:

        LinkSol=ListRetrievalsDesc[j].link
        for k in Solutions:
            if k.link==LinkSol:
                ListRetrievalSol.append(k)
        if len(ListRetrievalSol)>NumberRetrievals:
            ListRetrievalSol=[ListRetrievalSol[0],ListRetrievalSol[1],ListRetrievalSol[2],ListRetrievalSol[3],ListRetrievalSol[4]]
            break
    return ListRetrievalSol

def Performance(NestedSolutionList,GenSolution):
    #Get performance indicator from each one of the nested solutions
    #Pick the best
    #return the best solution
    #This has to be done with a metric and a case base of performance indicators
    #Because of time I will skip this, leaving this as room for improvement
    pass

