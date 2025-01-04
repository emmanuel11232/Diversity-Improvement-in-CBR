#This file is made for validation purposes, it takes 40 cases from the case base and compares the results 
# of the modified CNN with the results of the original CNN

import pandas as pd
import numpy as np
import math
import sklearn
import matplotlib.pyplot as plt

import seaborn as sns # librerías para EDA visual

#Nuevo
from functools import reduce
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
import gensim.downloader as api
from numpy import dot
from numpy.linalg import norm
import Levenshtein
import random

from Methods2 import CasoInd,Diversity,ModifiedCNN,SearchSimilarModCNN, DescriptionsAndSolutions
from Methods import SearchSimilar
#First I will set the original case base
#Case Base comes from an Excel provided by the supervisor
path = r'C:\Users\emman\Documents\TEC\DLIG\Case Based Reasoning\CaseBase\CleanedDATA V12-05-2021.csv'
df = pd.read_csv(path, sep=';', encoding='windows-1252')

CaseBase=[]
for i in range(0,len(df)):
    Description=df.loc[i,['Task', 'Case study type', 'Case study', 'Online/Off-line', 'Input for the model']]
    Solution=df.loc[i,['Model Approach', 'Model Type', 'Models', 'Data Pre-processing', 'Complementary notes', 'Publication identifier,,,,,,,,,,,,,,,,,,']] #Gotta fix this bug
    Performance=df.loc[i,['Performance indicator', 'Performance', 'Publication Year']]
    case=CasoInd(Description.values,Solution.values,Performance.values,i)
    CaseBase.append(case)

#Global variables
CaseBase_Train=CaseBase #Set to modify with the modified CNN
CaseBase_Test=CaseBase[:40] #Set to make the queries
shared_data = {}
shared_data['Diversity_local'] = []
shared_data['Diversity_average'] = []

shared_data['Diversity_local_Org'] = []
shared_data['Diversity_average_Org'] = []

def apply_algorithms(Threshold):
    global shared_data
    
    DescriptionList,SolutionList=DescriptionsAndSolutions(CaseBase)
    Nested_Descriptions,Nested_Solutions=ModifiedCNN(DescriptionList,SolutionList,[0.2,0.2,0.2,0.2,0.2],[0.2,0.2,0.2,0.2,0.2],Threshold,Threshold)
    
    shared_data['Nested_Descriptions'] = Nested_Descriptions
    shared_data['Nested_Solutions'] = Nested_Solutions

def save_data(values):
    global shared_data

    Nested_Descriptions = shared_data.get('Nested_Descriptions', [])
    Nested_Solutions = shared_data.get('Nested_Solutions', [])
 
    Weights= [0.2, 0.2, 0.2, 0.2, 0.2] 
    NumberRetrievals=5

    ListRetrievals=SearchSimilarModCNN(values,Nested_Descriptions,Nested_Solutions,NumberRetrievals,Weights)

    # Función para mostrar la información de los objetos en la GUI
    Div=Diversity(ListRetrievals,Weights)
    print("Diversidad de soluciones:", Div)
    print(len(Nested_Descriptions),len(Nested_Solutions))
    return Div

def Original_diversity(values):
    Weights= [0.2, 0.2, 0.2, 0.2, 0.2] 
    ListRetrievals,ListSim=SearchSimilar(values,CaseBase,5,Weights)

    # Función para mostrar la información de los objetos en la GUI
    Div=Diversity(ListRetrievals,Weights)

    return Div

def Diversity_Calculations_Original():
    global shared_data
    Divs=[]
    for i in range(0,40):
        values=CaseBase_Test[i].description
        Div=Original_diversity(values)
        Divs.append(Div)
    shared_data['Diversity_local_Org'] = Divs
    shared_data['Diversity_average_Org'].append(np.mean(Divs))
    print(f"Original Diversity: {np.mean(Divs)}")

def Diversity_Calculations():
    global shared_data
    Thresholds=[0.4,0.5,0.6,0.7,0.8,0.9,0.95]
    for j in Thresholds:
        apply_algorithms(j)
        Divs=[]
        for i in range(0,40):
            values=CaseBase_Test[i].description
            Div=save_data(values)
            Divs.append(Div)
        shared_data['Diversity_local'] = Divs
        shared_data['Diversity_average'].append(np.mean(Divs))
        print(f"Average Diversity: {np.mean(Divs)}")

def plot_diversity():
    global shared_data
    Thresholds=[0.4,0.5,0.6,0.7,0.8,0.9,0.95]
    Divs=shared_data.get('Diversity_local', [])
    Divs_average=shared_data.get('Diversity_average', [])
    plt.plot(Thresholds,Divs_average)
    plt.title('Diversity')
    plt.xlabel('Threshold')
    plt.ylabel('Diversity')
    plt.show()
    print(f"Average Diversity: {Divs_average}")

Diversity_Calculations()
Diversity_Calculations_Original()
plot_diversity()
