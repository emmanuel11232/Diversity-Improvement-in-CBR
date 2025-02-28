#This file is made for validation purposes, it takes 40 cases from the case base and compares the results 
# of the modified CNN with the results of the original CNN

import pandas as pd
import numpy as np
import math
import sklearn
import matplotlib.pyplot as plt
import statistics

import seaborn as sns # librerías para EDA visual

from Methods2 import DescriptionsAndSolutions, CaseBase, CompareSimilarity, Description
from Modified_Condensed_Nearest_Neighbors import search_solutions_from_descriptions, compute_diversity
from Condensed_CaseBase_Generation import create_condensed_case_base
from Methods import SearchSimilar
#First I will set the original case base
#Case Base comes from an Excel provided by the supervisor
path = r'C:\Users\emman\Documents\TEC\DLIG\Case Based Reasoning\CaseBase\CleanedDATA V12-05-2021.csv'
df = pd.read_csv(path, sep=';', encoding='windows-1252')


#Global variables
CaseBase_Train=CaseBase[40:] #Set to modify with the modified CNN
CaseBase_Test=CaseBase[:40] #Set to make the queries
shared_data = {}
shared_data['Diversity_local'] = []
shared_data['Diversity_average'] = []

shared_data['Diversity_local_Org'] = []
shared_data['Diversity_average_Org'] = []

def apply_CNN(threshold_description,threshold_solution):
    solution_list,description_list=DescriptionsAndSolutions(CaseBase_Train)
    weights_description_feature=[0.2,0.2,0.2,0.2,0.2]
    weights_solution_feature=[0.25,0.25,0.25,0.25]

    solutions_condensed,descriptions_condensed=create_condensed_case_base(description_list=description_list,
                                                                          solution_list=solution_list,
                                                                          weights_description_feature=weights_description_feature,
                                                                          weights_solution_feature=weights_solution_feature,
                                                                          threshold_description=threshold_description,
                                                                          threshold_solution=threshold_solution)
    
    return solutions_condensed,descriptions_condensed

def retrieval_for_ModCNN(solutions,descriptions,query):
    retrievals = search_solutions_from_descriptions(sample=query,
                                                    descriptions_store=descriptions,
                                                    solutions_store=solutions,
                                                    weights_description=[0.2,0.2,0.2,0.2,0.2])
    list_retrievals_solutions=[]
    for i in retrievals:
        list_retrievals_solutions.append(i.data)
    diversity=compute_diversity(list_retrievals_solutions,[0.25,0.25,0.25,0.25])
    return diversity

diversity_matrix = []

Thresholds=[0.65,0.7,0.8,0.9,0.95]
print("start")
for i in Thresholds:
    row_values = []
    for j in Thresholds:
        solutions_condensed,descriptions_condensed=apply_CNN(i,j)
        shared_data['Diversity_local']=[]
        for case in CaseBase_Test:
            query=Description(case.description,1)
            diversity=retrieval_for_ModCNN(solutions_condensed,descriptions_condensed,query)
            shared_data['Diversity_local'].append(diversity)
        shared_data['Diversity_average'].append(statistics.mean(shared_data['Diversity_local']))
        row_values.append(statistics.mean(shared_data['Diversity_local']))  # Store in matrix
        
    diversity_matrix.append(row_values)  # Add row to matrix


def plot_diversity():
    global shared_data
    Thresholds=[0.65,0.7,0.8,0.9,0.95]
    Divs=shared_data.get('Diversity_local', [])
    Divs_average=shared_data.get('Diversity_average', [])
    plt.plot(Thresholds,Divs_average)
    plt.title('Diversity')
    plt.xlabel('Threshold')
    plt.ylabel('Diversity')
    plt.show()
    print(f"Average Diversity: {Divs_average}")


# Convert to NumPy array
diversity_matrix = np.array(diversity_matrix)

# 🔹 Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(diversity_matrix, cmap='coolwarm', interpolation='nearest')

# Add color bar
plt.colorbar(label='Diversity Score')

# Label axes
plt.xticks(range(len(Thresholds)), Thresholds)
plt.yticks(range(len(Thresholds)), Thresholds)
plt.xlabel("Threshold j")
plt.ylabel("Threshold i")
plt.title("Diversity Heatmap for Different Thresholds")

# Show values in cells
for i in range(len(Thresholds)):
    for j in range(len(Thresholds)):
        plt.text(j, i, f"{diversity_matrix[i, j]:.2f}", ha='center', va='center', color='black')

plt.show()