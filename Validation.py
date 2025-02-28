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

# Thresholds (θ_sol and θ_des)
sol_thresholds = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0 ]
des_thresholds = [ 1.0, 0.98, 0.96, 0.94, 0.92, 0.9, 0.85, 0.8]

diversity_results = {sol: [] for sol in sol_thresholds}
print("start")
for sol in sol_thresholds:
    for des in des_thresholds:
        solutions_condensed,descriptions_condensed=apply_CNN(sol, des)
        shared_data['Diversity_local']=[]
        for case in CaseBase_Test:
            query=Description(case.description,1)
            diversity=retrieval_for_ModCNN(solutions_condensed,descriptions_condensed,query)
            shared_data['Diversity_local'].append(diversity)
        avg_diversity = statistics.mean(shared_data['Diversity_local'])
        shared_data['Diversity_average'].append(avg_diversity)
        diversity_results[sol].append(avg_diversity)


plt.figure(figsize=(10, 6))

for sol, diversities in diversity_results.items():
    plt.plot(des_thresholds, diversities, marker='o', label=f'θ_sol = {sol}')

# Labels and title
plt.xlabel("Description generalization threshold θ_des")
plt.ylabel("Diversity")
plt.title("Diversity vs. Description Generalization Threshold")
plt.legend()
plt.grid(True)

# Show plot
plt.show()