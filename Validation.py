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

#wEIGHTS
weights_description=[0.3,0.3,0.2,0.1,0.1]
weights_solution=[0.1,0.25,0.35,0.3]

def apply_CNN(threshold_description,threshold_solution,CaseBase_Train,weights_description, weights_solution):
    solution_list,description_list=DescriptionsAndSolutions(CaseBase_Train)
    print("ORIGINAL:",len(solution_list),len(description_list))
    weights_description_feature=weights_description
    weights_solution_feature= weights_solution
    solutions_condensed,descriptions_condensed=create_condensed_case_base(description_list=description_list,
                                                                          solution_list=solution_list,
                                                                          weights_description_feature=weights_description_feature,
                                                                          weights_solution_feature=weights_solution_feature,
                                                                          threshold_description=threshold_description,
                                                                          threshold_solution=threshold_solution)
    print("NUEVA:",len(solutions_condensed),len(descriptions_condensed))
    print("threshold:",threshold_solution,threshold_description)
    print("")
    return solutions_condensed,descriptions_condensed

def retrieval_for_ModCNN(solutions,descriptions,query,weights_description,weights_solution):
    retrievals = search_solutions_from_descriptions(sample=query,
                                                    descriptions_store=descriptions,
                                                    solutions_store=solutions,
                                                    weights_description=weights_description)
    list_retrievals_solutions=[]
    for i in retrievals:
        list_retrievals_solutions.append(i.data)
    diversity=compute_diversity(list_retrievals_solutions,weights_solution)
    return diversity

diversity_matrix = []

# Thresholds (θ_sol and θ_des)
sol_thresholds = list(np.arange(0.6, 1.1, 0.05))  # De 0.6 a 1.2 con incrementos de 0.2
des_thresholds = list(np.arange(0.6, 1.1, 0.05))  # De 0.6 a 1.2 con incrementos de 0.2
des_thresholds = sorted(des_thresholds)

case_base_sizes = {sol_1: [] for sol_1 in sol_thresholds}
case_base_sizes_desc = {sol_2: [] for sol_2 in sol_thresholds}
diversity_results = {sol: [] for sol in sol_thresholds}

print("start")
for sol in sol_thresholds:
    for des in des_thresholds:
        solutions_condensed,descriptions_condensed=apply_CNN(sol, des, CaseBase_Train, weights_description, weights_solution)
        case_base_sizes[sol].append(len(solutions_condensed))
        case_base_sizes_desc[sol].append(len(descriptions_condensed))
        shared_data['Diversity_local']=[]
        for case in CaseBase_Test:
            query=Description(case.description,1)
            diversity=retrieval_for_ModCNN(solutions_condensed,descriptions_condensed,query)
            shared_data['Diversity_local'].append(diversity)
        avg_diversity = statistics.mean(shared_data['Diversity_local'])
        shared_data['Diversity_average'].append(avg_diversity)
        diversity_results[sol].append(avg_diversity)

#Diversity plot
plt.figure(figsize=(10, 6))
for sol, diversities in diversity_results.items():
    plt.plot(des_thresholds, diversities, marker='o', label=f'θ_sol = {sol}')

plt.gca().invert_xaxis()

# Labels and title
plt.xlabel("Description generalization threshold θ_des")
plt.ylabel("Diversity")
plt.title("Diversity vs. Description Generalization Threshold")
plt.legend()
plt.grid(True)

#Solutions plot
plt.figure(figsize=(10, 6))
for sol, sizes in case_base_sizes.items():
    plt.plot(des_thresholds, sizes, marker='o', label=f'θ_sol = {sol}')

# Labels and title
plt.xlabel("Description generalization threshold θ_des")
plt.ylabel("Case Base Size")
plt.title("Case Base Size vs. Description Generalization Threshold")
plt.legend()
plt.grid(True)

#Descriptions plot
plt.figure(figsize=(10, 6))

for sol, sizes in case_base_sizes_desc.items():
    plt.plot(des_thresholds, sizes, marker='o', label=f'θ_sol = {sol}')

# Labels and title
plt.xlabel("Description generalization threshold θ_des")
plt.ylabel("Case Base Size")
plt.title("Case Base Size vs. Description Generalization Threshold")
plt.legend()
plt.grid(True)

# Show plot
plt.show()