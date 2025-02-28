from Modified_Condensed_Nearest_Neighbors import Mod_CNN,compute_diversity,search_solutions_from_descriptions
from Methods2 import DescriptionsAndSolutions,CaseBase


def create_condensed_case_base(description_list,
                               solution_list,
                               weights_description_feature,
                               weights_solution_feature,
                               threshold_description,
                               threshold_solution):
    #Create the description store
    Descriptions=Mod_CNN(description_list)
    Solutions=Mod_CNN(solution_list)
    #Take first random sample
    sample=Descriptions.random_sample()
    #Add first parent to store
    Descriptions.add_parent(sample)

    #Now make loop to add the rest
    for i in range(0,len(Descriptions.dataset_objects)):
        #Take one random sample
        sample=Descriptions.random_sample()
        #Compute the similarity of the sample with the samples on store
        similarities=Descriptions.compute_similarity_store_vs_sample(sample,"description",weights_description_feature)
        #Sort the similarities from higher to lower
        similarities_sorted=Descriptions.sort(similarities)
        #If the higher value is greater than the threshold then nest the description
        if similarities_sorted[0][1] > threshold_description:
            Descriptions.nest_higher_similarity(sample,similarities_sorted)
        else:
            Descriptions.add_parent(sample)

    #Almost the same thing, but for solutions

    sample=Solutions.random_sample()
    #Add first parent to store
    Solutions.add_parent(sample)

    #Now make loop to add the rest
    for i in range(0,len(Solutions.dataset_objects)):
        #Take one random sample
        sample=Solutions.random_sample()
        #Compute the similarity of the sample with the samples on store
        similarities=Solutions.compute_similarity_store_vs_sample(sample,"solution",weights_solution_feature)
        #Sort the similarities from higher to lower
        similarities_sorted=Solutions.sort(similarities)
        #If the higher value is greater than the threshold then nest the solution
        if similarities_sorted[0][1] > threshold_solution:
            Solutions.nest_higher_similarity(sample,similarities_sorted)
        else:
            Solutions.add_parent(sample)


    #Now we have both organized and can reassign solutions from child descriptions to GCs:
    Solutions.store=Descriptions.assign_child_solutions_to_parent_descriptions(Solutions.store)

    #After, we can reorder the store in solutions so the GCs would be the ones with highest performance
    Solutions.update_solutions_by_performance()

    return Solutions.store, Descriptions.store

#Apply everything
#Create the descriptions and solutions lists
solution_list,description_list=DescriptionsAndSolutions(CaseBase)

#Important stuff
weights_description_feature=[0.2,0.2,0.2,0.2,0.2]
weights_solution_feature=[0.25,0.25,0.25,0.25]
threshold_description=0.7
threshold_solution=0.6

solutions_condensed,descriptions_condensed=create_condensed_case_base(description_list=description_list,
                                                                      solution_list=solution_list,
                                                                      weights_description_feature=weights_description_feature,
                                                                      weights_solution_feature=weights_solution_feature,
                                                                      threshold_description=threshold_description,
                                                                      threshold_solution=threshold_solution)

print(solutions_condensed)
print(descriptions_condensed)