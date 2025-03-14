from Methods2 import CompareSimilarity
import random

class Mod_CNN:
    def __init__(self,cases):
        #Initialize the dataset
        self.dataset_objects=cases #List of descriptions or solutions from the original dataset
        self.store=[] #List of descriptions or solutions to generalize/generalized
    
    def random_sample(self):
        #Take one random sample from the dataset
        #Removes it from the original dataset, since it shouldn't be repeated
        random_index=random.randint(0,len(self.dataset_objects)-1)
        random_sample=self.dataset_objects[random_index]
        self.dataset_objects.pop(random_index)
        return random_sample

    def add_parent(self,sample):
        #Add a sample to store
        sample.state="parent"
        self.store.append(sample)
        return self.store

    def compute_similarity_store_vs_sample(self,sample,des_sol:str,weights):
        #Compute similarity with all GCs and descriptions in store
        #Returns the list of similarities and the reference from each one
        similarities=[]
        #Go over GCs and take similarities
        for generalized_case in self.store:
            similarities.append((generalized_case.link,CompareSimilarity(generalized_case.data,sample.data,weights,des_sol)))
            #if GC has nested cases take similarity from nested cases too
            if len(generalized_case.nested_cases)>0:
                for nested in generalized_case.nested_cases:
                    similarities.append((nested.link,CompareSimilarity(nested.data,sample.data,weights,des_sol)))
        return similarities
    
    def sort(self,list_unsorted):
        #Helper function to sort the similarities from higher to lower
        sorted_list = sorted(list_unsorted, key=lambda x: x[1], reverse=True)
        return sorted_list
    
    def search_case_by_reference(self,ref):
        #Go over store and search for a solution/description by reference
        retrieved_case=None
        for generalized_case in self.store:
            if generalized_case.link==ref:
                retrieved_case = generalized_case
                break
            else:
                if len(generalized_case.nested_cases)>0:
                     for nested in generalized_case.nested_cases:
                        if nested.link==ref:
                            retrieved_case=nested
                            break
        return retrieved_case

    def search_parent_from_child_reference(self,ref):
        #Retrieves the parent from a child reference link
        retrieved_case=self.search_case_by_reference(ref)
        if retrieved_case.state == "child":
            parent=self.search_case_by_reference(retrieved_case.parent)
            return parent
        else:
            raise Exception(f"Case not a child, case: {retrieved_case.link}, {retrieved_case.state}")
    
    def update_store_parents(self,parent):
        #Updates the GCs on store
        for generalized_case in range(0,len(self.store)):
            if self.store[generalized_case].link==parent.link:
                self.store[generalized_case]=parent
    
    def remove_with_id(self,id):
        for i in range(0,len(self.store)):
            if self.store[i].link==id:
                self.store.pop(i)
                break
            else:
                if len(self.store[i].nested_cases) > 0:
                    for j in range(0,len(self.store[i].nested_cases)):
                        if self.store[i].nested_cases[j].link==id:
                            self.store[i].nested_cases.pop(j)
                            return

    def nest_higher_similarity(self, sample, sorted_similarities):
        #Given the list of similarities takes the higher value and nestes it
        #Take threshold into consideration before using this function (filter the similarity list with the threshold)
        higher_similarity_case=self.search_case_by_reference(sorted_similarities[0][0])
        if higher_similarity_case.state == "child":
            parent=self.search_parent_from_child_reference(sorted_similarities[0][0])
            sample.state="child"
            sample.parent=parent.link
            parent.nested_cases.append(sample)
            self.update_store_parents(parent)
        elif higher_similarity_case.state == "parent":
            sample.state="child"
            sample.parent=higher_similarity_case.link
            higher_similarity_case.nested_cases.append(sample)
            self.update_store_parents(higher_similarity_case)
        else:
            raise Exception(f"Error in the case base, unclasified, case: {higher_similarity_case.link}, {higher_similarity_case.state}")

    def assign_child_solutions_to_parent_descriptions(self,solutions):
        #Reassign solutions from child descriptions to parent descriptions
        #Function for description object and receives the solutions store
        for generalized_solutions in range(0,len(solutions)):
            #for parent solution
            description=self.search_case_by_reference(solutions[generalized_solutions].parent_description)
            #if the matching case is a child, assign the parent reference
            if description.state == "child":
                parent_description=self.search_parent_from_child_reference(description.link)
                solutions[generalized_solutions].parent_description = parent_description.link
            #If there are nested solutions
            if len(solutions[generalized_solutions].nested_cases)>0:
                for nested in range(0,len(solutions[generalized_solutions].nested_cases)):         
                    description=self.search_case_by_reference(solutions[generalized_solutions].nested_cases[nested].parent_description)
                    #if the matching case is a child, assign the parent reference
                    if description.state == "child":
                        parent_description=self.search_parent_from_child_reference(description.link)
                        solutions[generalized_solutions].nested_cases[nested].parent_description = parent_description.link
        return solutions
                
    def update_solutions_by_performance(self):
        #function only for solutions object
        #Use this until the solutions store is completed
        for generalized_solution in self.store:
            if len(generalized_solution.nested_cases)>0:
                previous_performance=generalized_solution.performance
                id=generalized_solution.link
                for nested in generalized_solution.nested_cases:
                    if nested.performance>previous_performance:
                        previous_performance=nested.performance
                        id=nested.link
                if previous_performance!=generalized_solution.performance:
                    new_parent=self.search_case_by_reference(id)
                    new_parent.nested_cases=generalized_solution.nested_cases
                    generalized_solution.state="child"
                    generalized_solution.parent=new_parent.link
                    generalized_solution.nested_cases=[]
                    new_parent.nested_cases.append(generalized_solution)
                    new_parent.state="parent"
                    new_parent.parent=None
                    for i in range(0,len(new_parent.nested_cases)):
                        if new_parent.nested_cases[i].link==new_parent.link:
                            new_parent.nested_cases.pop(i)
                            break
                    self.remove_with_id(generalized_solution.link)
                    self.store.append(new_parent)
                
def search_solutions_from_descriptions(sample,descriptions_store,solutions_store,weights_description,amount=5):
    #Retrieve most similar GC and take the GSs from that one if they are not enough,
    #take the next most similar and retrieve the rest
    #This is a function for description objects
    similar_GC=[]
    for generalized_case in descriptions_store:
            similar_GC.append((generalized_case.link,CompareSimilarity(generalized_case.data,sample.data,weights_description,"description")))
    similar_GC = sorted(similar_GC, key=lambda x: x[1], reverse=True)
    list_solution=[]

    for most_similar in similar_GC:
        for generalized_solution in solutions_store:
            if generalized_solution.parent_description == most_similar[0]:
                list_solution.append(generalized_solution)
                if len(list_solution) >= amount:
                    return list_solution


def compute_diversity(candidates,weights):
    """
    Computes the diversity of a set of candidates based on pairwise similarities.
    
    :param candidates: A list of candidate objects.
    :param similarity_fn: A function that computes similarity between two objects.
    :return: The diversity value.
    """
    n = len(candidates)
    if n < 2:
        return 0  # No diversity if there's only one or no candidate
    
    total_dissimilarity = sum(
        1 - CompareSimilarity(candidates[i], candidates[j],weights,"solution")
        for i in range(n) for j in range(i + 1, n)
    )
    
    total_comparisons = (n * (n - 1)) / 2  # Number of unique pairs
    return total_dissimilarity / total_comparisons