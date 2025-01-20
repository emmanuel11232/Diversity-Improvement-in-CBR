from typing import List, Dict, Tuple, Callable
import pandas as pd
from utils.case import BaseCase, Query, Case, Description, GC, Solution
from utils.sim_func import *

def _retrieve_topk(query: Query, candidates: List[BaseCase], attr_functions: Dict[str, Callable], 
                      weights: List[float] | Dict[str, float] = None, k: int = None) -> List[Tuple[BaseCase, float]]:
        '''Retrieve the top-k most similar cases to the query from the case base.
        Args:
            query: The query case.
            k [optional]: The number of cases to retrieve. If None, return all cases. Default is None.
        Returns:
            A list of tuples, each tuple contains a case and its similarity to the query.
        '''
        if len(candidates) == 0:
            return []

        if weights is None:
            weights = [1.0]*len(attr_functions)
            
        # Calculate the similarity between the query and all cases in the case base
        sims = []
        for c in candidates:
            sim = query.cal_sim(c, attr_functions, weights)
            sims.append((c, sim))
        # Sort the cases by similarity
        sims = sorted(sims, key=lambda x: x[1], reverse=True)
        if k is None:
            return sims
        else:
            return sims[:k]
        
class CaseBase:
    attr_functions = {
                # Description
                'Task': sim_task, 'Case study type': sim_case_study_type, 'Case study': sim_leven, 
                'Online/Off-line': sim_bin, 'Input for the model': sim_leven,
                # Solution
                'Model Approach': sim_leven, 'Model Type': sim_leven, 'Models': sim_taxon, 
                'Data Pre-processing': sim_bin, 'Complementary notes': sim_leven, 'Publication identifier': sim_leven,
                # Performance
                'Performance indicator': None, 'Performance': None, 'Publication Year': None
                }

    def __init__(self, cases: List[Case], attr_functions: Dict[str, Callable] = None):
        if attr_functions is not None:
            self.attr_functions = attr_functions
        attributes = set(self.attr_functions)
        for c in cases:
            assert set(c.attributes.keys()).issubset(attributes), f"Attributes of {c} are not in {attributes}!"
        self.cases = cases
    
    @classmethod 
    def from_dataframe(cls, df: pd.DataFrame, columns: List[str] = None) -> List[Case]:
        '''Build a case base from a dataframe.
        Args:
            df: The dataframe.
            columns [optional]: The columns of the attributes to include in the case base. If None, 
            include all attributes. Default is None.
        Returns:
            A list of cases.
        '''
        if isinstance(columns, list):
            df = df[columns].copy()

        cases = []
        for i in range(len(df)):
            s = df.iloc[i]
            c = Case.from_series(s, _id=i)
            cases.append(c)
        return cls(cases)

    def retrieve_topk(self, query: Query, attr_functions: Dict[str, Callable] = None, 
                      weights: List[float] | Dict[str, float] = None, k: int = None) -> List[Tuple[Case, float]]:
        '''Retrieve the top-k most similar cases to the query from the case base.
        Args:
            query: The query case.
            k [optional]: The number of cases to retrieve. If None, return all cases. Default is None.
        Returns:
            A list of tuples, each tuple contains a case and its similarity to the query.
        ''' 
        if attr_functions is None:
            attr_functions = self.attr_functions    
        return _retrieve_topk(query, self.cases, attr_functions, weights, k)
        
class MCNN_CaseBase(CaseBase):
    
    def __init__(self, thr_desc: float = 1.0, thr_sol:float = 1.0):
        # super().__init__(cases)
        self.descriptions = []
        self.solutions = []
        self.thr_desc = thr_desc
        self.thr_sol = thr_sol

    # def add_description(self, desc: Description):
    #     self.descriptions.append(desc)
    #     # Relink the solution
    #     if desc.sol is not None:
    #         self.solutions.append(desc.sol)

    def retrieve_topk(self, query: Query, attr_functions: Dict[str, Callable] = None, 
                      weights: List[float] | Dict[str, float] = None, k: int = None) -> List[Tuple[Description, float]]:
        '''Retrieve the top-k most similar descriptions to the query from the case base.
        Args:
            query: The query case.
            k [optional]: The number of cases to retrieve. If None, return all cases. Default is None.
        Returns:
            A list of tuples, each tuple contains a case and its similarity to the query.
        '''
        if attr_functions is None:
            attr_functions = self.attr_functions    
        return _retrieve_topk(query, self.descriptions, attr_functions, weights, k)
    
    def retrieve_topk_sol(self, query: Query, attr_functions: Dict[str, Callable] = None, 
                          weights: List[float] | Dict[str, float] = None, k: int = None) -> List[Tuple[Solution, float]]:
        '''Retrieve the top-k most similar solutions to the query from the case base.
        Args:
            query: The query case.
            k [optional]: The number of cases to retrieve. If None, return all cases. Default is None.
        Returns:
            A list of tuples, each tuple contains a case and its similarity to the query.
        '''
        if attr_functions is None:
            attr_functions = self.attr_functions    
        return _retrieve_topk(query, self.solutions, attr_functions, weights, k)
    