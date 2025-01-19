
from typing import List, Dict, Tuple, Callable
from utils.case import Query, Case
from utils.casebase import CaseBase

def retrieve_topk(query: Query, case_base: CaseBase | List[Case], attr_functions: Dict[str, Callable] = None, 
                  weights: List[float] | Dict[str, float] = None, k: int = None) -> List[Tuple[Case, float]]:
    '''Retrieve the top-k most similar cases to the query from the case base.
    Args:
        query: The query case.
        case_base: The case base.
        weights: The weights for the attributes.
        k [optional]: The number of cases to retrieve. If None, return all cases. Default is None.
    Returns:
        A list of tuples, each tuple contains a case and its similarity to the query.
    '''
    if isinstance(case_base, list):
        case_base = CaseBase(case_base)
    
    sims_topk = case_base.retrieve_topk(query, attr_functions, weights, k)
    return sims_topk