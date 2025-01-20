
from typing import List, Tuple, Dict
import numpy as np
from utils.case import BaseCase, Query, Case, Description, GC, Solution
from utils.casebase import CaseBase

def cal_sim_matrix(cases: List[BaseCase], case_base: CaseBase = None, 
                   attr_functions: List[float] | Dict[str, float] = None,
                   weights: List[float] | Dict[str, float] = None) -> float:
    '''Calculate the similarity matrix of a group of cases.
    Args:
        cases: A list of cases.
        attr_functions: The attribute functions.
        weights: The weights for the attributes.
    Returns:
        The similarity matrix of the cases.
    '''
    if isinstance(case_base, CaseBase):
        attr_functions = case_base.attr_functions
    assert attr_functions is not None, 'Attribute functions are required.'

    if weights is None:
        weights = [1.0] * len(attr_functions)

    sim_matrix = np.eye(len(cases))
    for i in range(len(cases)):
        for j in range(len(cases)):
            sim = cases[i].cal_sim(cases[j], attr_functions, weights)
            sim_matrix[i, j] = sim
    return sim_matrix

# def cal_diversity(cases: List[BaseCase], case_base: CaseBase = None, 
#                   attr_functions: List[float] | Dict[str, float] = None,
#                   weights: List[float] | Dict[str, float] = None) -> float:
#     '''Calculate the diversity of a group of cases.
#     Args:
#         cases: A list of cases.
#         attr_functions: The attribute functions.
#         weights: The weights for the attributes.
#     Returns: 
#         The diversity of the cases.
#     '''
#     assert len(cases) > 1, 'At least two cases are required.'
    
#     if isinstance(case_base, CaseBase):
#         attr_functions = case_base.attr_functions
#     assert attr_functions is not None, 'Attribute functions are required.'

#     if weights is None:
#         weights = [1.0] * len(attr_functions)

#     div = 0
#     for c1 in cases:
#         for c2 in cases:
#             sim = c1.cal_sim(c2, attr_functions, weights)
#             div += 1 - sim
#     div /= len(cases) * (len(cases) - 1) / 2
    
#     return div

def cal_diversity(sim_matrix: np.ndarray) -> float:
    '''Calculate the diversity of a group of cases.
    Args:
        sim_matrix: The similarity matrix of the cases.
    Returns: 
        The diversity of the cases.
    '''
    assert len(sim_matrix.shape) == 2, 'The similarity matrix must be 2D.'
    
    div = 0
    for i in range(len(sim_matrix)):
        for j in range(len(sim_matrix)):
            div += 1 - sim_matrix[i, j]
    div /= len(sim_matrix) * (len(sim_matrix) - 1) / 2
    
    return div

# TODO: cal_coverage()