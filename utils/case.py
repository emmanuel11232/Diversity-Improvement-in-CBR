from typing import List, Dict
from utils.similarity import *

class Query:
    case_num = 0
    attr_func = {
                 # Description
                 'Task': sim_task, 'Case study type': sim_case_study_type, 'Case study': sim_leven, 
                 'Online/Off-line': sim_bin, 'Input for the model': sim_leven,
                 }

    def __init__(self, *args, **kwargs):
        self.attributes: Dict[str, Any] = {k: None for k in self.attr_func.keys()}
        for i, attr_name in enumerate(self.attr_func.keys()):
            if i < len(args):
                self.attributes[attr_name] = args[i]
        for attr_name, attr_value in kwargs.items():
            if attr_name in self.attr_func:
                self.attributes[attr_name] = attr_value

    @classmethod
    def from_series(cls, series):
        '''Create a Case object from a pandas Series.
        '''
        return cls(**series.to_dict())
    
    def __str__(self):
        return f'Case {self.case_id}'

    def cal_sim(self, case: 'Query', weights: List[float] | Dict[str, float]) -> float: # TODO
        '''Calculate the similarity between two cases.
        Args:
            case: The case to compare with.
            weights: The weight of each attribute. Default is [1.0]*len(attr_func).
        Returns:
            The similarity between two cases.
        '''
        
        assert isinstance(case, Query)        

        if isinstance(weights, list):
            weights = {attr_name: w for attr_name, w in zip(self.attributes.keys(), weights)}

        attr_func = {k: v for k, v in self.attr_func.items() if k in weights and k in case.attributes}
        
        global_sim = 0
        w_sum = 0
        for attr_name, sim_func in attr_func.items():
            if sim_func is None:
                continue
            w = weights[attr_name]
            sim = sim_func(self.attributes[attr_name], case.attributes[attr_name])
            global_sim += sim * w
            w_sum += w
            
        return global_sim / w_sum
    
class Case(Query):
    case_num = 0
    attr_func = {
                 # Description
                 'Task': sim_task, 'Case study type': sim_case_study_type, 'Case study': sim_leven, 
                 'Online/Off-line': sim_bin, 'Input for the model': sim_leven,
                 # Solution
                 'Model Approach': sim_leven, 'Model Type': sim_leven, 'Models': sim_taxon, 
                 'Data Pre-processing': sim_bin, 'Complementary notes': sim_leven, 'Publication identifier': sim_leven,
                 # Performance
                 'Performance indicator': None, 'Performance': None, 'Publication Year': None
                 }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.case_id = Case.case_num
        Case.case_num += 1