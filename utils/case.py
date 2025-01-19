from typing import List, Dict

# TODO: should have better ways to represent case and CB!

class BaseCase:

    def __init__(self, _id=0, **kwargs):
        self._id = _id
        self.attributes = kwargs
        
    @classmethod
    def from_series(cls, series, _id=0):
        '''Create a BaseCase object from a pandas Series.
        '''
        return cls(_id, **series.to_dict())

    def cal_sim(self, case: 'BaseCase', attr_functions: List[float] | Dict[str, float], 
                weights: List[float] | Dict[str, float]) -> float:
        '''Calculate the similarity between two cases.
        Args:
            case: The case to compare with.
            weights: The weight of each attribute. Default is [1.0]*len(attr_func).
        Returns:
            The similarity between two cases.
        '''
        
        assert isinstance(case, BaseCase)        

        if isinstance(weights, list):
            weights = {attr_name: w for attr_name, w in zip(self.attributes.keys(), weights)}
        if isinstance(attr_functions, list):
            attr_functions = {attr_name: attr_func for attr_name, attr_func in zip(self.attributes.keys(), attr_functions)}
        global_sim = 0
        w_sum = 0
        for attr_name, attr_func in attr_functions.items():
            if attr_func is None or attr_name not in weights or attr_name not in case.attributes \
                or attr_name not in self.attributes:
                continue
            w = weights[attr_name]
            sim = attr_func(self.attributes[attr_name], case.attributes[attr_name])
            global_sim += w*sim
            w_sum += w
        
        if w_sum == 0:
            return 0
        return global_sim / w_sum

class Query(BaseCase):

    def __init__(self, _id=0, **kwargs):
        super().__init__(_id, **kwargs)
        # self._id = _id
        # self.attributes = {attr_name: None for attr_name in Query.attr_names}
        
        # for i, attr_name in enumerate(self.attr_names):
        #     if i < len(args):
        #         self.attributes[attr_name] = args[i]
        # for attr_name, attr_value in kwargs.items():
        #     if attr_name in self.attributes:
        #         self.attributes[attr_name] = attr_value
    
    def __str__(self):
        return f'Query {self._id}'
    
    def __repr__(self):
        return f'Query {self._id}'

class Case(BaseCase):
    attr_names = {
                 # Description
                 'Task': None, 'Case study type': None, 'Case study': None, 
                 'Online/Off-line': None, 'Input for the model': None,
                 # Solution
                 'Model Approach': None, 'Model Type': None, 'Models': None, 
                 'Data Pre-processing': None, 'Complementary notes': None, 'Publication identifier': None,
                 # Performance
                 'Performance indicator': None, 'Performance': None, 'Publication Year': None
                 }

    def __init__(self, _id=0, **kwargs):
        super().__init__(_id, **kwargs)
    
    def __str__(self):
        return f'Case {self._id}'
    
    def __repr__(self):
        return f'Case {self._id}'
        
# TODO
class Description(BaseCase):
    attr_names = {
                 # Description
                 'Task': None, 'Case study type': None, 'Case study': None, 
                 'Online/Off-line': None, 'Input for the model': None,
                 }
    pass

class Solution(BaseCase):
    attr_names = {
                 # Solution
                 'Model Approach': None, 'Model Type': None, 'Models': None, 
                 'Data Pre-processing': None, 'Complementary notes': None, 'Publication identifier': None,
                 # Performance
                 'Performance indicator': None, 'Performance': None, 'Publication Year': None
                 }
    pass