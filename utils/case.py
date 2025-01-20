from typing import List, Dict, Tuple, Iterable

class BaseCase:

    def __init__(self, _id=0, **kwargs):
        self._id = _id
        self.attributes = kwargs
        
    @classmethod
    def from_series(cls, series, _id=0):
        '''Create a BaseCase object from a pandas Series.
        '''
        return cls(_id, **series.to_dict())

    def __cal_sim(self, case: 'BaseCase', attr_functions: List[float] | Dict[str, float], 
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
    
    def cal_sim(self, case: 'BaseCase', attr_functions: List[float] | Dict[str, float], 
                weights: List[float] | Dict[str, float]) -> float:
        '''Calculate the similarity between two cases.
        Args:
            case: The case to compare with.
            weights: The weight of each attribute. Default is [1.0]*len(attr_func).
        Returns:
            The similarity between two cases.
        '''
        if isinstance(case, GC):
            aver_sim = 0
            for desc in case.descriptions:
                aver_sim += desc.cal_sim(self, attr_functions, weights)
            aver_sim /= len(case.descriptions)
            return aver_sim
        else:
            return self.__cal_sim(case, attr_functions, weights)

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

    def __init__(self, _id=0, **kwargs):
        super().__init__(_id, **kwargs)
    
    def __str__(self):
        return f'Case {self._id}'
    
    def __repr__(self):
        return f'Case {self._id}'
    
    def to_desc_sol_pair(self, desc_attrs: Iterable[str], sol_attrs: Iterable[str]
                         ) -> Tuple['Description', 'Solution']:
        '''Convert a case to a pair of description and solution.
        '''
        desc_attributes = {attr_name: self.attributes[attr_name] for attr_name in desc_attrs if attr_name in self.attributes}
        sol_attributes = {attr_name: self.attributes[attr_name] for attr_name in sol_attrs if attr_name in self.attributes}
        
        desc = Description(self._id, **desc_attributes)
        sol = Solution(self._id, **sol_attributes)
        desc.sol = sol
        sol.desc = [desc]
        
        return desc, sol
        
class Description(BaseCase):
    def __init__(self, _id=0, **kwargs):
        super().__init__(_id, **kwargs)
        self.sol = None
        
    def __str__(self):
        return f'Description {self._id}'
    
    def __repr__(self):
        return f'Description {self._id}'

class GC(Description):
    def __init__(self, descriptions: List[Description], _id=0, **kwargs):
        super().__init__(_id, **kwargs)
        self.descriptions = descriptions
        self.solutions = [desc.sol for desc in descriptions if desc.sol is not None]
        
    def __str__(self):
        return f'GC {self._id}: ' + '{' + ', '.join([str(desc) for desc in self.descriptions]) + '}'
    
    def __repr__(self):
        return f'GC {self._id}'
    
    def add_description(self, desc: Description):
        self.descriptions.append(desc)
        # Relink the solution
        if desc.sol is not None:
            self.solutions.append(desc.sol)

    def cal_sim(self, case: BaseCase, attr_functions: List[float] | Dict[str, float], 
                weights: List[float] | Dict[str, float]) -> float:
        '''Calculate the similarity between two cases.
        Args:
            case: The case to compare with.
            weights: The weight of each attribute. Default is [1.0]*len(attr_func).
        Returns:
            The similarity between two cases.
        '''
        
        assert isinstance(case, BaseCase)        
        # what if some descriptions are missing some attributes?
        avg_sim = 0
        for desc in self.descriptions:
            sim = desc.cal_sim(case, attr_functions, weights)
            avg_sim += sim
        avg_sim /= len(self.descriptions)
        return avg_sim
    
class Solution(BaseCase):
    def __init__(self, _id=0, **kwargs):
        super().__init__(_id, **kwargs)
        self.desc = []
        self.perf = 0 # TODO: performance unify
        
        self.children = []
        
    def __str__(self):
        if len(self.children) == 0:
            return f'Solution {self._id}'
        else:
            return f'Solution {self._id}: ' + '{' + ', '.join([str(sol) for sol in self.children]) + '}'
    
    def __repr__(self):
        return f'Solution {self._id}'
        
