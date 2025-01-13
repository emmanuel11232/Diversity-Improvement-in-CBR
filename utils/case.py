from similarity import *

class Case:
    attr_func = {'Task': sim_task, 'Case study type': sim_case_study_type, 'Case study': Levenshtein, 
                 'Online/Off-line': sim_on_offline, 'Input for the model': Levenshtein
                 }

    def __init__(self, *args, **kwargs):
        attributes = [None] * len(self.attr_func)
        for i, attr in enumerate(self.attr_func.keys()):
            if i < len(args):
                attributes[i] = args[i]
            if attr in kwargs:
                attributes[i] = kwargs[attr]
        
        self.attributes = attributes

    @classmethod
    def from_series(cls, series):
        return cls(**series.to_dict())
    

    def cal_similarity(self, case, weights=[1.0]*len(attr_func)):
        assert isinstance(case, Case)
        assert len(weights) == len(self.attr_func)

        sims = []
        for i, (attr, func) in enumerate(self.attr_func.items()):
            sims.append(func(self.attributes[i], case.attributes[i]))
        global_sim = sum([sim * weight for sim, weight in zip(sims, weights)]) / sum(weights)
        return global_sim