import pandas as pd
import numpy as np
from functools import reduce

class DTRuler(object):
    
    def __init__(self, model, training_data, features):
        self.model = model
        self.feature_names = features
        self.training_data = training_data.copy()
        self.n_nodes = self.model.tree_.node_count
        self.children_left = self.model.tree_.children_left
        self.children_right = self.model.tree_.children_right
        self.feature = self.model.tree_.feature
        self.threshold = self.model.tree_.threshold
        self.extract_rules()
        
    def find_path(self, node_numb, path, x):
        path.append(node_numb)
        if node_numb == x:
            return True
        left = False
        right = False
        if (self.children_left[node_numb] !=-1):
            left = self.find_path(self.children_left[node_numb], path, x)
        if (self.children_right[node_numb] !=-1):
            right = self.find_path(self.children_right[node_numb], path, x)
        if left or right :
            return True
        path.remove(node_numb)
        return False

    def get_rule(self, path, column_names):
        mask = []
        for index, node in enumerate(path):
            #We check if we are not in the leaf
            if index!=len(path)-1:
                sign = '<=' if (self.children_left[node] == path[index+1]) else '>'
                mask.append(f'{column_names[self.feature[node]]} {sign} {self.threshold[node]}')
        return mask
    

    def extract_rules(self):
        leave_id = self.model.apply(self.training_data[self.feature_names])

        paths = {}
        for leaf in np.unique(leave_id):
            path_leaf = []
            self.find_path(0, path_leaf, leaf)
            paths[leaf] = np.unique(np.sort(path_leaf))

        self.rules = {}
        for key in paths:
            self.rules[key] = self.get_rule(paths[key], self.training_data[self.feature_names].columns)
            
            
    def filter_data(self,data,rules):
        d = data.copy()
        return d[reduce(lambda acc,val: acc&val, [d.eval(r) for r in rules])]
    
    def predict(self,data, aggfunc=None):
        d = data.copy()
        d['y_pred'] = self.model.predict(d[self.feature_names])
        d['leave_id'] = self.model.apply(d[self.feature_names])
        d['tree_data'] = d['leave_id'].apply(lambda x: self.rules.get(x,None))
        d['tree_data'] = d['tree_data'].apply(lambda x: self.filter_data(self.training_data,x) if not isinstance(x,type(None)) else None)
        
        if not isinstance(aggfunc,type(None)):
            agg_fields = pd.DataFrame.from_dict(d['tree_data'].apply(aggfunc).values.tolist())
            d[agg_fields.columns.tolist()] = agg_fields.values
        return d
        
    

class DTRulerAvg(object):
    
    def __init__(self, model, training_data, features):
        self.model = model
        self.feature_names = features
        self.training_data = training_data.copy()
        self.estimators = [ DTRuler(model=x, training_data=self.training_data, features=self.feature_names) for x in self.model.estimators_]
         

    def avg_pred(self, data):
        d = data.copy()
        p = [x.model.predict(data[self.feature_names]) for x in self.estimators]
        return np.array(p).mean(axis=0)
    
    def predict(self,data, aggfunc=None):
        d = data.copy()
        cdfs = list()
        for i in range(len(d)):
            cdf = pd.concat([x.predict(d.loc[i].to_frame().T)['tree_data'].values[0] for x in self.estimators],ignore_index=True)
            cdfs.append(cdf)
            
        d['all_trees_data'] = cdfs
        
        if not isinstance(aggfunc,type(None)):
            agg_fields = pd.DataFrame.from_dict(d['all_trees_data'].apply(aggfunc).values.tolist())
            d[agg_fields.columns.tolist()] = agg_fields.values
        return d
