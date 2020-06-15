import numpy as np
from collections import Counter



STARTING_LABEL = '*'        # Label of t=-1
STARTING_LABEL_INDEX = 0




    def get_feature_list(self, X, t):
        feature_list_dic = dict()
        for feature_string in self.feature_func(X, t):
            for (prev_y, y), feature_id in self.feature_dic[feature_string].items():
                if (prev_y, y) in feature_list_dic.keys():
                    feature_list_dic[(prev_y, y)].add(feature_id)
                else:
                    feature_list_dic[(prev_y, y)] = {feature_id}
        return [((prev_y, y), feature_ids) for (prev_y, y), feature_ids in feature_list_dic.items()]
