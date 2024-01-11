import numpy as np

class OneHotEncoder:
    def __init__(self):
        self.unique_labels = None       
    
    def transform(labels):

        max_label = np.max(labels)

        labels_onehot = np.zeros((len(labels), max_label + 1))
        
        for i, label in enumerate(labels):
            labels_onehot[i, label] = 1
        
        return labels_onehot

