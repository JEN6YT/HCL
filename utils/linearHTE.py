from Model.rlearner import RLearner
import numpy as np

class linearHTE:
    
    def __init__(self):
        self.rlearnermodel_O = RLearner() 
        self.rlearnermodel_C = RLearner() 
    
    def fit_rlearner(self, X_tr, O_tr, C_tr, w_tr): 
        
        ## fit the r-learner model 
        ## model from paper: "Quasi-Oracle Estimation of Heterogeneous Treatment Effects" 
        ## "https://arxiv.org/pdf/1712.04912.pdf" 
        ## code sources: 
        ## "https://code.uberinternal.com/diffusion/MASAP/browse/master/sapphire_optimization/models/targeting_models/rlearner/core/rlearner.py" 
        ## "https://github.com/xnie/rlearner" 
        ##   X_tr: training data, each row is a data vector 
        ##   O_tr: labels (rewards or value), vertical vector  
        ##   C_tr: cost labels, vertical vector 
        ##   w_tr: treatment labels {1, 0}, vertical vector 
        ## return: fitted model object 
        
        z = np.zeros([len(O_tr), 1]) 
        
        o = np.concatenate((np.reshape(O_tr, [-1, 1]), z), axis=1) 
        o = np.concatenate((o, np.reshape(w_tr, [-1, 1])), axis=1) 
        
        c = np.concatenate((np.reshape(C_tr, [-1, 1]), z), axis=1)
        c = np.concatenate((c, np.reshape(w_tr, [-1, 1])), axis=1) 
        
        self.rlearnermodel_O.fit(X_tr, o) 
        self.rlearnermodel_C.fit(X_tr, c) 
        
        return self.rlearnermodel_O.tau_model, self.rlearnermodel_C.tau_model