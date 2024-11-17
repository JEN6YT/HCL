from Model.rlearner import RLearner
import numpy as np

class linearHTE:
    def fit_rlearner_lagrangian(self, X_tr, O_tr, C_tr, w_tr, lambd, m_model_specs = '', tau_model_specs = '', p_model_specs = ''): 
        
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
        
        ## process values_tr, zero cost vector and w_tr into rlearner input 
        if m_model_specs == '' and tau_model_specs == '' and p_model_specs == '': 
            self.rlearnermodel_L = RLearner() 
        else: 
            self.rlearnermodel_L = RLearner(m_model_specs=m_model_specs, tau_model_specs=tau_model_specs, p_model_specs=p_model_specs)         
        
        y = np.concatenate((np.reshape(O_tr, [-1, 1]), np.reshape(C_tr, [-1, 1])), axis=1)
        y = np.concatenate((y, np.reshape(w_tr, [-1, 1])), axis=1) 
        
        self.rlearnermodel_L.fit(X_tr, y, lambd) 
        
        return self.rlearnermodel_L.tau_model