import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold

from Model.rlearner import RLearner

class DualRLearner:
    def fit_dual(self, X_tr, O_tr, C_tr, w_tr, lambd, m_model_specs = '', tau_model_specs = '', p_model_specs = ''):
        """
        fit the r-learner model 
        model from paper: "Quasi-Oracle Estimation of Heterogeneous Treatment Effects" 
        "https://arxiv.org/pdf/1712.04912.pdf" 
        code sources: 
        "https://code.uberinternal.com/diffusion/MASAP/browse/master/sapphire_optimization/models/targeting_models/rlearner/core/rlearner.py" 
        "https://github.com/xnie/rlearner" 
        
        :param X_tr: training data, each row is a data vector 
        :param O_tr: labels (rewards or value), vertical vector  
        :param C_tr: cost labels, vertical vector 
        :param w_tr: treatment labels {1, 0}, vertical vector 
        :return: fitted model object 
        """
        if m_model_specs == '' and tau_model_specs == '' and p_model_specs == '': 
            self.rlearnermodel_L = RLearner() 
        else: 
            self.rlearnermodel_L = RLearner(m_model_specs=m_model_specs, tau_model_specs=tau_model_specs, p_model_specs=p_model_specs)         
        
        y = np.concatenate((np.reshape(O_tr, [-1, 1]), np.reshape(C_tr, [-1, 1])), axis=1)
        y = np.concatenate((y, np.reshape(w_tr, [-1, 1])), axis=1) 
        
        self.rlearnermodel_L.fit(X_tr, y, lambd) 
        
        return self.rlearnermodel_L.tau_model
    
    def select_lambda(self, X_tr, O_tr, C_tr, w_tr, lambda_list, X_va):
        rlearnerscores = []
        for l in lambda_list:
            rl_ridge_model_L = self.fit_dual(X_tr, O_tr, C_tr, w_tr, l)
            rlearnerscores.append(rl_ridge_model_L.predict(X_va))
        return rlearnerscores