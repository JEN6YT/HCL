import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold


class DualityRLearner:
    def __init__(self, B, alpha=0.01, max_iter=1000, tol=1e-6, ridge_alpha=1.0):
        """
        Initializes the Duality R-Learner.

        Parameters
        ----------
        B : float
            Budget constraint.
        alpha : float
            Step size for lambda updates.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Tolerance for convergence.
        ridge_alpha : float
            Regularization parameter for ridge regression.
        """
        self.B = B
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.ridge_alpha = ridge_alpha

        # Model attributes
        self.lamda_ = None
        self.z_ = None
        self.tau_r_model = None
        self.tau_c_model = None
        self.tau_r = None
        self.tau_c = None

    def _optimize_z(self, tau_r, tau_c ,lbd):
        s = tau_r - lbd * tau_c
        z = np.where(s >= 0, 1, 0)
        return z

    def _optimize_lambda(self, lamda, B, alpha, z, tau_c):
        # Update lambda using the dual ascent step
        lamda = lamda + alpha * (B - np.sum(z * tau_c))
        return lamda

    def fit(self, X, y_r, y_c, w):
        """
        Fit the Duality R-Learner model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,)
            Observed outcomes for the scenario.
            label array with columns [cost, value, is_treatment]

        Returns
        -------
        self : object
            Fitted estimator.
        """

        n_samples = X.shape[0]

        # Initialize lambda and z
        lamda = 0.0
        z = np.zeros(n_samples)

        # Initialize the RLearner model
        tau_r_model = RLearner_rorc()
        tau_c_model = RLearner_rorc()

        # Initial fit
        tau_r_model.fit(X, y_r, w)
        tau_c_model.fit(X, y_c, w)
        
        tau_r = tau_r_model.predict(X)
        tau_c = tau_c_model.predict(X)

        for iteration in range(self.max_iter):
            
            # Step 1: Update z
            z_new = self._optimize_z(tau_r=tau_r, tau_c=tau_c, lbd=lamda)

            # Step 2: Update lambda
            lamda_new = self._optimize_lambda(lamda, self.B, self.alpha, z_new, tau_c)

            # Step 3: Update tau
            # According to the methodology, construct pseudo-outcomes.
            # This is a simplified version:

            # Refit tau
            tau_r_model.fit(X, y_r, w)
            tau_c_model.fit(X, y_c, w)
            
            tau_r_new = tau_r_model.predict(X)
            tau_c_new = tau_c_model.predict(X)

            # Check for convergence: monitor lambda difference
            lamda_diff = np.abs(lamda_new - lamda)
            if lamda_diff < self.tol:
                # Converged
                lamda = lamda_new
                z = z_new
                tau_r = tau_r_new
                tau_c = tau_c_new
                break

            # Update parameters for next iteration
            lamda = lamda_new
            z = z_new
            tau_r = tau_r_new
            tau_c = tau_c_new

        # Store results in class attributes
        self.lamda_ = lamda
        self.z_ = z
        self.tau_r_model_ = tau_r_model
        self.tau_c_model_ = tau_c_model
        self.tau_r = tau_r
        self.tau_c = tau_c

        return self
    
    def predict(self, X):
        """
        Predict method for the Duality R-Learner.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix for which we want to predict.
        
        Returns
        -------
        z_pred : ndarray of shape (n_samples,)
            Predicted selection indicators, where z_pred[i] is 1 if we should choose
            the action for the i-th instance, and 0 otherwise.
        tau_r_pred : ndarray of shape (n_samples,)
            Predicted \tau_r(X) values.
        tau_c_pred : ndarray of shape (n_samples,)
            Predicted \tau_c(X) values.
        """
        # Predict tau_r and tau_c using the trained models
        tau_r_pred = self.tau_r_model_.predict(X)
        tau_c_pred = self.tau_c_model_.predict(X)

        # Compute z based on the rule z = 1 if tau_r(x) - lambda * tau_c(x) >= 0
        s = tau_r_pred - self.lamda_ * tau_c_pred
        z_pred = np.where(s >= 0, 1, 0)

        lamda_value = self.lamda_ 
        
        return z_pred, tau_r_pred, tau_c_pred, lamda_value

    
    
class RLearner_rorc:
    """
    R-learner, estimate the heterogeneous causal effect
    replicate paper: https://arxiv.org/pdf/1712.04912.pdf
    Github R reference: https://github.com/xnie/rlearner
    """    
    
    def __init__( 
            self, 
            p_model_specs=None, 
            # ToDo: change the default to Ridge regression as OLS will have explosive coefficients
            m_model_specs={'model': linear_model.Ridge, 'params': {'alpha': 1.0}},
            tau_model_specs={'model': linear_model.Ridge, 'params': {'alpha': 1.0}},
            shadow=None,
            k_fold=5,
    ):
        """
        Constructor
        :param p_model_specs: a dictionary of model and hyper-params, specification for the model of E[W|X],
        propensity of the sample in treatment, if None, assume perfect randomized experiment and will use a constant p
        calculated from is_treatment from y
        :param m_model_specs: specification for the model of E[Y|X], example args
        {'model': linear_model.Ridge, 'params': {'alpha': 1.0}}
        :param tau_model_specs: specification for the model of E[Y(1) - Y(0)|X]
        :param shadow: shadow scale for objective cost - shadow * value
        :param k_fold: number of folds to use k-fold to predict p_hat and m_hat
        """
        
        self.p_model_specs = p_model_specs
        self.m_model_specs = m_model_specs
        self.tau_model_specs = tau_model_specs
        
        # self.p_model = None
        # self.m_model = None
        self.tau_model = None
        
        self.shadow = shadow
        self.k_fold = k_fold
    
    def _fit_predict_p_hat(self, X, w): 
        """ 
        Given X and T, fix a model to predict T (e) given X
        Fit and predict for p_hat.
        :param X: feature matrix 
        :param w: binary indicator for treatment / control 
        :return: a numpy array of predicted p_hat, same shape as w 
        """ 
        
        if self.p_model_specs is None: 
            return np.sum(w) / float(len(w)) * np.ones_like(w)
        
        kf = KFold(n_splits=self.k_fold)
        
        p_hat = np.zeros_like(w)
        
        # initialize m model 
        p_model = self.p_model_specs['model'](**self.p_model_specs['params']) 
        
        for fit_idx, pred_idx in kf.split(X):
            
            # split data into fit and predict
            X_fit, X_pred = X[fit_idx], X[pred_idx]
            w_fit = w[fit_idx]
                        
            p_model.fit(X_fit, w_fit)
            p_hat[pred_idx] = p_model.predict(X=X_pred)
        
        p_hat = np.clip(p_hat, 0 + 1e-7, 1 - 1e-7) 
        
        return p_hat 
    
    def _fit_predict_m_hat(self, X, m):
        """
        Given X and m, fit a model to predict m given X.
        Fit and predict for m_hat
        :param X: feature matrix
        :param m: cost or value
        :return: a numpy array of predicted m_hat, same shape as m
        """
        # ToDo: add hyper-param tuning for m_hat model here
        
        kf = KFold(n_splits=self.k_fold)
        
        m_hat = np.zeros_like(m)
        
        # initialize m model
        self.m_model = self.m_model_specs['model'](**self.m_model_specs['params'])
        
        for fit_idx, pred_idx in kf.split(X):
            
            # split data into fit and predict
            X_fit, X_pred = X[fit_idx], X[pred_idx]
            m_fit = m[fit_idx]
            
            self.m_model.fit(X_fit, m_fit)
            m_hat[pred_idx] = self.m_model.predict(X=X_pred)
                
        return m_hat
    
    def fit(self, X, y, w, sample_weight=None, **kwargs):
        """
        Fit
        :param X: feature matrix
        :param y: label array with columns (cost or revenue)
        :param sample_weight:
        :return: None
        """
        
        y_ = y
        
        p_hat = self._fit_predict_p_hat(X, w)
        m_hat = self._fit_predict_m_hat(X, y_)
        
        r_pseudo_y = (y_ - m_hat) / (w - p_hat)
        r_weight = np.square(w - p_hat)
        
        self.tau_model = self.tau_model_specs['model'](**self.tau_model_specs['params'])
        
        # fit tau model with pseudo y and sample weights 
        self.tau_model.fit(X, r_pseudo_y) 
        #self.tau_model.fit(X, r_pseudo_y, sample_weight=r_weight) 
        
        return None
    
    def predict(self, X, **kwargs):
        """
        Predict
        :param X: feature matrix
        :return: - predicted tau, aka -(cost - shadow * value)
        """
        return self.tau_model.predict(X)
    
    def get_params(self):
        """
        :return: dictionary of hyper-parameters of the model.
        """
        return {
            'p_model_specs': self.p_model_specs,
            'm_model_specs': self.m_model_specs,
            'tau_model_specs': self.tau_model_specs,
            'shadow': self.shadow,
            'k_fold': self.k_fold,
        }

    @property
    def coef_(self):
        """
        Estimated coefficients for tau model
        :return: array, shape (n_features, )
        """
        return self.tau_model.coef_