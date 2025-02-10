import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn import linear_model
from sklearn.model_selection import KFold
import numpy as np

class SimpleTCModelDNN(nn.Module):
    def __init__(self, input_dim, num_hidden,
                 p_model_specs={'model': linear_model.LogisticRegression, 'params': {'C': 1.0, 'max_iter': 1000}}, 
                 k_fold=5):
        """
        Direct Ranking Model.

        Args:
        - input_dim (int): Dimensionality of user context features.
        - num_hidden (int): Number of hidden units. If 0, skips hidden layer.
        - p_model_specs (dict): Model and hyperparams for estimating E[W|X].
        - k_fold (int): Number of folds for k-fold cross-validation to predict p_hat.
        """
        super().__init__()
        self.p_model_specs = p_model_specs
        self.k_fold = k_fold

        self.hidden_layer = nn.Linear(input_dim, num_hidden) if num_hidden > 0 else None
        self.ranker = nn.Linear(num_hidden if num_hidden > 0 else input_dim, 1)

    def fit_predict_p_hat(self, X, w):
        """
        Predict p_hat using k-fold cross-validation.

        Args:
        - X (numpy array): Feature matrix.
        - w (numpy array): Binary indicator for treatment (1) and control (0).

        Returns:
        - p_hat (numpy array): Predicted probabilities.
        """
        if self.p_model_specs is None:
            return np.full_like(w, np.mean(w), dtype=np.float32)

        p_hat = np.zeros_like(w, dtype=np.float32)
        kf = KFold(n_splits=self.k_fold)

        p_model = self.p_model_specs['model'](**self.p_model_specs['params'])
        
        for fit_idx, pred_idx in kf.split(X):
            p_model.fit(X[fit_idx], w[fit_idx])
            p_hat[pred_idx] = p_model.predict_proba(X[pred_idx])[:, 1]

        return np.clip(p_hat, 1e-7, 1 - 1e-7)

    def forward(self, D_tre, D_unt):
        """
        Forward pass for ranking.

        Args:
        - D_tre (torch.Tensor): Treatment group features.
        - D_unt (torch.Tensor): Control group features.

        Returns:
        - h_tre_rnkscore (torch.Tensor): Rank scores for treatment.
        - h_unt_rnkscore (torch.Tensor): Rank scores for control.
        """
        if self.hidden_layer:
            D_tre, D_unt = torch.tanh(self.hidden_layer(D_tre)), torch.tanh(self.hidden_layer(D_unt))
        
        return torch.tanh(self.ranker(D_tre)), torch.tanh(self.ranker(D_unt))

    def apply_top_k_mask(self, scores, top_k_percent):
        """
        Zero out lower ranks based on top-k percentage.

        Args:
        - scores (torch.Tensor): Rank scores.
        - top_k_percent (float): Percentage of top scores to retain.

        Returns:
        - masked_scores (torch.Tensor): Scores with lower-ranked entries zeroed.
        """
        k = max(1, int(top_k_percent * scores.shape[0]))  # Ensure at least one sample
        top_k_values, _ = torch.topk(scores.squeeze(), k)
        threshold = top_k_values[-1]  # Get lowest score in top-k
        return torch.where(scores >= threshold, scores, torch.tensor(0.0, device=scores.device))

    def compute_objective(self, h_tre, h_unt, c_tre, c_unt, o_tre, o_unt, e_hat, e_x_tre, e_x_unt, alpha, use_propensity):
        """
        Compute the cost-gain effectiveness objective.

        Args:
        - h_tre, h_unt (torch.Tensor): Rank scores for treatment/control.
        - c_tre, c_unt (torch.Tensor): Cost labels for treatment/control.
        - o_tre, o_unt (torch.Tensor): Order labels for treatment/control.
        - e_hat (float): Treatment proportion.
        - e_x_tre, e_x_unt (torch.Tensor): Estimated propensities.
        - alpha (float): Trade-off hyperparameter.
        - use_propensity (bool): Whether to apply propensity weighting.

        Returns:
        - obj (torch.Tensor): Objective function value.
        """

        obj = None
        s_tre = torch.exp(F.log_softmax(h_tre.squeeze(), dim=0))
        s_unt = torch.exp(F.log_softmax(h_unt.squeeze(), dim=0))

        if use_propensity:
            dc_tre = torch.sum((s_tre * c_tre) / e_x_tre)
            dc_unt = torch.sum((s_unt * c_unt) / (1 - e_x_unt))
            do_tre = torch.sum((s_tre * o_tre) / e_x_tre)
            do_unt = torch.sum((s_unt * o_unt) / (1 - e_x_unt))
            obj = (e_hat*do_tre - (1-e_hat)*do_unt) - alpha * (e_hat*dc_tre - (1-e_hat)*dc_unt) 
        else:
            s_tre, s_unt = F.softmax(h_tre.squeeze(), dim=0), F.softmax(h_unt.squeeze(), dim=0)
            dc_tre, dc_unt = torch.sum(s_tre * c_tre), torch.sum(s_unt * c_unt)
            do_tre, do_unt = torch.sum(s_tre * o_tre), torch.sum(s_unt * o_unt)
            obj = (do_tre - do_unt) - alpha * (dc_tre - dc_unt)

        return obj
    
    def optimize_model(self, X, w, D_tre, D_unt, c_tre, c_unt, o_tre, o_unt, lr=0.0001, epochs=10, alpha=0.5, use_propensity=True, top_k_percent=0.1):
        """
        Train the model.

        Args:
        - X, w (numpy array): Features and treatment indicators.
        - D_tre, D_unt (torch.Tensor): Features for treatment/control.
        - c_tre, c_unt, o_tre, o_unt (torch.Tensor): Labels for cost and order.
        - lr (float): Learning rate.
        - epochs (int): Number of training epochs.
        - alpha (float): Trade-off parameter.
        - use_propensity (bool): Whether to apply propensity weighting.
        - top_k_percent (list): List of top-k percentages for ranking.

        Returns:
        - best_obj (torch.Tensor): Best objective value.
        """
        optimizer = Adam(self.parameters(), lr=lr)
        e_x = torch.tensor(self.fit_predict_p_hat(X, w), dtype=torch.float32)

        treat_idx, untreat_idx = np.where(w == 1)[0], np.where(w == 0)[0]
        e_hat = len(treat_idx) / (len(treat_idx) + len(untreat_idx))

        e_x_tre, e_x_unt = e_x[treat_idx], e_x[untreat_idx]

        for epoch in range(epochs):
            h_tre, h_unt = self.forward(D_tre, D_unt)

            h_tre_masked, h_unt_masked = self.apply_top_k_mask(h_tre, top_k_percent), self.apply_top_k_mask(h_unt, top_k_percent)
            obj = self.compute_objective(h_tre_masked, h_unt_masked, c_tre, c_unt, o_tre, o_unt, e_hat, e_x_tre, e_x_unt, alpha, use_propensity)

            optimizer.zero_grad()
            (-obj).backward()  # Negative objective for maximization
            optimizer.step()

        return obj