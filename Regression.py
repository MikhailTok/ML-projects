﻿
import numpy as np



class Regression:
    
    def __init__(self, fit_intercept=True, solver='gd', penalty ='none',
                 alpha=1, eta=1e-4, tol=1e-6, max_iter=1e7, standardize=False,
                 stop_criteria='mse', calcul_mse=True):
        
        """Linear Regression (least squares) based on gradient descent solver 
            with options L1 and L2 regulations 
            aka Lasso and Ridge Regression respectively.
        
        
        Parameters
        ----------
        
        fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
        
        solver : {'gd', 'sgd'}, default='gd'
        Used to specify the optimizer 
        
        penalty : {'none', 'l2', 'l1'}, default='none'
        Used to specify the norm used in the penalization.
        
        alpha : float, default=1.0
        Regularization strength; must be a positive float.
        
        eta : float, default=1e-4
        The step-size for gradient descent; must be a positive float.
        
        tol : float, default=1e-6
        Precision of the solution.
        
        max_iter : int, default=1e7
        Maximum number of iterations for gradient descent
        
        standardize : bool, default=False
        Standardize features by removing the mean and scaling to 
        unit variance
        
        stop_criteria: {'mse', 'weights'}, default='mse'
        Stop criteria for gradient descent
        
        calcul_mse : bool, default=True
        Whether to calculate MSE for each iteration of gradient descent steps
        
        
        Attributes
        ----------
        coef_ : array of shape (n_features, 1)
            Estimated coefficients for the linear regression problem.

        rank_ : int
            Rank of matrix `X`.

        intercept_ : float
            Independent term in the linear model. Set to 0.0 if
            `fit_intercept = False`
            
        error : list
            MSE error through gradient descent steps
        """
        
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.penalty = penalty
        self.alpha = alpha
        self.eta = eta
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.stop_criteria = stop_criteria
        self.calcul_mse = calcul_mse
        
        self.error = []
        self.coef_ = None
        self.intercept_ = None
        self.flag = None
        self.mean = None
        self.norm = None
        self.iteration = None


    def gradient(self, X, y, w):
        
        if self.solver == 'gd':
            grad = np.sum((np.dot(X, w) - y) * X, axis=0)
            grad = grad.reshape(X.shape[1], 1) / X.shape[0]
            
        if self.solver == 'sgd':
            ind = np.random.randint(X.shape[0])
            grad = (np.dot(X[ind], w) - y[ind]) * X[ind]
            grad = grad[:, np.newaxis]
        
        if self.penalty == 'none':
            
            return grad
        
        if self.penalty == 'l2':
            weights = w.copy()
            if self.fit_intercept:
                weights[0] = 0
            
            return grad + self.alpha * weights
                
        if self.penalty == 'l1':
            weights = w.copy()
            if self.fit_intercept:
                weights[0] = 0
            weights = np.sign(weights)
            
            return grad + self.alpha * weights
        
        
    def get_rank(X):
        return np.linalg.matrix_rank(X)

  
    def mse(X, y, w):
        return np.sum(np.square(np.dot(X, w) - y)) / X.shape[0]
                
    def fit(self, X, y):
        
        """Fit linear model.
        
        Parameters
        ----------
        X : ndarray
            Training data.
        y : ndarray
            Target labels.
        """

        y = y[:, np.newaxis]
            
        if self.standardize:
            self.mean = np.mean(X, axis=0)
            self.norm = np.std(X, axis=0)
            self.X = (X - self.mean) / self.norm
        
        if self.fit_intercept:
            inter_column = np.ones(X.shape[0])[:, np.newaxis]
            X = np.hstack((inter_column, X))

            
        w_init = np.random.normal(0, 0.1, X.shape[1])[:, np.newaxis]
            
        w = w_init
        
        iteration = 0
        
        while iteration < self.max_iter:
            
            w_prev = w.copy()
            
            w -= self.eta * self.gradient(X, y, w)
            
            if self.calcul_mse:
                self.error.append(Regression.mse(X, y, w))
            
            if self.stop_criteria == 'mse' and iteration != 0:
                if np.abs(self.error[-1] - self.error[-2]) < self.tol:
                    self.flag = 'mse'
                    break
                    
            if self.stop_criteria == 'weights':
                if np.linalg.norm(w_prev - w) < self.tol:
                    self.flag = 'weights'
                    break
            
            iteration += 1

        self.w = w
        
        self.iteration = iteration
        
        if self.fit_intercept:
            self.intercept_ = self.w[0]
            self.coef_ = self.w[1:]
        else:
            self.intercept_ = 0
            self.coef_ = self.w
            
    def predict(self, X):
        
        if self.fit_intercept:
            inter_column = np.ones(X.shape[0])[:, np.newaxis]
            X = np.hstack((inter_column, X))

        return np.dot(X, self.w)