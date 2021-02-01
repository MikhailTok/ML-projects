

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LogisticRegression:
    
    def __init__(self, fit_intercept=True, penalty='none', C=1, solver='gd',
                 l_r=1e-3, tolerance=1e-6, max_iterarion=1e8, gamma = 0.05,
                 stop_criteria='logloss', calcul_logloss=True,
                 random_state = 123):
        
        """Logistic Regression  based on gradient descent solver 
            with options L1 and L2 regulations.
        
        
        Parameters
        ----------
        
        fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.

        penalty : {'none', 'l2', 'l1'}, default='none'
        Used to specify the norm used in the penalization.
        
        C : float, default=1.0
        Regularization strength; must be a positive float.
        
        solver : {'gd', 'sgd'}, default='gd'
        Used to specify the optimizer 
        
        l_r : float, default=1e-4
        The step-size for gradient descent; must be a positive float.
        
        tol : float, default=1e-6
        Precision of the solution.
        
        max_iter : int, default=1e7
        Maximum number of iterations for gradient descent

        
        stop_criteria: {'logloss', 'weights'}, default='logloss'
        Stop criteria for gradient descent
        
        calcul_logloss : bool, default=True
        Whether to calculate LogLoss for each iteration of gradient descent steps
        
        
        Attributes
        ----------
        coef_ : array of shape (n_features, 1)
            Estimated coefficients for the linear regression problem.

        intercept_ : float
            Independent term in the linear model. Set to 0.0 if
            `fit_intercept = False`
            
        error : list
            LogLoss error through gradient descent steps
        """
        
        self.fit_intercept = fit_intercept
        self.penalty = penalty
        self.C = C
        self.solver = solver
        
        self.l_r = l_r
        self.tolerance = tolerance
        self.gamma = gamma
        self.max_iteration = max_iterarion
        self.stop_criteria = stop_criteria
        self.calcul_logloss = calcul_logloss
        
        self.intercept_ = None
        self.coef_ = None
        self.flag = None
        self.error = []
        self.ema = []
        np.random.seed(random_state)
    
    
    def sigmoid(self, x, w):
        return 1 / (1 + np.exp(-(x @ w[:, np.newaxis])))


    def get_index(self, x):
        return np.random.randint(x.shape[0])
    
    
    def gradient(self, x, w, y, ind=None):
        
        if self.solver == 'gd':
            grad = np.sum((self.sigmoid(x, w) - y) * x, axis=0)
            
        if self.solver in ['sgd', 'sag']:
            grad = (self.sigmoid(x[ind], w) - y[ind]) * x[ind]

        if self.penalty == 'l2':
            if self.fit_intercept:
                w[0] = 0
            grad = self.C * grad + w
                
        if self.penalty == 'l1':
            if self.fit_intercept:
                w[0] = 0
            w = np.sign(w)
            grad = self.C * grad + w

        return grad
    
    
    def logloss(self, x, w, y, ind=None):
        
        if ind == None:
            pos_index = (y != 0)[:,0]
            neg_index = (y == 0)[:,0]
            positive = - np.log(self.sigmoid(x[pos_index], w))
            negative = - np.log(1 - self.sigmoid(x[neg_index], w))
            log_loss = np.sum(positive, axis=0) + np.sum(negative, axis=0)
            return log_loss / len(y)
        
        else:
            if y[ind] != 0:
                return (- np.log(self.sigmoid(x[ind], w)))
            if y[ind] == 0:
                return (- np.log(1 - self.sigmoid(x[ind], w)))
            

    
    def fit(self, x, y):
        
        y = y[:, np.newaxis]
        
        if self.fit_intercept:
            x = np.hstack((np.ones(x.shape[0])[:, np.newaxis], x))
        
        w = np.random.normal(0, 0.1, x.shape[1])
        
        if self.solver == 'sag':
            grad_init = (self.sigmoid(x, w) - y) * x
            grad_average = np.sum(grad_init, axis=0) / y.shape[0]
            
        if self.stop_criteria == 'ema':
            ema = self.logloss(x, w, y) / y.shape[0]
            self.ema.append(ema)
        
        iteration = 0
        
        while iteration < self.max_iteration:
            
            w_prev = np.copy(w)
                
            if self.solver == 'gd':
                w -= self.l_r * self.gradient(x, w, y)
            
            if self.solver == 'sgd':
                ind = self.get_index(x)
                w -= self.l_r * self.gradient(x, w, y, ind)


            if self.solver == 'sag':
                
                ind = self.get_index(x)
                grad_i = self.gradient(x, w, y, ind)
                grad_average -= grad_init[ind]
                grad_average += grad_i
                grad_init[ind] = grad_i

                w -= self.l_r * grad_average
                
            self.w = w
            self.iteration = iteration
                
                
            if self.calcul_logloss:
                self.error.append(self.logloss(x, w, y))
                
            if self.stop_criteria == 'ema':
                ema = (self.gamma * self.logloss(x, w, y, ind)
                                                  + (1 - self.gamma) * ema)
                self.ema.append(ema)
                if np.abs(self.ema[-1] - self.ema[-2]) < self.tolerance:
                        self.flag = 'ema'
                        break
                
            if iteration > 1:
                if self.stop_criteria == 'logloss':
                    if np.abs(self.error[-1] - self.error[-2]) < self.tolerance:
                        self.flag = 'logloss'
                        break
                    
            if self.stop_criteria == 'weights':
                if np.linalg.norm(w_prev - w) < self.tolerance:
                    self.flag = 'weights'
                    break
            
            iteration += 1
        
        if self.fit_intercept:
            self.intercept_ = self.w[0]
            self.coef_ = self.w[1:]
        else:
            self.intercept_ = 0
            self.coef_ = self.w

        
    def predict(self, X):
        
        prediction = np.zeros(X.shape[0])
        
        if self.fit_intercept:
            X = np.hstack((np.ones(X.shape[0])[:,np.newaxis], X))
            
        res = ((X @ self.w[:,np.newaxis]) > 0)[:,0]
        prediction[res] = 1
        
        return prediction
    
    
    def predict_proba(self, X):
        
        if self.fit_intercept:
            X = np.hstack((np.ones(X.shape[0])[:,np.newaxis], X))
        
        prediction = self.sigmoid(X, self.w)
        
        return prediction


# data = pd.read_csv("/t_data/train.csv")
# X = data[['Pclass', 'SibSp', 'Fare']].to_numpy()
# y = data['Survived'].to_numpy()

# test = LogisticRegression(fit_intercept=True, penalty='l2', C=1.1, solver='sag',
#                 l_r=1e-6, tolerance=1e-5, max_iterarion=1e5, gamma = 0.08,
#                 stop_criteria='ema', calcul_logloss=True,
#                 random_state = 123)


# test.fit(X, y)

# print(test.iteration)
# print(test.flag)

# plt.plot(test.error)


# print(test.predict(X[:4,]))

# print(test.predict_proba(X[:4,]))










