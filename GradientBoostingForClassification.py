

import RegressionTree as RT
import numpy as np



import pandas as pd

import matplotlib.pyplot as plt


class GradientBoosting():
    
    def __init__(self, max_depth = 2, n_estimators = 10, learning_rate = 0.1, min_size=10):
        
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_estimators = n_estimators
        self.l_r = learning_rate
        
        self.estimators = []
        
        
    def sigmoid(self, p):
        
        return 1 / (1 + np.exp(-p))
    
    def init_prediction(self, target):
                
        dic = {}
        counter = 0
        
        for i in target:
            
            if i in dic:
                dic[i] += 1
            else:
                dic[i] = 1
                
            if dic[i] > counter:
                counter = dic[i]
                predict = i
            
        return predict


    def fit(self, X, y):
        
        self.init_pred = self.init_prediction(y)
        
        prediction = self.init_pred
        
        for i in range(self.n_estimators):
            
            if i == 0:
                residual = self.init_pred + np.zeros(y.shape[0])
            else:
                residual = - self.sigmoid(prediction) + y
                
                
            tree = RT.Tree(self.max_depth, self.min_size)
            
            tree.fit(X, residual)
            
            self.estimators.append(tree)
            
            prediction += self.l_r * tree.predict(X)

                        
    def predict(self, X):
        
        result = np.zeros(X.shape[0]) + self.init_pred
        
        for i in range(self.n_estimators):
            
            result += self.l_r * self.estimators[i].predict(X)
            
            
        pred = np.zeros(X.shape[0])
        
        pred[result > 0] = 1
            
        return pred.astype(int)
    
    def predict_proba(self):
        pass
            
            
# data = pd.read_csv("/t_data/train.csv")
# X = data[['Pclass', 'SibSp', 'Fare']].to_numpy()
# y = data['Survived'].to_numpy()

# test = GradientBoosting(max_depth=1, n_estimators = 5, min_size=10)
# test.fit(X, y)

# # print(test.iteration)
# # print(test.flag)

# # plt.plot(test.error)

# print(X[:2,])
# print()
# print('result', test.predict(X[:2,]))

# #print(test.predict_proba(X[:4,]))


            
            