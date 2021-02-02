

import numpy as np


class Tree():
    
    def __init__(self, max_depth=5, min_size=1):
        
        self.max_depth = max_depth
        self.min_size = min_size
        
        self.mean = None
        self.column = None
        self.threshold = None
        
        self.left = None
        self.right = None
        
    def var(self, a):
        
        return np.sum(np.square(a - a.mean()))
        
        
    def fit(self, x, y):
        
        self.mean = y.mean()
        
        if self.max_depth <= 0:
            return
        
        error = self.var(y)
        

        for col in range(x.shape[1]):
            
            index = np.argsort(x[:, col])

            for i in range(0, x.shape[0] - 1):
                
                thres = x[index, col][i]
                
                if i < x.shape[0] - 1:
                    if x[index, col][i] == x[index, col][i + 1]:
                        continue
                
                left_tree = y[index][:i+1]
                left_err = self.var(left_tree)
            
                right_tree = y[index][i+1:]
                right_err = self.var(right_tree)
                
                total_err = left_err / y.shape[0] + right_err / y.shape[0]
            
                if total_err < error:
                    if len(y[x[:, col] <= thres]) >= self.min_size and \
                        len(y[x[:, col] > thres]) >= self.min_size:
                            
                        error = total_err
                        
                        self.column = col
                        self.threshold = thres
                        
        if self.threshold == None:
            return
        
        index_left = (x[:, self.column] <= self.threshold)
        index_right = (x[:, self.column] > self.threshold)
        
        self.left = Tree(self.max_depth - 1, self.min_size)
        self.right = Tree(self.max_depth - 1, self.min_size)
        
        self.left.fit(x[index_left,:], y[index_left])
        self.right.fit(x[index_right,:], y[index_right])
        
        
    def __predict(self, x):
            
        if self.threshold == None:
            return self.mean
            
        if x[self.column] <= self.threshold:
            return self.left.__predict(x)
            
        else:
            return self.right.__predict(x)
            
            
    def predict(self, x):
            
        result = np.zeros(x.shape[0])
            
        for i in range(len(result)):
                
            result[i] = self.__predict(x[i])
              
        return result

    

