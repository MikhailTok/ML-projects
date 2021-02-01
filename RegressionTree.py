

import numpy as np
import pandas as pd

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


'''data = pd.read_csv('bikes_rent.csv')

X_train = data.drop(['cnt'], axis=1).values
Y_train = data['cnt'].values

#X_train = X_train[:, -1].reshape(X_train.shape[0], 1)       
            
error_test = []


for i in range(1, 25):
    test = Tree(max_depth=i, min_size=10)
    test.fit(X_train, Y_train)
    prediction = test.predict(X_train)
    
    error_test.append(np.sum(np.square(prediction - Y_train)) / prediction.shape[0])
    print(error_test[-1])


test = Tree(max_depth=3)
X = np.array([2, 2, 3, 2, 2])

Y = np.array([1, 1, 3, 4, 4])

test.fit(X, Y)'''



#print(error_test)
'''
 2284159.876900757
1189358.0939978634
785164.9924267998
581155.1913870642
432491.8631040201
317558.1490250252
212043.19428007948
144416.4936238276
80623.89870280989
50902.711824960264
32582.581866278684
20593.522452900495
13317.787547005119
7771.248664853539
4109.520767376717
1482.7940036479706
404.4747117451631
109.74110807113543
15.223666210670315
1.2982216142270862
0.0
0.0
0.0
0.0   '''        
            
    

