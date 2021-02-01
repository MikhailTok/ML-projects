

import ClassificationTree as CT

import numpy as np

import pandas as pd


class RandomForest():
    
    def __init__(self, max_depth=3, n_estimators = 50, min_size=10):
        
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_estimators = n_estimators
        
        self.estimators = [CT.ClassificationTree(self.max_depth, self.min_size) for _ in range(self.n_estimators)]
        
        
    def get_indices(X_train):
        
        indices = np.random.randint(0, X_train.shape[0], X_train.shape[0])
        
        return indices
        
    def fit(self, X_train, Y_train):
        
        for i in range(self.n_estimators):
            
            indx = RandomForest.get_indices(X_train)
            
            self.estimators[i].fit(X_train[indx, :], Y_train[indx])
             
            
    def predict(self, X):
        
        predictions = np.zeros((self.n_estimators, X.shape[0]))
        result = np.zeros(X.shape[0])
        
        for i in range(self.n_estimators):
            
            predictions[i] = self.estimators[i].predict(X)

        for j in range(predictions.shape[1]):
            
            a = {}
            counter = 0

            for k in predictions[:, j]:
                
                if k in a:
                    a[k] += 1
                else:
                    a[k] = 1
                    
                if a[k] > counter:
                    counter = a[k]
                    result[j] = k
                    
        return result.astype(int)
            
        
# data = pd.read_csv("/t_data/train.csv")
# X = data[['Pclass', 'SibSp', 'Fare']].to_numpy()
# y = data['Survived'].to_numpy()

# test = RandomForest(max_depth=1, n_estimators = 5, min_size=10)
# test.fit(X, y)

# # print(test.iteration)
# # print(test.flag)

# # plt.plot(test.error)

# print(X[:2,])
# print()
# print('result', test.predict(X[:2,]))

# #print(test.predict_proba(X[:4,]))

    

# If int, then consider max_features features at each split.

# If float, then max_features is a fraction and round(max_features * n_features) features are considered at each split.

# If “auto”, then max_features=sqrt(n_features).

# If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).

# If “log2”, then max_features=log2(n_features).

# If None, then max_features=n_features.


    
    