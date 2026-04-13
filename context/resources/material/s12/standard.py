import numpy
from sklearn.preprocessing import StandardScaler


data = [[0,0],
        [0,1],
        [1,0],
        [1,1]]

sc = StandardScaler()
print(sc.fit_transform(data))