import os
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.data.shape)
print(os.path.abspath(__file__))
