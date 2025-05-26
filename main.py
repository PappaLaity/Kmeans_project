"""
* Main Function

"""

import pandas as pd
from models.Kmean import KMeans
from models.Kmean_plus import KMeans_plus


if __name__ == "__main__":
    # kmeans = KMeans()
    # kmeans_plus = KMeans_plus()

    # Load Datasets 

    print("Hello From Main")
    datasets = pd.read_csv('datasets/Mall_Customers.csv')
    print(datasets.head())