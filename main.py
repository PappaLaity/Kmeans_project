"""
* Main Function
The aim of this function is to:
    
    - Load Datasets 
    - Test Kmeans Algorithm
    - Test Kmeans++ Algorithm
    - Compare and discuss there Results
    - Experiment Elbow Method to find Optimal K



"""

import numpy as np
import pandas as pd
from models.Kmean import KMeans
from models.Kmean_plus import KMeans_plus
from utils.functions import plot


if __name__ == "__main__":
    
    # Define Kmeans Model with Random Initialization
    kmeans = KMeans()
    # Define Kmeans Model with Specific Initialization
    kmeans_plus = KMeans_plus(init="Kmeans++")

    # Data Initialization for Elbow Method       
    K = np.arange(1,10)
    losses = []

    # Load Datasets 
    data = pd.read_csv('datasets/Mall_Customers.csv')
    
    # print(data.head())
    
    """
            CustomerID  Gender  Age  Annual Income (k$)  Spending Score (1-100)
        0           1    Male   19                  15                      39
        1           2    Male   21                  15                      81
        2           3  Female   20                  16                       6
        3           4  Female   23                  16                      77
        4           5  Female   31                  17                      40
    """
    #  Data Cleaning
    """
        First of All we can clean the data by:
          - Transform Categorical data to Numerical data (Male == 0 & Female == 1)
          - Remove the First Column (CustomerID)
    """

    #  Transform Categorical to Numerical Data
    data.Gender = data.Gender.map(lambda x : 0 if x == "Male" else 1)
    # Remove the first Column
    data = data.iloc[:,1:]
    # Coonvert data to numpy before calling the kmeans Algorithm
    data = data.to_numpy()
    
    # Kmeans & Kmeans++ Model 
    print("\n")
    print("Kmeans Model Training with K = 4 - Max Epochs = 100")
    labels_kmeans = kmeans.predict(data)
    # print(kmeans.nb_iters)
    # print(kmeans.losses)
    plot(np.arange(kmeans.nb_iters+1),kmeans.losses,"Epochs","Loss","Kmeans Algorithm")
    print("\n")
    
    print("\n")
    print("Kmeans++ Model Training with K = 4 - Max Epochs = 100")
    print("\n")
    labels_kmeans_plus = kmeans_plus.predict(data)
    plot(np.arange(kmeans_plus.nb_iters+1),kmeans_plus.losses,"Epochs","Loss","Kmeans++ Algorithm")
    print("\n")

    #Conclusion A good Initialization May help to Converge Faster

    # Elbow Method Considering the Kmeans++
    for idx in K:
        model = KMeans_plus(K=idx)
        model.predict(data)
        losses.append(model.loss())

    plot(K,losses,"K value","Loss","Elbow Method")
