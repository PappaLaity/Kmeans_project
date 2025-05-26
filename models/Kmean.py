from matplotlib import pyplot as plt
import numpy as np


class KMeans:
    """
    
    """
    def __init__(self, K=4, max_iters=100, plot_steps=False):
        self.K= K
        self.max_iters = max_iters
        self.nb_iters = 0
        self.plot_steps = plot_steps
        self.centroids = []
        self.clusters = []
        self.labels = []

    def random_init(self):
      """
        Random Init Function
      """
      self.n_samples, self.n_features = self.X.shape
      idx = np.random.choice(self.n_samples,self.K)
      self.centroids= self.X[idx]


    def predict(self, X):
        """
        Predict Function
        """
        self.X = X

        # Initialize
        self.random_init()

        # Optimize clusters
        for i in range(self.max_iters):
          self.nb_iters+=1
          self.centroids_old = self.centroids
          self.clusters = self._create_clusters()
          if self.plot_steps:
            self.plot()
          self.centroids =self._get_centroids()
          self.loss()
          if self._is_converged(self.centroids_old,self.centroids):
            break

        # Classify samples as the index of their clusters
        print(f"The model converge after {self.nb_iters} iterations.")
        labels = self._get_cluster_labels()

        return labels

    def _get_cluster_labels(self):
        """
            Get labels for Clusters
        """
        # each sample will get the label of the cluster it was assigned to
        labels = []
        for elt in range(self.n_samples):
          for idx, cluster in enumerate(self.clusters):
            if elt in cluster:
              labels.append(idx)
        return labels


    def _create_clusters(self):
        """
            Create Cluster in basis of The distances with closest Centroid
        """
        # Assign the samples to the closest centroids to create clusters
        clusters = []
        for i in range(self.K):
          clusters.append([])
        for i in range(self.n_samples):
          sample = self.X[i]
          idx = self._closest_centroid(sample,self.centroids)
          clusters[idx].append(i)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        dists = [self.euclidean_distance(sample,centroid) for centroid in centroids ]
        return np.argmin(dists)

    def _get_centroids(self):
        """
            Get New Centroids functions after creating clusters
        """
        # assign mean value of clusters to centroids
        centroids = []
        for elt in self.clusters:
          centroids.append(np.mean(self.X[elt],axis=0))
        return np.array(centroids)

    def _is_converged(self, centroids_old, centroids):
        """
            Check the convergence of the Algorithm using the centroids criteria
        """
        # distances between each old and new centroids, fol all centroids
        return np.sum(centroids - centroids_old) == 0

    def loss(self):
      """
        Loss Function
      """
      # Compute the loss function for the model
      sumCluster = 0
      loss = 0
      for i, centroid in enumerate(self.centroids):
        sumCluster = 0
        for x in self.clusters[i]:
          sumCluster+= self.euclidean_distance(x,centroid)**2
          # sumCluster+= np.linalg.norm(x - centroid ,2)**2
        loss+= sumCluster
      print(f"Epoch {self.nb_iters} / {self.max_iters} -- Loss: {loss}")
      return self

    def euclidean_distance(self,x1,x2):
      """
        Euclidian distance
      """
      return np.linalg.norm(x2-x1,ord=2)


    def plot(self):
        """
            Plot Centroids Evolution during the process of learning
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plotting points for each cluster
        for i, cluster in enumerate(self.clusters):
            if len(cluster) > 0:
                points = X[cluster]
                ax.scatter(points[:, 0], points[:, 1])

        # Plotting centroids
        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()