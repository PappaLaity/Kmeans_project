from matplotlib import pyplot as plt
import numpy as np


class KMeans_plus:
    def __init__(self, K=4, max_iters=100, init="default", plot_steps=False):
        self.K= K
        self.max_iters = max_iters
        self.nb_iters = 0
        self.plot_steps = plot_steps
        self.centroids = []
        self.clusters = []
        self.labels = []
        self.init = init
        self.losses = []

    def random_init(self):
      self.n_samples, self.n_features = self.X.shape
      idx = np.random.choice(self.n_samples,self.K)
      self.centroids= self.X[idx]


    def kmean_plus_centroid_init(self,):
      self.centroids = []
      self.n_samples, self.n_features = self.X.shape
      idx = np.random.choice(self.n_samples)
      self.centroids.append(self.X[idx])
      for i in range(1,self.K):
        distances = []
        for elt in self.X :
          min = np.linalg.norm(elt - self.centroids[0])
          for center in self.centroids:
            if np.linalg.norm(elt - center) < min :
              min = np.linalg.norm(elt - self.centroids[0])
          distances.append(min)
        distances = np.array(distances)
        proba = distances / distances.sum()
        centroid = np.random.choice(self.n_samples, p=proba)
        self.centroids.append(self.X[centroid])

      return self


    def predict(self, X):
        self.X = X

        # Initialize
        if self.init == "Kmeans++":
          self.kmean_plus_centroid_init()
        else:
          self.random_init()

        # Optimize clusters
        for i in range(self.max_iters):
          self.nb_iters+=1
          self.centroids_old = self.centroids
          self.clusters = self._create_clusters()
          if self.plot_steps:
            self.plot()
          self.centroids =self._get_centroids()
          loss = self.loss()
          print(f"Epoch {self.nb_iters} / {self.max_iters} -- Loss: {loss}")
          self.losses.append(loss)
          if self._is_converged(self.centroids_old,self.centroids):
            break

        # Classify samples as the index of their clusters
        print(f"The model converge after {self.nb_iters} iterations.")
        labels = self._get_cluster_labels()

        return labels

    def _get_cluster_labels(self):
        # each sample will get the label of the cluster it was assigned to
        labels = []
        for elt in range(self.n_samples):
          for idx, cluster in enumerate(self.clusters):
            if elt in cluster:
              labels.append(idx)
        return labels


    def _create_clusters(self):
        # Assign the samples to the closest centroids to create clusters
        # remind the clusters is composed of the idx of datapoint and not datapoint itself
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
        # assign mean value of clusters to centroids
        centroids = []
        for elt in self.clusters:
          centroids.append(np.mean(self.X[elt],axis=0))
        return np.array(centroids)

    def _is_converged(self, centroids_old, centroids):
        # distances between each old and new centroids, fol all centroids
        return np.sum(centroids - centroids_old) == 0

    def loss(self):
      # Compute the loss function for the model
      sumCluster = 0
      loss = 0
      for i, centroid in enumerate(self.centroids):
        sumCluster = 0
        for idx in self.clusters[i]:
          sumCluster+= self.euclidean_distance(self.X[idx],centroid)**2
          # sumCluster+= np.linalg.norm(x - centroid ,2)**2
        loss+= sumCluster
      return loss
      

    def euclidean_distance(self,x1,x2):
      return np.linalg.norm(x2-x1,ord=2)


    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plotting points for each cluster
        for i, cluster in enumerate(self.clusters):
            if len(cluster) > 0:
                points = self.X[cluster]
                ax.scatter(points[:, 0], points[:, 1])

        # Plotting centroids
        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()