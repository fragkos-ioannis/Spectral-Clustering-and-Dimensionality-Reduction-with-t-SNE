import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import NearestCentroid
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import eigs  
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_and_preprocess_mnist(sample_size=10000, test_size=0.2, random_state=42):
    mnist = fetch_openml('mnist_784', version=1)
    
    X, y = mnist.data[:sample_size], mnist.target[:sample_size].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    return X_train, X_test, y_train, y_test

def load_and_preprocess_cifar10(sample_size=10000):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train, y_train = X_train[:sample_size], y_train[:sample_size]
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1)) 
    X_train = X_train.astype('float32') / 255.0        
    X_test = X_test.astype('float32') / 255.0

    print(f"Number of training samples: {X_train.shape}")
    print(f"Number of test samples: {X_test.shape}")
    print(f"Number of training targets: {y_train.shape}")
    return X_train, X_test, y_train, y_test


def applyTSNEAndPlot(X, y, n_components=2, X_test=None):
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=100)
    X_tsne = tsne.fit_transform(X)
    
    if X_test is not None:
        X_test_tsne = tsne.fit_transform(X_test)

    plt.figure(figsize=(3, 3))
    scatter = plt.scatter(
        X_tsne[:, 0], 
        X_tsne[:, 1], 
        c=y, 
        cmap=plt.cm.tab10, 
        s=10, 
        alpha=0.8
    )
    plt.colorbar(scatter, label='Digit Label')
    plt.title('2D Representation of Data via t-SNE')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()
    plt.show()

    if X_test is not None:
        return X_tsne, X_test_tsne
    else:
        return X_tsne

def spectralClusteringAndPlot(data, n_clusters, n_neighbors):
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(data)
    adjacency_matrix = knn.kneighbors_graph(data).toarray()

    adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)

    degree_matrix_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(adjacency_matrix, axis=1)))
    normalized_laplacian = np.eye(len(data)) - degree_matrix_inv_sqrt @ adjacency_matrix @ degree_matrix_inv_sqrt

    eigenvalues, eigenvectors = eigs(normalized_laplacian, k=min(len(data), 50))
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42
    )
    clustering.fit(adjacency_matrix)
    labels = clustering.labels_

    plt.figure(figsize=(3, 3))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=plt.cm.get_cmap('viridis', n_clusters), s=50)
    plt.title(f'Spectral Clustering (Graph-Cut) with {n_clusters} Clusters')
    plt.xlabel('Isomap Component 1')
    plt.ylabel('Isomap Component 2')
    plt.colorbar(label='Cluster')
    plt.show()

    plt.figure(figsize=(3, 3))
    plt.scatter(range(len(eigenvalues)), eigenvalues, color='blue', marker='o')
    plt.title('Eigenvalues of the Laplacian Matrix (Sorted)')
    plt.xlabel('Index of Eigenvalue')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.show()

    return labels, clustering

def calculate_silhouette_score(data, labels):
    print(f"Shape of data: {data.shape}")
    print(f"Shape of labels: {labels.shape}")
    print(f"Labels: {labels[:10]}")

    if labels.ndim != 1:
        labels = np.ravel(labels)

    if data.ndim != 2:
        raise ValueError("Data must be a 2D array with shape (n_samples, n_features).")

    score = silhouette_score(data, labels)
    print(f"Silhouette Score: {score:.4f}")
    return score


def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_mnist(sample_size=1000)
    #print(X_train.shape, X_test.shape)
    #X_train, X_test, y_train, y_test = load_and_preprocess_cifar10(1000)
    print(X_train.shape, X_test.shape)
   
    X_tsne_2D = applyTSNEAndPlot(X_train, y_train, n_components=2, X_test=None)

    k_clusters = [5, 8, 10, 12]

    for i in k_clusters:
        
        labels = spectralClusteringAndPlot(X_tsne_2D, i, 20)
        if isinstance(labels, tuple):
            labels = labels[0]
        silhouette = calculate_silhouette_score(X_tsne_2D, labels)

if __name__ == "__main__":
    main()


