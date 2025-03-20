# Spectral Clustering and Dimensionality Reduction with t-SNE

This project demonstrates the use of Spectral Clustering and t-SNE for dimensionality reduction and clustering analysis on datasets such as MNIST and CIFAR-10.

## Project Description

The goal of this project is to preprocess the MNIST or CIFAR-10 datasets, reduce their dimensions using t-SNE, and perform Spectral Clustering on the reduced data. The model evaluates clustering performance with different numbers of clusters and uses the silhouette score as a measure of cluster quality.

### Key Techniques

- **t-SNE**: t-Distributed Stochastic Neighbor Embedding (t-SNE) is used for dimensionality reduction to 2D space, making it easier to visualize high-dimensional data.
- **Spectral Clustering**: A clustering method that uses the eigenvalues of a similarity matrix to reduce the dimensions of the dataset and perform clustering.
- **Silhouette Score**: A metric used to evaluate the quality of clustering by measuring how similar objects are within their cluster compared to other clusters.

## Requirements

The following Python libraries are required:

- `numpy`
- `matplotlib`
- `tensorflow`
- `scikit-learn`
- `scipy`

You can install these libraries using `pip`:

```bash
pip install numpy matplotlib tensorflow scikit-learn scipy
