# Sparse-AE
Sparse Autoencoders and k-means Clustering on MNIST Dataset
Overview
This project demonstrates how to implement Sparse Autoencoders and perform k-means clustering on the MNIST digit dataset. Sparse Autoencoders are a type of neural network used for unsupervised learning and feature extraction. The k-means clustering algorithm is then applied to the learned embeddings to group similar data points together.

# Requirements
To run the code in this project, you'll need the following libraries:

1. Python 3.x
2. PyTorch
3. NumPy
4. Matplotlib
5. Scikit-learn

 Install the required libraries using pip: pip install torch numpy matplotlib scikit-learn
## Steps
# Data Preprocessing

Download the MNIST dataset from http://yann.lecun.com/exdb/mnist/.
Preprocess the data by converting pixel values to a range between 0 and 1 and splitting it into training and test sets.
Build and Train Sparse Autoencoders

# Implement a Sparse Autoencoder neural network using PyTorch.
Train the Autoencoder on the MNIST training set using Mean Squared Error (MSE) loss and L1 regularization to enforce sparsity.
Extract Embeddings

Extract embeddings from the trained Autoencoder for both the training and test sets.
# Perform k-means Clustering

Apply k-means clustering on the embeddings.
Set the number of clusters equal to the number of classes in the MNIST dataset (i.e., 10 clusters for digits 0 to 9).
Evaluate the performance of k-means using the available labels in the dataset.

## code example:
# Step 1: Data Preprocessing
 - Download the MNIST dataset and preprocess it.
 - Split the data into training and test sets.

# Step 2: Build and Train Sparse Autoencoder
 - Implement the Sparse Autoencoder model using PyTorch.
 - Train the model on the MNIST training set.

# Step 3: Extract Embeddings
 - Extract embeddings from the trained Autoencoder for both training and test sets.

# Step 4: Perform k-means Clustering
 - Apply k-means clustering on the embeddings.
 - Set the number of clusters equal to the number of classes in MNIST (10 clusters).
 - Evaluate the performance using the available labels in the dataset.

# Running the Code
Ensure you have installed the required libraries as mentioned in the Requirements section.

Run the code in your preferred Python environment.

The program will print the training set accuracy and test set accuracy after performing k-means clustering on the embeddings.

# Conclusion
This project demonstrates how to implement Sparse Autoencoders and perform k-means clustering on the MNIST dataset. The k-means algorithm can be used for clustering tasks, while the Sparse Autoencoder helps in learning useful embeddings for the data. By clustering the embeddings, we can effectively group similar data points together based on their learned representations.
