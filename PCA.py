import numpy as np

def normalize(X):
    """Normalize the given dataset X to have zero mean.
    Args:
        X: ndarray, dataset of shape (N,D) where D is the dimension of the data,
           and N is the number of datapoints
    
    Returns:
        (Xbar, mean): tuple of ndarray, Xbar is the normalized dataset
        with mean 0; mean is the sample mean of the dataset.
    """
    mu = np.mean(X, axis=0) 
    Xbar = X - mu           
    return Xbar, mu

def eig(S):
    """Compute the eigenvalues and corresponding eigenvectors
        for the covariance matrix S.
    Args:
        S: ndarray, covariance matrix

    Returns:
        (eigvals, eigvecs): ndarray, the eigenvalues and eigenvectors

    Note:
        the eigenvals and eigenvecs should be sorted in descending
        order of the eigen values
    """
    eigvals, eigvecs = np.linalg.eig(S)
    sort_indices = np.argsort(eigvals)[::-1] 
    return eigvals[sort_indices], eigvecs[:, sort_indices]

def projection_matrix(B):
    """Compute the projection matrix onto the space spanned by `B`
    Args:
        B: ndarray of dimension (D, M), the basis for the subspace
    
    Returns:
        P: the projection matrix
    """
    return B @ np.linalg.inv(B.T @ B) @ B.T 

# Function to use
def PCA(X, num_components):
    """
    Args:
        X: ndarray of size (N, D), where D is the dimension of the data,
           and N is the number of datapoints
        num_components: the number of principal components to use.
    Returns:
        the reconstructed data, the sample mean of the X, principal values
        and principal components
    """
    X_normalized, mean = normalize(X) # EDIT THIS
    S = (X_normalized.T @ X_normalized) / len(X_normalized) # EDIT THIS

    eig_vals, eig_vecs = eig(S)
    principal_vals, principal_components = eig_vals[:num_components], eig_vecs[:,:num_components]
    principal_components = np.real(principal_components) 
    
    projected_vector = (projection_matrix(principal_components) @ X_normalized.T).T
    reconst = projected_vector + mean
    return reconst, mean, principal_vals, principal_components
