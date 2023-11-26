def PCA_high_dim(X, num_components):
    """Compute PCA for small sample size but high-dimensional features. 
    Args:
        X: ndarray of size (N, D), where D is the dimension of the sample,
           and N is the number of samples
        num_components: the number of principal components to use.
    Returns:
        X_reconstruct: (N, D) ndarray. the reconstruction
        of X from the first `num_components` pricipal components.
    """
    N, D = X.shape
    X_normalized, mean = normalize(X)

    M = np.dot(X_normalized, X_normalized.T) / N

    S = (X_normalized.T @ X_normalized) / N
    eig_vals, eig_vecs = eig(M)
    eig_vecs = (X_normalized.T @ eig_vecs)

    eig_vals = eig_vals[:D]
    eig_vecs = eig_vecs[:,:D]
    
    principal_values = eig_vals[:num_components]

    principal_components = eig_vecs[:,:num_components]
    principal_components = np.real(principal_components)

    reconst = ((projection_matrix(principal_components) @ X_normalized.T).T) + mean
    return reconst, mean, principal_values, principal_components
