# PCA-implementation
Use PCA Algorithm to apply dimention reduction 

To use it, call the function PCA(X, num_components)

Args:

        X: ndarray of size (N, D), where D is the dimension of the data,
           and N is the number of datapoints
        
        num_components: the number of principal components to use.
    
    Returns:
        the reconstructed data, the sample mean of the X, principal values
        and principal components
