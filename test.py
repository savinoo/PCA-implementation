X = np.array([[3, 6, 7],
              [8, 9, 0],
              [1, 5, 2]])

reconst, mean, principal_vals, principal_components = PCA(X, 1)

print('Cheacking mean...')
mean_exp = np.array([4, 20 / 3, 3])
np.testing.assert_allclose(mean, mean_exp, rtol=1e-5)
print('Mean is computed correctly!')

print('Checking principal values...')
principal_vals_exp = np.array([15.39677773])
np.testing.assert_allclose(principal_vals, principal_vals_exp, rtol=1e-5)
print('Principal Values are computed correctly!')

print('Checking principal components...')
principal_components_exp = np.array([[-0.68811066],
                                     [-0.40362611],
                                     [ 0.60298398]])
np.testing.assert_allclose(principal_components, principal_components_exp, rtol=1e-5)
print("Principal components are computed correctly!")

print('Checking reconstructed data...')
reconst_exp = np.array([[ 1.68166528,  5.30679755,  5.03153182],
                        [ 7.7868029 ,  8.8878974 , -0.31833472],
                        [ 2.53153182,  5.80530505,  4.2868029 ]])
np.testing.assert_allclose(reconst, reconst_exp, rtol=1e-5)
print("Reconstructed data is computed correctly!")
