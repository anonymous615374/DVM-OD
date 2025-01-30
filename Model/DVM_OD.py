import numpy as np
from scipy.spatial.distance import cdist


class DVM_OD:
    def __init__(self):
        """
        Initialize the DVM_OD object.
        Parameters:
        - 
        """
        self.npd = np.array([])  
        self.basepoint_X =  np.array([])  
    
    def compute_discriminants(self, X, y, k=5):
        """
        Compute discriminant directions using an algorithm based on within-class scatter and total scatter matrices.
        
        Parameters:
        - X (ndarray): Data matrix of shape (N, d) where N is the number of samples, and d is the number of features.
        - y (ndarray): Label array of shape (N,), containing class labels.
        - k (int): Number of eigenvectors to compute (default is 5).
        
        Returns:
        - theta (ndarray): Discriminant directions matrix of shape (d, k).
        """
        # Step 1: Compute within-class scatter (P_W) and scatter of the entire dataset (P_S)
        classes = np.unique(y)  # Unique class labels
        N, d = X.shape  # Number of samples (N) and features (d)

        # Reassign k = d to match the hyperparameter-free configuration
        k = d 
        
        mean_total = np.mean(X, axis=0)  # Compute the mean of all data points
        
        # Compute within-class scatter P_W
        P_W = []
        for cls in classes:
            X_c = X[y == cls]  # Data points for current class
            mean_cls = np.mean(X_c, axis=0)  # Mean for the current class
            P_W.append((X_c - mean_cls).T)  # Scatter of current class
            
        P_W = np.hstack(P_W)  # Combine scatter matrices from all classes
    
        # Compute scatter of the entire dataset P_S
        P_S = (X - mean_total).T  # Scatter from the global mean
    
        # Step 2: Eigenvalue decomposition for scatter matrices
        S_S = np.dot(P_S, P_S.T)  # Scatter matrix of the dataset
        eigvals_S, Q_S = np.linalg.eigh(S_S)  # Eigenvectors and eigenvalues of scatter matrix
        D = np.diag(eigvals_S + 1e-8)  # Diagonal matrix of eigenvalues, adding small value for stability
    
        # Step 3: Compute transformation matrix for eigenvalue decomposition
        transformation_matrix = np.dot(
            np.linalg.inv(D), 
            np.dot(Q_S.T, np.dot(P_W, np.dot(P_W.T, Q_S)))
        )
        
        eigvals, eigvecs = np.linalg.eigh(transformation_matrix)  # Eigenvalues and eigenvectors of transformation matrix
        
        # Select the top k eigenvectors based on the largest eigenvalues
        top_k_indices = np.argsort(eigvals)[-k:]  
        u = eigvecs[:, top_k_indices]  # Selected eigenvectors
    
        # Step 4: Compute discriminant directions (theta)
        theta = np.dot(Q_S, u)  # Discriminant directions matrix
        
        return theta
        
    def distance_vector(self, point_X, point_Y):
        """
        Compute pairwise Euclidean distances between two sets of points (point_X and point_Y).
        
        Parameters:
        - point_X (ndarray): First set of points (N_train, d).
        - point_Y (ndarray): Second set of points (N_test, d).
        
        Returns:
        - distance (ndarray): Matrix of pairwise distances (N_test, N_train).
        """
        # Compute squared Euclidean norms for each point
        norm_X = np.sum(point_X**2, axis=1)  # Norms for point_X
        norm_Y = np.sum(point_Y**2, axis=1)  # Norms for point_Y
        
        # Compute dot product between point_X and point_Y
        dot_product = np.dot(point_Y, point_X.T)  # Shape: (N_test, N_train)
        
        # Apply Euclidean distance formula
        distance = np.sqrt(abs(norm_Y[:, np.newaxis] + norm_X[np.newaxis, :] - 2 * dot_product))
        return distance
    
    def minimum_distance(self, A, B):
        """
        Compute the minimum Euclidean distance from each point in A to all points in B.
        
        Parameters:
        - A (ndarray): Set of points (N_A, d).
        - B (ndarray): Set of points (N_B, d).
        
        Returns:
        - min_distances (ndarray): Minimum distances from each point in A to the nearest point in B.
        """
        A = np.asarray(A)
        B = np.asarray(B)
        
        # Initialize an array for storing minimum distances
        min_distances = np.empty(A.shape[0], dtype=np.float64)
        
        # Iterate over each point in A and calculate minimum distance to points in B
        for i, a in enumerate(A):
            distances = cdist([a], B, metric='euclidean')  # Compute all pairwise distances
            min_distances[i] = np.min(distances)  # Store minimum distance
        
        return min_distances   
    
    def fit(self, X_train, y_train):
        """
        Fit the model using the training data (X_train and y_train).
        Add an artificial class 1 using the largest values in each feature column.
        
        Parameters:
        - X_train (ndarray): Training feature matrix.
        - y_train (ndarray): Training labels.
        """
        # Add an artificial class 1 using the largest values in each feature column
        largest_values = np.max(X_train, axis=0)  # Largest value in each feature column
        y_train = np.hstack([y_train, 1])  # Append class label '1'
        X_train = np.vstack([X_train, largest_values])  # Add the largest values as a new row
        
        # Compute the discriminants using the modified training data
        self.npd = self.compute_discriminants(X_train, y_train)
        
        # Compute point projections (mapping of training data to discriminants)
        self.basepoint_X = np.dot(X_train, self.npd)
            
    def predict(self, X_test):
        """
        Predict the class labels for the test data (X_test).
        Use distance computation to find the closest class.
        
        Parameters:
        - X_test (ndarray): Test feature matrix.
        
        Returns:
        - scores (ndarray): Anomaly scores indicating the distance to the closest class.
        """
        # Compute the point projections for the test data
        projected_point_X_t = np.dot(X_test, self.npd)
        
        # Use the minimum distance method for large datasets
        if projected_point_X_t.shape[0] > 50000:
            scores = self.minimum_distance(projected_point_X_t, self.basepoint_X)
        else:
            # Otherwise, compute Euclidean distances
            distest = self.distance_vector(self.basepoint_X, projected_point_X_t)
            scores = np.amin(distest, axis=1)  # Minimum distance for each test point
        
        return scores

    def transform(self, X):
        """
        Project input data X onto the discriminant directions.
        
        Parameters:
        - X (ndarray): Data matrix to be transformed (N, d).
        
        Returns:
        - Transformed data (ndarray): Projected data (N, k) onto the discriminant directions.
        """
        return np.dot(X, self.npd)  # Project data onto the discriminant directions
