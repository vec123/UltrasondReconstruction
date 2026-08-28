import numpy as np
from dataclasses import dataclass, field
from scipy.stats import chi2
from scipy import sparse

@dataclass
class StatisticalShapeModel:
    mean: np.ndarray                    # (3N,)
    eigenfunctions: np.ndarray         # (3N, K)
    eigenvalues: np.ndarray            # (K,)
    landmarks: dict

    @property
    def n_vertices(self):
        return self.mean.size // 3

    @property
    def n_components(self):
        return self.eigenfunctions.shape[1]

    def reconstruct(self, alpha: np.ndarray):
        return self.mean + self.eigenfunctions @ alpha

    def instance(self, alpha=None):
        if alpha is None:
            alpha = np.zeros(self.n_components)
        return ShapeInstance(self, alpha)

    def phi_landmarks(self, idx_list):
        """
        Φ_r extraction: (3R, K)
        """
        Phi = self.eigenfunctions.reshape(-1, 3, self.n_components)
        return Phi[idx_list].reshape(-1, self.n_components)

    def x_landmarks(self, vertices, idx_list):
        return vertices.reshape(-1, 3)[idx_list].reshape(-1)

    def landmark_indices(self):
        return np.asarray(self.landmarks["verts "], dtype=np.int32)

    def landmark_names(self):
        return [n.strip() for n in self.landmarks["names "]]

    def print_summary(self, alpha=None):
        """
        Prints statistical information regarding the model and a specific instance.
        """
        print("--- Summary ---")
        print(f"Vertices: {self.n_vertices}")
        print(f"Components: {self.n_components}")
        
        # Eigenvalue statistics
        print(f"\nEigenvalues (Total Variance: {np.sum(self.eigenvalues):.8f}):")
        print(f"  Range: [{self.eigenvalues.min():.8f}, {self.eigenvalues.max():.8f}]")
        print(f"  Mean: {np.mean(self.eigenvalues):.8f}")

        # Eigenfunction statistics (the modes of variation)
        print(f"\nEigenfunctions (Magnitude per mode):")
        mode_norms = np.linalg.norm(self.eigenfunctions, axis=0)
        print(f"  Range: [{mode_norms.min():.8f}, {mode_norms.max():.8f}]")
        print("Phi norm:", np.linalg.norm(self.eigenfunctions))

        # Mean shape position statistics
        mean_reshaped = self.mean.reshape(-1, 3)
        print(f"\nMean Shape Positions (Spatial):")
        print(f"  Centroid: {np.mean(mean_reshaped, axis=0)}")
        print(f"  Spatial Range (X, Y, Z):")
        print(f"    Min: {mean_reshaped.min(axis=0)}")
        print(f"    Max: {mean_reshaped.max(axis=0)}")

        if alpha is not None:
            print(f"\nProvided Alpha Coefficients (Shape Instance):")
            print(f"  Range: [{alpha.min():.4f}, {alpha.max():.4f}]")
            print(f"  L2 Norm: {np.linalg.norm(alpha):.4f}")

@dataclass
class ShapeInstance:
    model: StatisticalShapeModel
    alpha: np.ndarray = field(default_factory=lambda: None)

    def __post_init__(self):
        if self.alpha is None:
            self.alpha = np.zeros(self.model.n_components)

    def parameters(self):
        return self.alpha

    def reconstruct(self):
        return self.model.reconstruct(self.alpha)
    
    def solve_delta_alpha(self, Phi, x_current, x_target):
        x_current = np.asarray(x_current).reshape(-1)
        x_target = np.asarray(x_target).reshape(-1)
        return Phi.T @ (x_target - x_current)
    
    def fit(self, target_vertices):
        target_vertices = target_vertices.reshape(-1)

        current_vertices = self.reconstruct()

        delta_alpha = self.solve_delta_alpha(
            self.model.eigenfunctions,
            current_vertices,
            target_vertices
        )

        self.alpha = self.alpha + delta_alpha
        return self.alpha

    def get_landmark_data(self):
        print("self.model.landmarks: ", self.model.landmarks)
        names = [n.strip() for n in self.model.landmarks["names "]]
        indices = np.asarray(self.model.landmarks["verts "])

        vertices = self.reconstruct().reshape(-1, 3)

        landmarks = {}
        for name, idx in zip(names, indices):
            entry = {"index": int(idx)}

            entry["position"] = np.asarray(
                vertices[idx],
                dtype=np.float32
            )

            landmarks[name] = entry

        return landmarks
    
    def get_landmark_indices(self):
        return self.model.landmark_indices()

    def get_landmark_names(self):
        return self.model.landmark_names()

    def get_current_landmarks(self):
        x = self.reconstruct()
        idx = self.get_landmark_indices()
        return self.model.x_landmarks(x, idx), idx

    def extract_target_landmarks(self, target_landmarks: dict):
        names = self.get_landmark_names()
        idx_all = self.get_landmark_indices()

        x_list = []
        idx_used = []

        for name, idx in zip(names, idx_all):
            if name not in target_landmarks:
                continue

            x_list.append(np.asarray(target_landmarks[name], dtype=np.float32))
            idx_used.append(idx)

        x_target = np.asarray(x_list).reshape(-1)
        idx_used = np.asarray(idx_used, dtype=np.int32)

        return x_target, idx_used

    def fit_landmarks(self, target_landmarks, learning_rate=0.01): 
        x_current_full = self.reconstruct()
        x_target, idx = self.extract_target_landmarks(target_landmarks)

        x_current = self.model.x_landmarks(x_current_full, idx)
        Phi_r = self.model.phi_landmarks(idx)

        delta_alpha = self.solve_delta_alpha(Phi_r, x_current, x_target)
        self.alpha += learning_rate * delta_alpha 
        
        return self.alpha
    
    def fit_to_matching(self, target_vertices, matches, learning_rate=0.1, gamma=10000, chi_clamp = None):
        """
        Fits the SSM instance to a target surface using sparse matches and 
        MAP-regularized optimization.
        
        gamma: Regularization strength. 
               - Increase if the shape is distorting/folding.
               - Decrease if the shape is too rigid to reach the target.
        """
        matches = np.array(matches)
        model_idx = matches[:, 0].astype(int)
        target_idx = matches[:, 1].astype(int)
        
        # Extract current and target point subsets
        current_full = self.reconstruct().reshape(-1, 3)
        x_current = current_full[model_idx].reshape(-1)
        x_target = target_vertices[target_idx].reshape(-1)
        
        # Extract eigenfunction rows (Phi_I)
        Phi = self.model.eigenfunctions.reshape(-1, 3, self.model.n_components)
        Phi_I = Phi[model_idx].reshape(-1, self.model.n_components)
        
        # Solve for optimal delta_alpha using MAP regularization
        # We solve: (Phi_I.T @ Phi_I + gamma * I) * delta_alpha = Phi_I.T @ residual
        delta_alpha = self.solve_delta_alpha_MAP(Phi_I, x_current, x_target, gamma=gamma)
        
        # Apply update with learning rate
        self.alpha += learning_rate * delta_alpha

        if chi_clamp != None:
            self.alpha = self.clamp_coefficients(chi_clamp)

        return self.alpha

    def solve_delta_alpha_MAP(self, Phi_I, x_current, x_target, gamma=1.0):
        residual = (x_target - x_current)
        rhs = Phi_I.T @ residual

        lhs = Phi_I.T @ Phi_I + gamma * np.eye(Phi_I.shape[1])
        rhs = Phi_I.T @ residual
        
        try:
            coeffs = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular (e.g. not enough matches)
            coeffs = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        return coeffs
    
    def clamp_coefficients(self, p=0.99):
        """
        Constrains alpha so that sum( (alpha_i / sqrt(lambda_i))^2 ) <= chi2_threshold
        """
        # Calculate normalized coefficients z_i = alpha_i / sqrt(lambda_i)
        # Note: eigenvalues are likely variances (sigma_i), 
        # so sqrt(lambda_i) is the standard deviation.
        std_devs = np.sqrt(self.model.eigenvalues)
        z = self.alpha / std_devs
        
        # Calculate the current Mahalanobis distance squared
        dist_sq = np.sum(z**2)
        
        # Define the threshold beta^2 from the Chi-square distribution
        M = self.model.n_components
        beta_sq = 10000**2 # * chi2.ppf(p, df=M)
        print(" M: ", M)
        print(" >>>>>>>>>>>>>>>> beta_sq: ", beta_sq)
        print(" >>>>>>>>>>>>>>>> beta: ", beta_sq**0.5)
        # If outside the hyper-ellipsoid, scale alpha back to the boundary
        if dist_sq > beta_sq:
            scale = np.sqrt(beta_sq / dist_sq)
            self.alpha = self.alpha * scale
            
        return self.alpha
    

    def regional_mask(self, lmk_name: str, radius: float):
        """
        Generates a binary mask of vertices within a specified radius 
        of a given landmark on the current shape instance.
        """
        # 1. Get landmarks to find the index of the requested landmark
        landmarks = self.get_landmark_data()
        
        if lmk_name not in landmarks:
            raise ValueError(f"Landmark '{lmk_name}' not found. Available: {list(landmarks.keys())}")
        
        lmk_idx = landmarks[lmk_name]['index']
        
        # 2. Get current full mesh vertices
        current_vertices = self.reconstruct().reshape(-1, 3)
        anchor_point = current_vertices[lmk_idx]
        
        # 3. Compute Euclidean distances from the anchor point to all vertices
        dists = np.linalg.norm(current_vertices - anchor_point, axis=1)
        
        # 4. Create binary mask
        mask = (dists <= radius).astype(np.float32)
        
        return mask
    
    def get_selection_matrix(self,  lmk_name: str, radius: float):
        """
        Converts a boolean/binary mask of length N into a Sparse Matrix (M, N),
        where M is the number of vertices where mask == 1.
        """
        mask = self.regional_mask(lmk_name, radius)
        # Get indices of the vertices that are 'selected'
        selected_indices = np.where(mask > 0)[0]
        M = len(selected_indices)
        N = len(mask)
        
        # Create the sparse matrix (M, N)
        # Each row 'i' has a '1' at column 'selected_indices[i]'
        rows = np.arange(M)
        cols = selected_indices
        data = np.ones(M)
        
        return sparse.csr_matrix((data, (rows, cols)), shape=(M, N))

    def region_vertices(self, lmk_name: str, radius: float ):
        mask = self.regional_mask(lmk_name, radius)
        verts = self.reconstruct().reshape(-1, 3)[mask.astype(bool)]
        return verts