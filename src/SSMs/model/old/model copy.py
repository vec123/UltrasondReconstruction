import numpy as np
from dataclasses import dataclass, field
from scipy.stats import chi2

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
    
    def fit_to_matching(self, target_vertices, matches, learning_rate=0.1):
            """
            matches: ndarray of shape (N, 2), where:
                    matches[:, 0] = index in model (source)
                    matches[:, 1] = index in target_surface
            """
            matches = np.array(matches)
            
            #  Extract the indices and targets
            model_idx = matches[:, 0].astype(int)
            target_idx = matches[:, 1].astype(int)
            
            # Get current model vertices at the matched indices
            current_full = self.reconstruct().reshape(-1, 3)
            x_current = current_full[model_idx].reshape(-1)
            
            # Get target vertices at the matched indices
            # target_vertices is (M, 3)
            x_target = target_vertices[target_idx].reshape(-1)
            
            # Extract corresponding eigenfunction rows (Phi_I)
            # Phi is (N, 3, K),  reshape to (3N, K)
            # requires dhape (3 * len(model_idx), K)
            Phi = self.model.eigenfunctions.reshape(-1, 3, self.model.n_components)
            Phi_I = Phi[model_idx].reshape(-1, self.model.n_components)
            
            # Solve for delta_alpha
            delta_alpha = self.solve_delta_alpha(Phi_I, x_current, x_target)
            
            # Apply update
            self.alpha += learning_rate * delta_alpha
            self.apply_regularization_clamp(p=0.95)

            residual = (x_target - x_current)
            print(f"Max residual: {np.max(np.abs(residual))}")
            projection = Phi_I.T @ residual
            print(f"Max projection: {np.max(np.abs(projection))}")
            print("new coefficients: ", self.alpha)    
            return self.alpha
    
    def apply_regularization_clamp(self, p=0.95):
        """
        Constrains alpha so that sum( (alpha_i / sqrt(lambda_i))^2 ) <= chi2_threshold
        """
        # Calculate normalized coefficients z_i = alpha_i / sqrt(lambda_i)
        # Note: Your eigenvalues are likely variances (sigma_i), 
        # so sqrt(lambda_i) is the standard deviation.
        std_devs = np.sqrt(self.model.eigenvalues)
        z = self.alpha / std_devs
        
        # Calculate the current Mahalanobis distance squared
        dist_sq = np.sum(z**2)
        
        # Define the threshold beta^2 from the Chi-square distribution
        M = self.model.n_components
        beta_sq = chi2.ppf(p, df=M)
        
        # If outside the hyper-ellipsoid, scale alpha back to the boundary
        if dist_sq > beta_sq:
            scale = np.sqrt(beta_sq / dist_sq)
            self.alpha = self.alpha * scale
            
        return self.alpha
    
    def solve_delta_alpha_regularized(self, Phi, x_current, x_target, gamma=0.001):
        """
        Phi: (3R, K)
        x_target - x_current: (3R,)
        gamma: Regularization weight (higher = more rigid/plausible, lower = fits noise more)
        """
        # Calculate the residual error
        residual = (x_target - x_current)
        
        # Regularized system: (Phi.T @ Phi + gamma * I) * delta_alpha = Phi.T * residual
        # Note: If components are normalized by eigenvalues (sigma_i),
        # use (Phi.T @ Phi + gamma * diag(1/sigma_i))
        
        # Simple Tikhonov implementation:
        I = np.eye(Phi.shape[1])
        # We scale gamma by the variance of the components
        reg_matrix = gamma * np.diag(1.0 / (self.model.eigenvalues + 1e-8))
        
        # Solving the linear system is more stable than matrix inversion
        lhs = Phi.T @ Phi + reg_matrix
        rhs = Phi.T @ residual
        
        return np.linalg.solve(lhs, rhs)