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
    
    def fit_to_matching(self, target_vertices, matches, learning_rate=0.1, gamma=100):
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
        
        # 1. Extract current and target point subsets
        current_full = self.reconstruct().reshape(-1, 3)
        x_current = current_full[model_idx].reshape(-1)
        x_target = target_vertices[target_idx].reshape(-1)
        
        # 2. Extract eigenfunction rows (Phi_I)
        Phi = self.model.eigenfunctions.reshape(-1, 3, self.model.n_components)
        Phi_I = Phi[model_idx].reshape(-1, self.model.n_components)
        
        # 3. Solve for optimal delta_alpha using MAP regularization
        # We solve: (Phi_I.T @ Phi_I + gamma * I) * delta_alpha = Phi_I.T @ residual
        delta_alpha = self.solve_delta_alpha_map(Phi_I, x_current, x_target, gamma=gamma)
        
        # 4. Apply update with learning rate
        self.alpha += learning_rate * delta_alpha
        print("new coefficients: ", self.alpha)
        return self.alpha

    def solve_delta_alpha_map(self, Phi_I, x_current, x_target, gamma=1.0):
        residual = (x_target - x_current)
        
        # DEBUG PRINTS
        print(f"Residual mean: {np.mean(residual):.4f}, std: {np.std(residual):.4f}")
        print(f"Phi_I mean: {np.mean(Phi_I):.4f}, std: {np.std(Phi_I):.4f}")
        
        rhs = Phi_I.T @ residual
        print(f"RHS (Projection) norm: {np.linalg.norm(rhs):.6f}")
        
        if np.linalg.norm(rhs) < 1e-10:
            print("CRITICAL: Projection is zero! Check if x_target/x_current are properly centered.")

        # The regularized system: (Phi^T * Phi + gamma * I) * delta = Phi^T * residual
        # Using a Identity-based regularization is generally more stable for 
        # arbitrary sparse matching than eigenvalue-weighted regularization.
        
        lhs = Phi_I.T @ Phi_I + gamma * np.eye(Phi_I.shape[1])
        rhs = Phi_I.T @ residual
        
        # Using np.linalg.solve is numerically superior to explicit inversion
        try:
            return np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular (e.g. not enough matches)
            return np.linalg.lstsq(lhs, rhs, rcond=None)[0]