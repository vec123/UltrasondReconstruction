import numpy as np
from scipy.stats import chi2


def mahalanobis_distance(alpha, eigenvalues):
    return np.sum(alpha**2 / eigenvalues)


def chi_square_threshold(probability, n_components):
    return chi2.ppf(probability, n_components)


def is_plausible(alpha,
                 eigenvalues,
                 probability=0.95):

    d2 = mahalanobis_distance(alpha, eigenvalues)

    threshold = chi_square_threshold(
        probability,
        len(alpha)
    )

    return d2 <= threshold


def project_to_plausible_space(alpha,
                               eigenvalues,
                               probability=0.95):

    d2 = mahalanobis_distance(alpha, eigenvalues)

    threshold = chi_square_threshold(
        probability,
        len(alpha)
    )

    if d2 <= threshold:
        return alpha.copy()

    scale = np.sqrt(threshold / d2)

    return alpha * scale