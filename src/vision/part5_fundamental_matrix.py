"""Fundamental matrix utilities."""

import numpy as np


def normalize_points(points: np.ndarray):
    """
    Perform coordinate normalization through linear transformations.
    Args:
        points: A numpy array of shape (N, 2) representing the 2D points in
            the image

    Returns:
        points_normalized: A numpy array of shape (N, 2) representing the
            normalized 2D points in the image
        T: transformation matrix representing the product of the scale and
            offset matrices
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    centroid = np.mean(points, axis=0)
    u_c, v_c = centroid

    std_devs = np.std(points, axis=0)
    s_u, s_v = std_devs[0], std_devs[1]

    s_u = 1.0 / s_u if s_u > 0 else 1.0
    s_v = 1.0 / s_v if s_v > 0 else 1.0

    T = np.array([
        [s_u, 0,   -s_u * u_c],
        [0,   s_v, -s_v * v_c],
        [0,   0,   1]
    ])

    points_homo = np.hstack((points, np.ones((points.shape[0], 1))))
    points_normalized_homo = (T @ points_homo.T).T
    points_normalized = points_normalized_homo[:, :2]

    # raise NotImplementedError(
    #     "`normalize_points` function in "
    #     + "`fundamental_matrix.py` needs to be implemented"
    # )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return points_normalized, T


def unnormalize_F(F_norm: np.ndarray, T_a: np.ndarray, T_b: np.ndarray) -> np.ndarray:
    """
    Adjusts F to account for normalized coordinates by using the transformation
    matrices.

    Args:
        F_norm: A numpy array of shape (3, 3) representing the normalized
            fundamental matrix
        T_a: Transformation matrix for image A
        T_B: Transformation matrix for image B

    Returns:
        F_orig: A numpy array of shape (3, 3) representing the original
            fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    F_orig = T_b.T @ F_norm @ T_a

    # raise NotImplementedError(
    #     "`unnormalize_F` function in "
    #     + "`fundamental_matrix.py` needs to be implemented"
    # )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F_orig


def make_singular(F_norm: np.array) -> np.ndarray:
    """
    Force F to be singular by zeroing the smallest of its singular values.
    This is done because F is not supposed to be full rank, but an inaccurate
    solution may end up as rank 3.

    Args:
    - F_norm: A numpy array of shape (3,3) representing the normalized fundamental matrix.

    Returns:
    - F_norm_s: A numpy array of shape (3, 3) representing the normalized fundamental matrix
                with only rank 2.
    """
    U, D, Vt = np.linalg.svd(F_norm)
    D[-1] = 0
    F_norm_s = np.dot(np.dot(U, np.diag(D)), Vt)

    return F_norm_s


def estimate_fundamental_matrix(
    points_a: np.ndarray, points_b: np.ndarray
) -> np.ndarray:
    """
    Calculates the fundamental matrix. You may use the normalize_points() and
    unnormalize_F() functions here. Equation (9) in the documentation indicates
    one equation of a linear system in which you'll want to solve for f_{i, j}.

    Since the matrix is defined up to a scale, many solutions exist. To constrain
    your solution, use can either use SVD and use the last Vt vector as your
    solution, or you can fix f_{3, 3} to be 1 and solve with least squares.

    Be sure to reduce the rank of your estimate - it should be rank 2. The
    make_singular() function can do this for you.

    Args:
        points_a: A numpy array of shape (N, 2) representing the 2D points in
            image A
        points_b: A numpy array of shape (N, 2) representing the 2D points in
            image B

    Returns:
        F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    points_a_norm, T_a = normalize_points(points_a)
    points_b_norm, T_b = normalize_points(points_b)

    u_a = points_a_norm[:, 0]
    v_a = points_a_norm[:, 1]
    u_b = points_b_norm[:, 0]
    v_b = points_b_norm[:, 1]

    A = np.vstack([
        u_b * u_a, u_b * v_a, u_b,
        v_b * u_a, v_b * v_a, v_b,
        u_a, v_a, np.ones_like(u_a)
    ]).T

    _, _, Vh = np.linalg.svd(A)
    f = Vh[-1, :]

    F_norm = f.reshape(3, 3)

    F_norm_singular = make_singular(F_norm)
    
    F = unnormalize_F(F_norm_singular, T_a, T_b)

    # raise NotImplementedError(
    #     "`estimate_fundamental_matrix` function in "
    #     + "`fundamental_matrix.py` needs to be implemented"
    # )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return F
