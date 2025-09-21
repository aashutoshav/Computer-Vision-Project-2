import math

import numpy as np
import cv2


def calculate_num_ransac_iterations(
    prob_success: float, sample_size: int, ind_prob_correct: int
) -> int:
    """
    Calculate the number of RANSAC iterations needed for a given guarantee of success.

    Args:
    -   prob_success: float representing the desired guarantee of success
    -   sample_size: int the number of samples included in each RANSAC iteration
    -   ind_prob_success: float representing the probability that each element in a sample is correct

    Returns:
    -   num_samples: int the number of RANSAC iterations needed

    """
    num_samples = None
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################
    
    prob_all_correct = ind_prob_correct ** sample_size
    prob_failure = 1 - prob_all_correct
    
    if prob_failure == 0:
        return 1
    else:
        num_samples = math.log(1 - prob_success) / math.log(prob_failure)
        return int(math.ceil(num_samples))

    # raise NotImplementedError(
    #     "`calculate_num_ransac_iterations` function in "
    #     + "`ransac.py` needs to be implemented"
    # )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return int(num_samples)

def ransac_homography(
    points_a: np.ndarray, points_b: np.ndarray
):
    """
    Uses the RANSAC algorithm to robustly estimate a homography matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) of points from image A.
    -   points_b: A numpy array of shape (N, 2) of corresponding points from image B.

    Returns:
    -   best_H: The best homography matrix of shape (3, 3).
    -   inliers_a: The subset of points_a that are inliers (M, 2).
    -   inliers_b: The subset of points_b that are inliers (M, 2).
    """
    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    #                                                                         #
    # HINT: You are allowed to use the `cv2.findHomography` function to       #
    # compute the homography from a sample of points. To compute a direct     #
    # solution without OpenCV's built-in RANSAC, use it like this:            #
    #   H, _ = cv2.findHomography(sample_a, sample_b, 0)                      #
    # The `0` flag ensures it computes a direct least-squares solution.       #
    ###########################################################################
    
    best_H = None
    max_inliers_count = 0
    inlier_threshold = 5.0
    sample_size = 4
    prob_success = 0.99
    
    num_points = points_a.shape[0]
    
    points_a_float = points_a.astype(np.float32)
    points_b_float = points_b.astype(np.float32)
    
    initial_inlier_prob = 0.1  
    num_iterations = calculate_num_ransac_iterations(prob_success, sample_size, initial_inlier_prob)
    
    best_inliers_mask = None
    
    for i in range(num_iterations):
        indices = np.random.choice(num_points, sample_size, replace=False)
        sample_a = points_a_float[indices]
        sample_b = points_b_float[indices]
        
        H, _ = cv2.findHomography(sample_a, sample_b, 0)
        
        if H is None:
            continue
            
        transformed_points_a = cv2.perspectiveTransform(points_a_float.reshape(-1, 1, 2), H).reshape(-1, 2)
        
        errors = np.sum((transformed_points_a - points_b_float)**2, axis=1)
        
        current_inliers_mask = errors < inlier_threshold**2
        current_inliers_count = np.sum(current_inliers_mask)
        
        if current_inliers_count > max_inliers_count:
            max_inliers_count = current_inliers_count
            best_inliers_mask = current_inliers_mask
            
    if max_inliers_count > sample_size and best_inliers_mask is not None:
        inliers_a_best_model = points_a_float[best_inliers_mask]
        inliers_b_best_model = points_b_float[best_inliers_mask]
        
        best_H, _ = cv2.findHomography(inliers_a_best_model, inliers_b_best_model, 0)
        
        if best_H is not None:
            final_transformed_points_a = cv2.perspectiveTransform(points_a_float.reshape(-1, 1, 2), best_H).reshape(-1, 2)
            final_errors = np.sum((final_transformed_points_a - points_b_float)**2, axis=1)
            final_inliers_mask = final_errors < inlier_threshold**2
            
            inliers_a = points_a[final_inliers_mask]
            inliers_b = points_b[final_inliers_mask]
        else:
            inliers_a = np.empty((0, 2))
            inliers_b = np.empty((0, 2))
            best_H = None
    else:
        inliers_a = np.empty((0, 2))
        inliers_b = np.empty((0, 2))
        best_H = None

    # raise NotImplementedError(
    #     "`ransac_homography` function in "
    #     + "`part6_ransac.py` needs to be implemented"
    # )

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return best_H, inliers_a, inliers_b
