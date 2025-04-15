import numpy as np
import torch


def expectile_regression(pred, target, expectile):
    diff = target - pred
    return torch.where(diff > 0, expectile, 1-expectile) * (diff**2)

def discounted_cum_sum(seq, discount):
    seq = seq.copy()
    for t in reversed(range(len(seq)-1)):
        seq[t] += discount * seq[t+1]
    return seq

def biased_bce_with_logits(adv1, adv2, label, bias=1.0):
    # Apply the log-sum-exp trick.
    # y = 1 if we prefer x2 to x1
    # We need to implement the numerical stability trick.

    logit21 = adv2 - bias * adv1
    logit12 = adv1 - bias * adv2
    max21 = torch.clamp(-logit21, min=0, max=None)
    max12 = torch.clamp(-logit12, min=0, max=None)
    nlp21 = torch.log(torch.exp(-max21) + torch.exp(-logit21 - max21)) + max21
    nlp12 = torch.log(torch.exp(-max12) + torch.exp(-logit12 - max12)) + max12
    loss = label * nlp21 + (1 - label) * nlp12
    return loss

def count_inversions(A, B=None) -> int:
    """
    Count inversions between two lists A and B.
    An inversion is a pair (i,j) where A[i] < A[j] but B[i] > B[j].

    If B is not provided, it counts inversions in A only.
    """
    if B is None:
        B = list(range(len(A)))
    n = len(A)
    # Create pairs of (A[i], B[i]) and sort by A values
    pairs = sorted([(A[i], B[i]) for i in range(n)])
    
    # Extract the B values in order of sorted A values
    sorted_B = [pair[1] for pair in pairs]
    
    # Count inversions in sorted_B using merge sort approach
    def merge_and_count(arr):
        if len(arr) <= 1:
            return arr, 0
        
        mid = len(arr) // 2
        left, inv_left = merge_and_count(arr[:mid])
        right, inv_right = merge_and_count(arr[mid:])
        
        merged = []
        i = j = 0
        inv_count = inv_left + inv_right
        
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                # This is an inversion
                merged.append(right[j])
                j += 1
                inv_count += len(left) - i
        
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, inv_count
    
    _, inversions = merge_and_count(sorted_B)
    return inversions

def order_consistency_rate(A, B):
    A = np.argsort(A)
    B = np.argsort(B)
    return 1 - count_inversions(A, B) / (len(A) * (len(A) - 1) / 2)

def top_k_overlap_rate(pred_list, gt_list, percent=0.1):
    """
    Calculate the proportion of indices in the top k% of pred_list that also appear in the top k% of gt_list.
    
    Args:
        pred_list: List of predicted values
        gt_list: List of ground truth values
        percent: Percentage to consider for top values as a decimal (default is 0.1, meaning 10%)
    
    Returns:
        float: The ratio of overlap between the two top-k% index sets
    """
    if len(pred_list) != len(gt_list) or len(pred_list) == 0:
        raise ValueError("Input lists must have the same non-zero length")
    
    # Calculate the number of elements to select for top k%
    k = max(1, int(len(pred_list) * percent))
    
    # Get indices of top k% elements
    pred_top_indices = set(sorted(range(len(pred_list)), key=lambda i: pred_list[i], reverse=True)[:k])
    gt_top_indices = set(sorted(range(len(gt_list)), key=lambda i: gt_list[i], reverse=True)[:k])
    
    # Calculate overlap
    overlap = pred_top_indices.intersection(gt_top_indices)
    
    # Return ratio of overlap to the size of top k%
    return len(overlap) / k

def spearman_rank_correlation(x, y):
    """
    Calculate Spearman's rank correlation coefficient between two lists.
    
    Args:
        x: First list of values
        y: Second list of values
    
    Returns:
        float: Spearman's rank correlation coefficient between -1 and 1
    """
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("Input lists must have the same non-zero length")

    n = len(x)
    
    # Convert values to ranks
    x_array = np.array(x)
    y_array = np.array(y)
    
    # argsort of argsort gives the ranks (0-based)
    # Adding 1 to make it 1-based ranking
    x_ranks = np.argsort(np.argsort(x_array)) + 1
    y_ranks = np.argsort(np.argsort(y_array)) + 1
    
    # Calculate di^2
    d_squared = np.sum((x_ranks - y_ranks) ** 2)
    
    # Apply Spearman's formula: rho = 1 - (6 * sum(d²) / (n³ - n))
    rho = 1 - (6 * d_squared) / (n**3 - n)
    
    return rho