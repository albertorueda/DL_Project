"""
metrics.py

This module contains evaluation metrics for trajectory prediction models.
It supports calculating distance-based errors between predicted and ground truth sequences,
including Haversine distance, Average Displacement Error (ADE), Final Displacement Error (FDE),
and Root Mean Square Error (RMSE).

All coordinates are expected in latitude and longitude format (degrees), and
all outputs are in kilometers.
"""

from math import radians, sin, cos, sqrt, atan2

def decode_predictions(predictions, lat_min, lat_max, lon_min, lon_max):
    """
    Decode normalized predictions back to original latitude and longitude.

    Args:
        predictions (Tensor): Normalized tensor of shape (..., 2).
        lat_min (float): Minimum latitude used for normalization.
        lat_max (float): Maximum latitude used for normalization.
        lon_min (float): Minimum longitude used for normalization.
        lon_max (float): Maximum longitude used for normalization.

    Returns:
        Tensor: Decoded predictions in original lat/lon coordinates.
    """
    decoded = predictions.clone()
    decoded[..., 0] = decoded[..., 0] * (lat_max - lat_min) + lat_min
    decoded[..., 1] = decoded[..., 1] * (lon_max - lon_min) + lon_min
    return decoded

def haversine_dist(coords1, coords2):
    """
    Compute average Haversine distance (in km) between batches of coordinate pairs.

    Args:
        coords1 (Iterable of tuples): First set of (lat, lon) coordinates.
        coords2 (Iterable of tuples): Second set of (lat, lon) coordinates.

    Returns:
        float: Average Haversine distance across all pairs (in km).
    """
    R = 6371.0  # Earth radius in kilometers
    distance = 0.0

    for coord1, coord2 in zip(coords1, coords2):
        lat1, lon1 = coord1
        lat2, lon2 = coord2

        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)

        a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance += R * c
    return distance / len(coords1)

def ADE(coords1, coords2):
    """
    Compute Average Displacement Error (ADE) between two sequences of coordinates.

    Args:
        coords1 (Tensor): Predicted coordinates, shape (batch_size, seq_len, 2).
        coords2 (Tensor): Ground truth coordinates, shape (batch_size, seq_len, 2).

    Returns:
        float: ADE (average error per time step) in km.
    """
    distance = 0.0
    batch_size, seq_len, _ = coords1.shape
    for i in range(seq_len):
        # Compute average Haversine distance for each time step across the batch
        distance += haversine_dist(coords1[:, i, :], coords2[:, i, :])
    return distance / seq_len

def FDE(coords1, coords2):
    """
    Compute Final Displacement Error (FDE) between last predicted and true point.

    Args:
        coords1 (Tensor): Predicted coordinates, shape (batch_size, seq_len, 2).
        coords2 (Tensor): Ground truth coordinates, shape (batch_size, seq_len, 2).

    Returns:
        float: FDE (error at last time step) in km.
    """
    final_pred = coords1[:, -1]
    final_gt   = coords2[:, -1]

    distance = 0.0
    # Calculate Haversine distance for each pair in the batch
    for pred_point, gt_point in zip(final_pred, final_gt):
        distance += haversine_dist(pred_point.unsqueeze(0), gt_point.unsqueeze(0))
    return distance / final_pred.shape[0]

def RMSE(coords1, coords2):
    """
    Compute Root Mean Square Error (RMSE) between predicted and true coordinates.

    Args:
        coords1 (Tensor): Predicted coordinates, shape (batch_size, seq_len, 2).
        coords2 (Tensor): Ground truth coordinates, shape (batch_size, seq_len, 2).

    Returns:
        float: RMSE (root of average squared Haversine error) in km.
    """
    batch_size, seq_len, _ = coords1.shape
    mse = 0.0
    for i in range(seq_len):
        # Sum squared Haversine distances for each time step
        mse += haversine_dist(coords1[:, i, :], coords2[:, i, :]) ** 2
    mse /= seq_len
    return sqrt(mse)