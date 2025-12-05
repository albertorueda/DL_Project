from geopy.distance import geodesic
import torch
from math import radians, sin, cos, sqrt, atan2

def decode_predictions(predictions, lat_min, lat_max, lon_min, lon_max):
    """Decode normalized predictions back to original latitude and longitude."""
    decoded = predictions.clone()
    decoded[..., 0] = decoded[..., 0] * (lat_max - lat_min) + lat_min  # Latitude
    decoded[..., 1] = decoded[..., 1] * (lon_max - lon_min) + lon_min  # Longitude
    return decoded

def haversine_dist(coords1, coords2):
    """
    Calculate the Haversine distance between two (lat, lon) coordinates in kilometers.
    Args:
        coords1: Tuple of (latitude, longitude) for the first point.
        coords2: Tuple of (latitude, longitude) for the second point.
    Returns:
        Distance in kilometers.
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
    return distance/len(coords1)


def ADE(coords1, coords2):
    """
    Calculate the Average Displacement Error (ADE) between two sets of (lat, lon) coordinates.
    Args:
        coords1: Tensor of shape (..., 2) for the first set of points.
        coords2: Tensor of shape (..., 2) for the second set of points.
    Returns:
        ADE in km.
    """
    distance = 0.0
    batch_size, seq_len, _ = coords1.shape
    for i in range(seq_len):
        distance += haversine_dist(coords1[:, i, :], coords2[:, i, :])
        
    return distance / seq_len
    


def FDE(coords1, coords2):
    """
    Calculate the Final Displacement Error (FDE) between two sets of (lat, lon) coordinates.
    Args:
        coords1: Tensor of shape (..., 2) for the first set of points.
        coords2: Tensor of shape (..., 2) for the second set of points.
    Returns:
        FDE in km.
    """
    final_pred = coords1[:, -1]
    final_gt   = coords2[:, -1]
    
    distance = 0.0
    for p, g in zip(final_pred, final_gt):
        distance += haversine_dist(p.unsqueeze(0), g.unsqueeze(0))
    return distance / final_pred.shape[0]



def RMSE(coords1, coords2):
    """
    Calculate the Root Mean Square Error (RMSE) between two sets of (lat, lon) coordinates.
    Args:
        coords1: Tensor of shape (..., 2) for the first set of points.
        coords2: Tensor of shape (..., 2) for the second set of points.
    Returns:
        RMSE in km.
    """
    batch_size, seq_len, _ = coords1.shape
    mse = 0.0
    for i in range(seq_len):
        mse += haversine_dist(coords1[:, i, :], coords2[:, i, :]) ** 2
    mse /= seq_len
    return sqrt(mse)