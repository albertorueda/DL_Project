"""
Loss functions for trajectory prediction models.

Includes:
- HaversineLoss: great-circle distance in kilometers between predicted and ground-truth coordinates (normalized inputs).
- HybridTrajectoryLoss: combined loss with position, direction, and speed components (radian input).
"""

### ================================================================
### --- IMPORTS ---
### ================================================================
import torch
import torch.nn.functional as F

### ================================================================
### --- HAVERSINE LOSS ---
### ================================================================
class HaversineLoss(torch.nn.Module):
    """
    Loss function computing the great-circle distance between predicted and true coordinates.

    Designed for inputs normalized to [0, 1] representing latitude and longitude.

    Input shape:
        y_pred, y_true: (batch, seq, 2) normalized lat/lon coordinates.
    """
    def __init__(self, 
                 lat_min, lat_max,
                 lon_min, lon_max,
                 radius_earth_km=6371.0):
        """
        Initializes normalization boundaries and Earth radius.

        Args:
            lat_min (float): Minimum latitude of the dataset (for unnormalization).
            lat_max (float): Maximum latitude of the dataset (for unnormalization).
            lon_min (float): Minimum longitude of the dataset (for unnormalization).
            lon_max (float): Maximum longitude of the dataset (for unnormalization).
            radius_earth_km (float, optional): Radius of the Earth in km. Defaults to 6371.0.
        """
        super().__init__()
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.radius_earth_km = radius_earth_km

    def forward(self, y_pred, y_true):
        """
        Computes mean Haversine distance (in kilometers) between predicted and true coordinates.

        Args:
            y_pred (Tensor): Predicted normalized lat/lon coordinates. Shape: (batch, seq, 2)
            y_true (Tensor): True normalized lat/lon coordinates. Shape: (batch, seq, 2)

        Returns:
            Tensor: Mean Haversine distance in kilometers.
        """

        # --- DESNORMALIZAR ---
        lat_true = y_true[..., 0] * (self.lat_max - self.lat_min) + self.lat_min
        lon_true = y_true[..., 1] * (self.lon_max - self.lon_min) + self.lon_min

        lat_pred = y_pred[..., 0] * (self.lat_max - self.lat_min) + self.lat_min
        lon_pred = y_pred[..., 1] * (self.lon_max - self.lon_min) + self.lon_min

        # --- PASAR A RADIANES ---
        lat1 = torch.deg2rad(lat_true)
        lon1 = torch.deg2rad(lon_true)
        lat2 = torch.deg2rad(lat_pred)
        lon2 = torch.deg2rad(lon_pred)

        # --- HAVERSINE ---
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (torch.sin(dlat / 2) ** 2 +
             torch.cos(lat1) * torch.cos(lat2) *
             torch.sin(dlon / 2) ** 2)

        a = torch.clamp(a, 0.0, 1.0)
        eps = 1e-9

        c = 2 * torch.atan2(torch.sqrt(a + eps), torch.sqrt(1 - a + eps))
        distance = self.radius_earth_km * c  # => km

        return torch.mean(distance)
