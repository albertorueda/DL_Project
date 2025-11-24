import torch

class HaversineLoss(torch.nn.Module):
    def __init__(self, radius_earth_km=6371.0):
        super().__init__()
        self.radius_earth_km = radius_earth_km

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (batch, seq, 2) lat/lon in radians
            y_true: (batch, seq, 2) lat/lon in radians
        """
        lat1 = y_true[..., 0]
        lon1 = y_true[..., 1]
        lat2 = y_pred[..., 0]
        lon2 = y_pred[..., 1]

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            torch.sin(dlat / 2) ** 2 +
            torch.cos(lat1) * torch.cos(lat2) *
            torch.sin(dlon / 2) ** 2
        )

        # Stabilize
        a = torch.clamp(a, 0.0, 1.0)
        eps = 1e-9

        c = 2 * torch.atan2(torch.sqrt(a + eps), torch.sqrt(1 - a + eps))

        distance = self.radius_earth_km * c
        return torch.mean(distance)
