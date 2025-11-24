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





import torch.nn.functional as F

class HybridTrajectoryLoss(torch.nn.Module):
    def __init__(self, w_pos=1.0, w_ang=0.2, w_spd=0.1, radius_km=6371.0, max_pos_error_km=10.0):
        super().__init__()
        self.w_pos = w_pos
        self.w_ang = w_ang
        self.w_spd = w_spd
        self.radius_km = radius_km
        self.max_pos_error_km = max_pos_error_km

    def haversine_dist(self, p1, p2):
        """
        Calculates Great Circle distance (km) between points.
        Input: (Batch, ..., 2) where 0=Lat, 1=Lon in RADIANS.
        """
        lat1, lon1 = p1[..., 0], p1[..., 1]
        lat2, lon2 = p2[..., 0], p2[..., 1]

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (torch.sin(dlat / 2) ** 2 +
             torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2)
        
        # Stability clamp
        a = torch.clamp(a, 0.0, 1.0)
        
        c = 2 * torch.atan2(torch.sqrt(a + 1e-9), torch.sqrt(1 - a + 1e-9))
        return self.radius_km * c

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (Batch, Seq_Len, 2) -> Lat/Lon in RADIANS
            y_true: (Batch, Seq_Len, 2) -> Lat/Lon in RADIANS
        """

        # --- 1. Position Loss (Haversine -> Tanh) ---
        dist_km = self.haversine_dist(y_pred, y_true)
        
        # Normalize: 0 to 1
        # If the error is > max_pos_error_km, the loss saturates to 1.0
        loss_pos = torch.tanh(dist_km.mean() / self.max_pos_error_km)


        # --- 2. Angle Loss (Euclidean Vector Approximation) ---
        vec_pred = y_pred[:, -1, :] - y_pred[:, 0, :]
        vec_true = y_true[:, -1, :] - y_true[:, 0, :]
        
        # Calculate Cosine Similarity
        cos_sim = F.cosine_similarity(vec_pred, vec_true, dim=1, eps=1e-8)
        
        # Normalize to [0, 1]
        # Range of cosine is [-1, 1], so (1 - cos) is [0, 2]. Divide by 2.
        loss_ang = (1.0 - cos_sim).mean() / 2.0


        # --- 3. Speed Loss (Euclidean Diff -> Tanh) ---
        vel_pred = torch.diff(y_pred, dim=1)
        vel_true = torch.diff(y_true, dim=1)
        
        # Magnitude of the difference vector
        speed_pred = torch.linalg.norm(vel_pred, dim=2)
        speed_true = torch.linalg.norm(vel_true, dim=2)
        
        # Squared Error
        speed_sq_err = (speed_pred - speed_true) ** 2
        
        # Normalize to [0, 1]
        loss_spd = torch.tanh(speed_sq_err.mean())


        # --- Combine ---
        total_loss = (self.w_pos * loss_pos) + \
                     (self.w_ang * loss_ang) + \
                     (self.w_spd * loss_spd)

        return total_loss, {
            "loss_pos": loss_pos.item(),
            "loss_ang": loss_ang.item(),
            "loss_spd": loss_spd.item()
        }
    