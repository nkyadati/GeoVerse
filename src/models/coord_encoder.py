import math, torch, torch.nn as nn

def sincos_posenc_latlon(lat, lon, k: int = 16):
    lat = lat * math.pi / 180.0
    lon = lon * math.pi / 180.0
    freqs = torch.arange(1, k+1, device=lat.device, dtype=lat.dtype)
    lat_feats = torch.cat([torch.sin(freqs * lat.unsqueeze(-1)), torch.cos(freqs * lat.unsqueeze(-1))], dim=-1)
    lon_feats = torch.cat([torch.sin(freqs * lon.unsqueeze(-1)), torch.cos(freqs * lon.unsqueeze(-1))], dim=-1)
    return torch.cat([lat_feats, lon_feats], dim=-1)

class CoordEncoder(nn.Module):
    def __init__(self, out_dim: int = 256, k: int = 16):
        super().__init__()
        in_dim = 4 * k
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, out_dim)
        )
    def forward(self, latlon: torch.Tensor):
        lat, lon = latlon[:,0], latlon[:,1]
        feats = sincos_posenc_latlon(lat, lon, self.k)
        return self.mlp(feats)
