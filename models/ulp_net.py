import torch
import torch.nn as nn

class ULPNet(nn.Module):
    def __init__(self, k: int = 32, act: str = "relu"):
        super().__init__()
        self.row_conv = nn.Sequential(nn.Conv2d(1, k, 1, bias=False), nn.BatchNorm2d(k), self._get_act(act))
        self.col_conv = nn.Sequential(nn.Conv2d(1, k, 1, bias=False), nn.BatchNorm2d(k), self._get_act(act))
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(2 * k, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), self._get_act(act),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), self._get_act(act),
        )
        self.head = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=False), nn.BatchNorm2d(256), self._get_act(act),
            nn.Conv2d(256, 1, 1),
        )

    @staticmethod
    def _get_act(act: str) -> nn.Module:
        return {"relu": nn.ReLU(True), "leaky_relu": nn.LeakyReLU(0.01, True), "gelu": nn.GELU(), "silu": nn.SiLU(True), "tanh": nn.Tanh()}[act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        f_row = self.row_conv(x).mean(dim=3, keepdim=True).expand(-1, -1, h, w)
        f_col = self.col_conv(x).mean(dim=2, keepdim=True).expand(-1, -1, h, w)
        feat = self.conv_blocks(torch.cat([f_row, f_col], dim=1))
        return self.head(feat.mean(dim=2, keepdim=True))[:, 0, 0, 1:]
