import torch
import torch.nn as nn

from models import register
from torch.nn import TransformerEncoderLayer, TransformerEncoder, LayerNorm
from torch.nn.init import xavier_uniform_
from einops import rearrange


@register('transformer_neck')
class TransformerNeck(nn.Module):
    def __init__(self, in_dim, out_dim=256, num_encoder_layers=3, dim_feedforward=512):
        super().__init__()
        self.input_proj = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
        self.global_op = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        encoder_layer = TransformerEncoderLayer(d_model=out_dim,
                                                nhead=8,
                                                dim_feedforward=dim_feedforward,
                                                dropout=0.1,
                                                activation="gelu",
                                                batch_first=True
                                                )
        encoder_norm = LayerNorm(out_dim)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.input_proj(x)  # x: BxCxHxW
        H = x.size(2)
        global_content = self.global_op(x)  # BxCx1x1
        x = rearrange(x, 'B C H W -> B (H W) C')
        global_content = rearrange(global_content, 'B C H W -> B (H W) C')
        all_x = torch.cat((global_content, x), dim=1)  # B (1+HW) C
        out_put = self.encoder(all_x)  # B (1+HW) C
        global_content = out_put[:, 0:1, :]
        x_rep = out_put[:, 1:, :]
        x_rep = rearrange(x_rep, 'B (H W) C -> B C H W', H=H)
        return global_content, x_rep
