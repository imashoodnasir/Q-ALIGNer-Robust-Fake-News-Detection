import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import TextEncoder, ImageEncoder
from .quantum_layer import QuantumEntangler

class QAligner(nn.Module):
    def __init__(self, num_classes=2, text_model='distilbert-base-uncased', image_model='resnet18',
                 proj_dim=64, use_quantum=True, num_qubits=6, q_depth=2, device='cpu'):
        super().__init__()
        self.text_enc = TextEncoder(text_model)
        self.img_enc  = ImageEncoder(image_model)
        self.t_proj = nn.Linear(self.text_enc.out_dim, proj_dim)
        self.v_proj = nn.Linear(self.img_enc.out_dim, proj_dim)
        self.use_quantum = use_quantum
        self.device_str = 'default.qubit'
        if use_quantum:
            self.quantum = QuantumEntangler(num_qubits=num_qubits, depth=q_depth, out_dim=proj_dim//8 if proj_dim>=8 else proj_dim, device_str=self.device_str)
            self.q_proj = nn.Linear(proj_dim//8 if proj_dim>=8 else proj_dim, proj_dim)
        else:
            self.quantum = None
        # Heads
        self.classifier = nn.Linear(proj_dim, num_classes)
        self.consistency_head = nn.Linear(proj_dim, 1)

    def forward(self, batch):
        # Text
        t_feats = self.text_enc(batch["input_ids"], batch["attention_mask"])   # (B, Ht)
        t_feats = self.t_proj(t_feats)                                         # (B, D)
        # Image
        v_feats = self.img_enc(batch["image"])                                 # (B, Hv)
        v_feats = self.v_proj(v_feats)                                         # (B, D)

        # L2 normalize for stability
        t = F.normalize(t_feats, dim=-1)
        v = F.normalize(v_feats, dim=-1)

        # Quantum fusion (simple: feed (t+v)/2 through quantum circuit to get fused feature)
        if self.quantum is not None:
            fused_in = 0.5 * (t + v)
            q_out = self.quantum(fused_in)                                     # (B, d_q)
            q_out = self.q_proj(q_out)                                         # (B, D)
            z = F.normalize(q_out + t + v, dim=-1)                             # residual add
        else:
            z = F.normalize(t + v, dim=-1)

        logits = self.classifier(z)
        cons   = torch.sigmoid(self.consistency_head(torch.abs(t - v)))        # simple proxy for consistency
        return {
            "logits": logits,
            "consistency": cons.squeeze(-1),
            "t": t, "v": v, "z": z
        }
