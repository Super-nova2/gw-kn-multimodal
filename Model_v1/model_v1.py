import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnablePeriodicEmbedding(nn.Module):
    """
    Learnable periodic time embedding used by the optical encoder.
    """
    def __init__(self, num_heads, ref_dim):
        super().__init__()
        self.num_heads = num_heads
        self.ref_dim = ref_dim

        self.w0 = nn.Parameter(torch.empty(1, 1, num_heads, 1))
        self.a0 = nn.Parameter(torch.empty(1, 1, num_heads, 1))
        self.wi = nn.Parameter(torch.empty(1, 1, num_heads, ref_dim - 1))
        self.ai = nn.Parameter(torch.empty(1, 1, num_heads, ref_dim - 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w0)
        nn.init.zeros_(self.a0)

        exponents = torch.linspace(0, self.ref_dim - 2, self.ref_dim - 1)
        scale = 100.0
        log_freqs = -math.log(scale) * (exponents / (self.ref_dim - 1))
        freqs = torch.exp(log_freqs)

        freq_init = freqs.view(1, 1, 1, -1).repeat(1, 1, self.num_heads, 1)
        freq_noise = torch.randn_like(freq_init) * 0.05
        freq_init = freq_init * torch.exp(freq_noise)

        with torch.no_grad():
            self.wi.copy_(freq_init)

        nn.init.uniform_(self.ai, 0, 2 * math.pi)

    def forward(self, t):
        t_expanded = t.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.num_heads, 1)
        linear_term = self.w0 * t_expanded + self.a0
        periodic_term = torch.sin(self.wi * t_expanded + self.ai)
        return torch.cat([linear_term, periodic_term], dim=-1)


class SpatialEmbedding(nn.Module):
    """
    Simple MLP to embed RA/Dec coordinates into a feature space.
    """
    def __init__(self, output_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, coordinates):
        ra = coordinates[:, 0]
        dec = coordinates[:, 1]
        theta = (90.0 - dec) * (math.pi / 180.0)
        phi = ra * (math.pi / 180.0)

        x = torch.cos(theta) * torch.cos(phi)
        y = torch.cos(theta) * torch.sin(phi)
        z = torch.sin(theta)

        coords = torch.stack([x, y, z], dim=1)
        return self.fc(coords)


class MultiTimeAttention(nn.Module):
    """
    Multi-time attention module for irregular optical light curves.
    """
    def __init__(self, input_dim, num_heads, ref_dim, k_dim, output_dim):
        super().__init__()
        self.H = num_heads
        self.D = input_dim
        self.d_k = k_dim
        self.sqrt_dk = math.sqrt(k_dim)

        self.W = nn.Linear(ref_dim, k_dim, bias=False)
        self.V = nn.Linear(ref_dim, k_dim, bias=False)
        self.U = nn.Linear(num_heads * input_dim, output_dim)
        self.lambda_param = nn.Parameter(torch.tensor(1.0))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.kaiming_uniform_(self.U.weight, a=math.sqrt(5))
        if self.U.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.U.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.U.bias, -bound, bound)

    def forward(self, query_emb, key_emb, values, mask=None, errors=None):
        batch_size = query_emb.size(0)

        Q_proj = self.W(query_emb)
        K_proj = self.V(key_emb)

        Q_perm = Q_proj.permute(0, 2, 1, 3)
        K_perm = K_proj.permute(0, 2, 1, 3)

        scores = torch.matmul(Q_perm, K_perm.transpose(-1, -2)) / self.sqrt_dk
        scores_expanded = scores.unsqueeze(-1).expand(-1, -1, -1, -1, self.D)

        if errors is not None:
            errors_expanded = errors.unsqueeze(1).unsqueeze(1)
            sigma_sq = torch.square(errors_expanded)
            error_bias = -self.lambda_param * torch.log(sigma_sq + 1e-9)
            scores_expanded = scores_expanded + error_bias

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(1)
            scores_expanded = scores_expanded.masked_fill(mask_expanded == 0, -1e9)

        attn_weights = F.softmax(scores_expanded, dim=3)
        v_expanded = values.unsqueeze(1).unsqueeze(1)
        x_hat = torch.sum(attn_weights * v_expanded, dim=3)
        x_hat_flat = x_hat.permute(0, 2, 1, 3).reshape(batch_size, -1, self.H * self.D)
        return self.U(x_hat_flat)


class OpticalEncoderWithCLS(nn.Module):
    """
    Optical encoder that outputs a CLS vector and a time-series feature sequence.
    """
    def __init__(self, input_dim, num_heads=4, ref_dim=64, k_dim=64, output_dim=128, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.ref_dim = ref_dim

        self.time_embedding = LearnablePeriodicEmbedding(num_heads, ref_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, num_heads, ref_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.mtan = MultiTimeAttention(input_dim, num_heads, ref_dim, k_dim, output_dim)
        self.spatial_embedding = SpatialEmbedding(output_dim=ref_dim)
        self.spatial_proj = nn.Linear(ref_dim, num_heads * ref_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, ra_dec_obs, t_obs, values_obs, t_ref, mask=None, errors_obs=None):
        batch_size = t_obs.size(0)

        spatial_feat = self.spatial_embedding(ra_dec_obs)
        spatial_emb = self.spatial_proj(spatial_feat).view(
            batch_size, 1, self.num_heads, self.ref_dim
        )

        key_emb = self.time_embedding(t_obs)
        ref_time_emb = self.time_embedding(t_ref)
        cls_emb = self.cls_token.expand(batch_size, -1, -1, -1) + spatial_emb
        query_emb = torch.cat([cls_emb, ref_time_emb], dim=1)

        full_output = self.mtan(query_emb, key_emb, values_obs, mask, errors_obs)
        full_output = self.output_dropout(full_output)

        z_l = full_output[:, 0, :]
        h_l = full_output[:, 1:, :]
        return z_l, h_l


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion head for GW and optical sequence features.
    """
    def __init__(self, gw_dim, opt_dim, attn_dim=None, hidden_dim=None, dropout=0.1):
        super().__init__()
        self.attn_dim = attn_dim if attn_dim is not None else opt_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else self.attn_dim * 2

        self.q_proj = nn.Linear(gw_dim, self.attn_dim)
        self.k_proj = nn.Linear(opt_dim, self.attn_dim)
        self.v_proj = nn.Linear(opt_dim, self.attn_dim)
        self.out_norm = nn.LayerNorm(self.attn_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.attn_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 2)
        )

    def forward(self, g_feat, h_l):
        q = self.q_proj(g_feat).unsqueeze(1)
        k = self.k_proj(h_l)
        v = self.v_proj(h_l)

        attn_scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.attn_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        fused = torch.matmul(attn_weights, v).squeeze(1)
        fused = self.out_norm(fused)

        logits = self.classifier(fused)
        return logits, fused


class GWPixelMLPEncoder(nn.Module):
    """
    MLP encoder for GW scalars and matched skymap pixel features.
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class GWOpticalFusionModel(nn.Module):
    """
    Simplified model: GW encoder + optical encoder + fusion classifier.
    """
    def __init__(
        self,
        gw_input_dim,
        optical_input_dim=6,
        ref_time_dim=64,
        enc_dim=128,
        fusion_attn_dim=None,
        fusion_hidden_dim=None,
        fusion_dropout=0.1,
        gw_dropout=0.1,
        opt_dropout=0.1,
        label_smoothing=0.0
    ):
        super().__init__()
        self.gw_encoder = GWPixelMLPEncoder(
            input_dim=gw_input_dim,
            hidden_dim=enc_dim,
            output_dim=enc_dim,
            dropout=gw_dropout
        )
        self.optical_encoder = OpticalEncoderWithCLS(
            input_dim=optical_input_dim,
            output_dim=enc_dim,
            num_heads=4,
            ref_dim=ref_time_dim,
            dropout=opt_dropout
        )
        self.fusion = CrossAttentionFusion(
            gw_dim=enc_dim,
            opt_dim=enc_dim,
            attn_dim=fusion_attn_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout
        )
        self.cls_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def encode(self, gw_input, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err):
        g = self.gw_encoder(gw_input)
        z_l, h_l = self.optical_encoder(
            opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, errors_obs=opt_err
        )
        return g, z_l, h_l

    def encode_optical(self, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err):
        z_l, h_l = self.optical_encoder(
            opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, errors_obs=opt_err
        )
        return z_l, h_l

    def fusion_logits(self, g_feat, h_l):
        logits, _ = self.fusion(g_feat, h_l)
        return logits

    def forward(self, gw_input, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err):
        g, _, h_l = self.encode(gw_input, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err)
        return self.fusion_logits(g, h_l)
