import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, value: float):
        self.lambda_ = float(value)


def _build_mlp(in_dim, hidden_dim, out_dim, dropout=0.0, final_bias=True):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim, bias=final_bias),
    )


def normalize_fusion_mode(fusion_mode=None, dual_fusion=False):
    if fusion_mode is None:
        return "legacy_dual" if bool(dual_fusion) else "legacy_g2o"
    mode = str(fusion_mode).strip().lower()
    if mode in {"", "none", "null"}:
        return "legacy_dual" if bool(dual_fusion) else "legacy_g2o"
    if mode in {"legacy_g2o", "legacy_dual", "physical_dual_hgw"}:
        return mode
    raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

# ==============================================================================
# 1. Learnable Periodic Time Embedding (phi_h(t)) and Spatial Embedding Module
# ==============================================================================
class LearnablePeriodicEmbedding(nn.Module):
    """
    Implements the time embedding phi_h(t) described in Equation (1).
    
    phi_h(t)[i] = w_0h * t + a_0h             if i = 0 (Linear term)
                = sin(w_ih * t + a_ih)        if 0 < i < d_r (Periodic term)
    """
    def __init__(self, num_heads, ref_dim):
        """
        Args:
            num_heads (H): Number of attention heads.
            ref_dim (d_r): Dimension of the reference embedding per head.
        """
        super().__init__()
        self.num_heads = num_heads
        self.ref_dim = ref_dim
        
        # Parameters for the linear term (i=0)
        # Shape: [1, 1, H, 1] for broadcasting over batch and time
        self.w0 = nn.Parameter(torch.empty(1, 1, num_heads, 1)) 
        self.a0 = nn.Parameter(torch.empty(1, 1, num_heads, 1))
        
        # Parameters for the periodic terms (0 < i < d_r)
        # We need (d_r - 1) frequencies per head
        self.wi = nn.Parameter(torch.empty(1, 1, num_heads, ref_dim - 1))
        self.ai = nn.Parameter(torch.empty(1, 1, num_heads, ref_dim - 1))

        self.reset_parameters()


    def reset_parameters(self):
        """
        Custom initialization logic.
        """
        # 1. Linear Terms: Use Xavier Uniform for weights, Zero for bias
        nn.init.xavier_uniform_(self.w0)
        nn.init.zeros_(self.a0)
        
        # 2. Periodic Frequencies (wi): Log-Linear Initialization
        # We want frequencies to span a range (e.g., from 1.0 to 100.0) geometrically.
        # This is similar to the fixed positional encoding in Transformers.
        # Generate exponents linearly spaced
        exponents = torch.linspace(0, self.ref_dim - 2, self.ref_dim - 1)
        
        # Scale factor (e.g., 100.0 means frequencies range from ~1 to ~100)
        # Formula: freq = scale ^ (i / d)
        scale = 100.0
        log_freqs = -math.log(scale) * (exponents / (self.ref_dim - 1))
        freqs = torch.exp(log_freqs) # Shape: [d-1]
        
        # Broadcast to heads [1, 1, H, d-1]
        freq_init = freqs.view(1, 1, 1, -1).repeat(1, 1, self.num_heads, 1)
        
        # Add small random noise so heads are not identical
        freq_noise = torch.randn_like(freq_init) * 0.05
        freq_init = freq_init * torch.exp(freq_noise)
        
        with torch.no_grad():
            self.wi.copy_(freq_init)
        
        # 3. Periodic Phases (ai): Uniform distribution over [0, 2pi]
        nn.init.uniform_(self.ai, 0, 2 * math.pi)

    def forward(self, t):
        """
        Args:
            t: Time points tensor. Shape [Batch, Seq_Len]
            
        Returns:
            phi(t): Embedded time. Shape [Batch, Seq_Len, H, d_r]
        """
        # Expand t to [Batch, Seq_Len, H, 1] to match heads
        t_expanded = t.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.num_heads, 1)
        
        # 1. Compute Linear Term (i=0): w_0h * t + a_0h
        # Output: [Batch, Seq_Len, H, 1]
        linear_term = self.w0 * t_expanded + self.a0
        
        # 2. Compute Periodic Terms (i > 0): sin(w_ih * t + a_ih)
        # Output: [Batch, Seq_Len, H, d_r - 1]
        # Note: t_expanded broadcasts against self.wi [1, 1, H, d_r-1]
        periodic_term = torch.sin(self.wi * t_expanded + self.ai)
        
        # 3. Concatenate along the last dimension to form d_r
        # Output: [Batch, Seq_Len, H, d_r]
        return torch.cat([linear_term, periodic_term], dim=-1)

class SpatialEmbedding(nn.Module):
    def __init__(self, output_dim=64):
        super().__init__()
        # Spatial Embedding for Spherical Coordinates (RA, Dec)
        # Using a simple MLP to map (x, y, z) on unit sphere to embedding
        self.fc = nn.Sequential(
            nn.Linear(3, output_dim), # Input: (x, y, z)
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    @staticmethod
    def coordinates_to_xyz(coordinates):
        # coordinates: [Batch, 2] (RA in degrees, Dec in degrees)
        # convert (RA, Dec) in degrees to radians
        ra = coordinates[:, 0]   # [Batch]
        dec = coordinates[:, 1]  # [Batch]
        theta = (90.0 - dec) * (math.pi / 180.0)  # polar angle
        phi = ra * (math.pi / 180.0)               # azimuthal angle

        # Convert Spherical to Cartesian Coordinates
        # Standard convention consistent with HEALPix (data_loader.sample_moc_skymap)
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)

        return torch.stack([x, y, z], dim=1) # [Batch, 3]

    def forward(self, coordinates, return_xyz=False):
        coords = self.coordinates_to_xyz(coordinates)
        feat = self.fc(coords)
        if return_xyz:
            return feat, coords
        return feat

# ==============================================================================
# 2. Multi-Time Attention Network (mTAN) Module
# ==============================================================================
class MultiTimeAttention(nn.Module):
    """
    Updated mTAN module to support unaligned/irregular sampling across dimensions.
    
    Key Change: 
    - Accepts mask of shape [Batch, L, D] (per-variable masking).
    - Computes attention scores independently for each dimension D.
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

        self.lambda_param = nn.Parameter(torch.tensor(1.0))  # Learnable scalar parameter

        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Initialize projection weights.
        """
        # W, V: Xavier Uniform (standard for attention projections)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.V.weight)
        
        # U: Kaiming Uniform (assuming ReLU follows in the next block)
        # If no ReLU follows immediately, Xavier is also acceptable.
        nn.init.kaiming_uniform_(self.U.weight, a=math.sqrt(5))
        if self.U.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.U.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.U.bias, -bound, bound)

    def forward(self, query_emb, key_emb, values, mask=None, errors=None):
        """
        Args:
            query_emb: [Batch, K, H, d_r]
            key_emb:   [Batch, L, H, d_r]
            values:    [Batch, L, D]
            mask:      [Batch, L, D] (1.0 for observed, 0.0 for padding/missing)
            errors:    [Batch, L, D] (Measurement errors / Sigma). Optional.
                       If provided, implements Inverse Variance Weighting.
        Returns:
            h: [Batch, K, J]
        """
        batch_size = query_emb.size(0)
        
        # --- 1. Compute Raw Attention Scores (Shared across D) ---
        # Q: [B, K, H, d_k]
        # K: [B, L, H, d_k]
        Q_proj = self.W(query_emb) 
        K_proj = self.V(key_emb)
        
        # Permute for matmul
        Q_perm = Q_proj.permute(0, 2, 1, 3) # [B, H, K, d_k]
        K_perm = K_proj.permute(0, 2, 1, 3) # [B, H, L, d_k]
        
        # Raw Scores: [B, H, K, L]
        # This represents similarity between Reference Time K and Observed Time L
        scores = torch.matmul(Q_perm, K_perm.transpose(-1, -2)) / self.sqrt_dk
        
        # --- 2. Handle Dimension-Specific Masking ---
        # We need independent attention distributions for each feature dimension D.
        # Eq (4) says Softmax is over observed points t_id *for that dimension*.
        
        # Expand Scores to D: [B, H, K, L, 1] -> [B, H, K, L, D]
        scores_expanded = scores.unsqueeze(-1).expand(-1, -1, -1, -1, self.D)

        if errors is not None:
            # errors shape: [Batch, L, D]
            # We treat 'errors' as sigma. We want to add log(1 / sigma^2) to logits.
            # Formula: Logit_new = Logit_original - log(sigma^2 + epsilon)
            
            # Expand errors to match scores shape: [B, 1, 1, L, D]
            errors_expanded = errors.unsqueeze(1).unsqueeze(1)
            
            # Compute bias term
            # Adding epsilon (1e-9) to avoid log(0) for perfect measurements or zero-padding
            # Note: For zero-padding (error=0), this bias becomes large positive, 
            # BUT the 'mask' step below will force them to -inf anyway.
            sigma_sq = torch.square(errors_expanded)
            error_bias = - self.lambda_param * torch.log(sigma_sq + 1e-9)
            
            # Add bias to scores
            scores_expanded = scores_expanded + error_bias
        
        if mask is not None:
            # Mask input: [B, L, D]
            # Expand Mask to match scores: [B, 1, 1, L, D]
            mask_expanded = mask.unsqueeze(1).unsqueeze(1)
            
            # Apply Mask: Set scores of missing bands/times to -inf
            # This ensures exp(-inf) = 0 in Softmax
            scores_expanded = scores_expanded.masked_fill(mask_expanded == 0, -1e9)
        
        # --- 3. Compute Softmax per Dimension ---
        # Softmax over L dimension (dim=3)
        # attn_weights: [B, H, K, L, D]
        # Now each dimension d has its own attention distribution based on its valid times.
        attn_weights = F.softmax(scores_expanded, dim=3)
        
        # --- 4. Weighted Sum (Interpolation) ---
        # values: [B, L, D] -> [B, 1, 1, L, D]
        v_expanded = values.unsqueeze(1).unsqueeze(1)
        
        # Element-wise multiplication broadcasted over H and K
        # Then sum over L (time axis)
        # x_hat: [B, H, K, D]
        x_hat = torch.sum(attn_weights * v_expanded, dim=3)
        
        # --- 5. Final Projection ---
        # Flatten H and D: [B, K, H * D]
        x_hat_flat = x_hat.permute(0, 2, 1, 3).reshape(batch_size, -1, self.H * self.D)
        
        output = self.U(x_hat_flat) # [Batch, K, J]
        
        return output

# ==============================================================================
# 3. Optical Encoder with CLS Token for GW-Optical Architecture
# ==============================================================================
class OpticalEncoderWithCLS(nn.Module):
    """
    Wrapper module for the Optical Encoder in the GW-Optical architecture.
    It combines:
      1. Periodic Time Embedding
      2. CLS Token Injection
      3. mTAN Attention Module
    """
    def __init__(self, input_dim, num_heads=4, ref_dim=64, k_dim=64, output_dim=128, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.ref_dim = ref_dim
        
        # 1. Time Embedding Layer (phi) [cite: 5, 6]
        self.time_embedding = LearnablePeriodicEmbedding(num_heads, ref_dim)
        
        # 2. Learnable CLS Token Parameter
        # This replaces phi(t) for the first token. 
        # Dimensions must match the embedded time: [1, 1, H, d_r]
        self.cls_token = nn.Parameter(torch.empty(1, 1, num_heads, ref_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02) # BERT initialization
        # self.cls_token.requires_grad = True
        
        # 3. mTAN Core Module [cite: 3, 4]
        self.mtan = MultiTimeAttention(input_dim, num_heads, ref_dim, k_dim, output_dim)

        # 4. spatial embedding
        self.spatial_embedding = SpatialEmbedding(output_dim=ref_dim)
        self.spatial_proj = nn.Linear(ref_dim, num_heads * ref_dim)
        self.output_dropout = nn.Dropout(dropout)

    def encode_coordinates(self, ra_dec_obs, return_xyz=False):
        spatial_feat, coord_xyz = self.spatial_embedding(ra_dec_obs, return_xyz=True)
        if return_xyz:
            return spatial_feat, coord_xyz
        return spatial_feat

    def forward(self, ra_dec_obs, t_obs, values_obs, t_ref, mask=None, errors_obs=None):
        """
        Args:
            ra_dec_obs: [Batch, 2] (RA, Dec in radians for observed points)
            t_obs: [Batch, L] (Union of all time points)
            values_obs: [Batch, L, D] (Values, padded with 0 where missing)
            t_ref: [Batch, N]
            mask: [Batch, L, D] (1 if band d is observed at time l, else 0)
            
        Returns:
            z_l: The global CLS vector [Batch, J] (for Alignment)
            H_l: The temporal matrix [Batch, N, J] (for Fusion)
        """
        batch_size = t_obs.size(0)
        
        # 0. Compute Spatial Embedding for Observed Points
        # Shape: [Batch, H * d_r]
        spatial_feat = self.encode_coordinates(ra_dec_obs)   # [Batch, ref_dim]
        spatial_emb = self.spatial_proj(spatial_feat).view(batch_size, 1, self.num_heads, self.ref_dim) # [Batch, 1, H, d_r]

        # 1. Embed Observed Times (Keys) -> phi(t_id)
        # Shape: [Batch, L, H, d_r]
        key_emb = self.time_embedding(t_obs)
        
        # 2. Embed Reference Times (Queries Part A) -> phi(t)
        # Shape: [Batch, N, H, d_r]
        ref_time_emb = self.time_embedding(t_ref)
        
        # 3. Prepare CLS Token (Queries Part B)
        # Expand learnable parameter to batch size
        # Shape: [Batch, 1, H, d_r]
        cls_emb = self.cls_token.expand(batch_size, -1, -1, -1) + spatial_emb
        
        # 4. Concatenate CLS and Reference Embeddings
        # Total Query Shape: [Batch, N + 1, H, d_r]
        query_emb = torch.cat([cls_emb, ref_time_emb], dim=1)
        
        # 5. Pass through mTAN
        # Output Shape: [Batch, N + 1, J]
        full_output = self.mtan(query_emb, key_emb, values_obs, mask, errors_obs)
        full_output = self.output_dropout(full_output)
        
        # 6. Split Output
        # Index 0 is CLS (z_l), Indices 1..N are time-series (H_l)
        z_l = full_output[:, 0, :]   # [Batch, J]
        H_l = full_output[:, 1:, :]  # [Batch, N, J]
        
        return z_l, H_l


class OpticalEncoderWithCLSNoCoord(nn.Module):
    """
    Optical encoder variant without coordinate conditioning.
    Used by optical-only pipeline to avoid spatial branch dependence.
    """
    def __init__(self, input_dim, num_heads=4, ref_dim=64, k_dim=64, output_dim=128, dropout=0.0):
        super().__init__()
        self.time_embedding = LearnablePeriodicEmbedding(num_heads, ref_dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, num_heads, ref_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.mtan = MultiTimeAttention(input_dim, num_heads, ref_dim, k_dim, output_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, t_obs, values_obs, t_ref, mask=None, errors_obs=None):
        batch_size = t_obs.size(0)

        key_emb = self.time_embedding(t_obs)
        ref_time_emb = self.time_embedding(t_ref)
        cls_emb = self.cls_token.expand(batch_size, -1, -1, -1)
        query_emb = torch.cat([cls_emb, ref_time_emb], dim=1)

        full_output = self.mtan(query_emb, key_emb, values_obs, mask, errors_obs)
        full_output = self.output_dropout(full_output)

        z_l = full_output[:, 0, :]
        h_l = full_output[:, 1:, :]
        return z_l, h_l

# ==============================================================================
# 4. 1D ResNet Basic Module
# ==============================================================================
class BasicBlock1D(nn.Module):
    """
    1D version of the ResNet BasicBlock.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()
        
        # First convolution: handling stride for downsampling
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second convolution
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection:
        # If input shape (channels or length) differs from output, project input to match.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet1D(nn.Module):
    """
    1D ResNet Encoder designed for long sequences (e.g., MOC Skymaps).
    Architecture is based on ResNet-18 but adapted for 1D data.
    """
    def __init__(self, in_channels, hidden_dim=256, output_dim=128):
        """
        Args:
            in_channels: Number of input channels (flexible, not fixed to 5).
            hidden_dim: Dimension of the hidden layer in the projection head.
            output_dim: Dimension of the final output vector.
        """
        super(ResNet1D, self).__init__()
        self.current_planes = 64
        
        # Initial Convolution: Rapidly downsample high-res input
        # Input: [B, in_channels, L=19200] -> Output: [B, 64, L/4=4800] (Stride=4)
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=4, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        # Pooling: [B, 64, L/4=4800] -> [B, 64, L/8=2400]
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet Layers (Stacking Blocks)
        # Layer 1: Keep dimension 64, Length same
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        # Layer 2: Dim 64 -> 128, Length halved
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        # Layer 3: Dim 128 -> 256, Length halved
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        # Layer 4: Dim 256 -> 512, Length halved
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # Global Average Pooling + Fully Connected Head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(512 * BasicBlock1D.expansion, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def _make_layer(self, in_planes, planes, blocks, stride):
        layers = []
        # The first block in a layer handles the stride (downsampling)
        layers.append(BasicBlock1D(in_planes, planes, stride))
        
        # Subsequent blocks maintain the dimension
        # (Note: For BasicBlock, expansion is 1, so in_planes matches planes)
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(planes, planes))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [Batch, in_channels, Length]
        x = self.conv1(x)   # [Batch, 64, L/4=4800]
        x = self.bn1(x)  # [Batch, 64, L/4=4800]
        x = self.relu(x)    
        x = self.maxpool(x)  # [Batch, 64, L/8=2400]

        x = self.layer1(x)  # [Batch, 64, L/8=2400]
        x = self.layer2(x)  # [Batch, 128, L/16=1200]
        x = self.layer3(x)  # [Batch, 256, L/32=600]
        x = self.layer4(x)  # [Batch, 512, L/64=300]
        feature_map = x     # preserve spatial features before pooling

        x = self.avgpool(x) # [Batch, 512, 1]
        x = x.flatten(1)    # [Batch, 512]
        x = self.fc(x)      # [Batch, Output_Dim]

        return x, feature_map

# ==============================================================================
# 5. Dual-Stream Fusion Encoder
# ==============================================================================
class GWMOCResNetEncoder(nn.Module):
    """
    Dual-Stream Encoder:
    1. Scalar Stream: MLP for mass, spin, etc.
    2. Image/Sequence Stream: 1D ResNet for MOC Skymap sequences.
    """
    def __init__(self, 
                 scalar_input_dim, 
                 skymap_channels,          # <--- Flexible channel input
                 scalar_hidden_dim=256, 
                 resnet_output_dim=128,
                 final_output_dim=128,
                 dropout=0.1,
                 need_param_head=False):
        super().__init__()
        
        # --- 1. Scalar Encoder (MLP) ---
        self.scalar_enc = nn.Sequential(
            nn.Linear(scalar_input_dim, scalar_hidden_dim),
            nn.BatchNorm1d(scalar_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(scalar_hidden_dim, scalar_hidden_dim),
            nn.BatchNorm1d(scalar_hidden_dim),
            nn.ReLU()
        )
        self.scalar_head = None
        if need_param_head:
            self.scalar_head = nn.Sequential(
                nn.Linear(scalar_hidden_dim, final_output_dim),
                nn.BatchNorm1d(final_output_dim),
                nn.ReLU(),
            )
        
        # --- 2. Skymap Encoder (1D ResNet) ---
        # Initialize ResNet with the specific number of channels provided
        self.skymap_enc = ResNet1D(
            in_channels=skymap_channels,
            output_dim=resnet_output_dim
        )
        
        # --- 3. Fusion Layer ---
        # Concatenate both features and project to final 'g' vector
        fusion_input_dim = scalar_hidden_dim + resnet_output_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, final_output_dim * 2),
            nn.BatchNorm1d(final_output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_output_dim * 2, final_output_dim)
            # Final output 'g' vector
        )

        # --- 4. Skymap sequence projection for dual fusion ---
        # Projects ResNet feature map (512-d) to enc_dim for cross-attention
        self.skymap_seq_proj = nn.Sequential(
            nn.Conv1d(512, final_output_dim, kernel_size=1),
            nn.BatchNorm1d(final_output_dim),
            nn.ReLU()
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Custom weight initialization for Conv1d, Linear, and BatchNorm1d layers.
        """
        for m in self.modules():
            # 1. Process Convolutional Layers (Conv1d)
            if isinstance(m, nn.Conv1d):
                # mode='fan_out' , making constant variance in the backward pass
                # nonlinearity='relu' for ReLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            # 2. Process Fully Connected Layers (Linear)
            elif isinstance(m, nn.Linear):
                # Linear layers also use Kaiming initialization (Xavier is also possible, but Kaiming is better after ReLU)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            # 3. Process Normalization Layers (BatchNorm1d) - Standard initialization
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # 4. ResNet Zero-Gamma Initialization
        # For each BasicBlock1D in ResNet, set the last BatchNorm weight to zero
        for m in self.modules():
            if isinstance(m, BasicBlock1D):
                # Set bn2 weight to zero
                nn.init.constant_(m.bn2.weight, 0)
    

    def encode_scalar_only(self, gw_scalars):
        h_scalar = self.scalar_enc(gw_scalars)
        if self.scalar_head is None:
            return h_scalar
        return self.scalar_head(h_scalar)

    def encode_features(self, gw_scalars, skymap_sequence):
        """
        Args:
            gw_scalars: [Batch, D_scalar]
            skymap_sequence: [Batch, skymap_channels, Length]
                             (e.g., [B, 7, 19200])
        Returns:
            g: [Batch, Final_Dim]
            g_param: [Batch, Final_Dim] or [Batch, scalar_hidden_dim]
            H_gw: [Batch, SeqLen, Final_Dim] — spatial feature map for dual fusion
        """
        # Encode Scalars
        h_scalar = self.scalar_enc(gw_scalars)  # [B, scalar_hidden]
        g_param = self.scalar_head(h_scalar) if self.scalar_head is not None else h_scalar

        # Encode Skymap Sequence
        h_skymap, skymap_feat = self.skymap_enc(skymap_sequence)  # h_skymap: [B, resnet_output], skymap_feat: [B, 512, 300]

        # Project spatial feature map: [B, 512, 300] -> [B, enc_dim, 300] -> [B, 300, enc_dim]
        H_gw = self.skymap_seq_proj(skymap_feat).permute(0, 2, 1)

        # Concatenate & Fuse
        combined = torch.cat([h_scalar, h_skymap], dim=1)  # [B, fusion_input_dim]
        g = self.fusion_head(combined)  # [B, final_output_dim]

        return g, g_param, H_gw

    def forward(self, gw_scalars, skymap_sequence):
        g, _g_param, H_gw = self.encode_features(gw_scalars, skymap_sequence)
        return g, H_gw

# ==============================================================================
# 6. Alignment Branch for Contrastive Learning
# ==============================================================================
class ProjectionHead(nn.Module):
    """
    MLP Projection Head used in Contrastive Learning (SimCLR/MoCo/ALBEF style).
    Projects features from Encoder Dim -> Latent Dim.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),  # 防止ITC过拟合
            nn.Linear(hidden_dim, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class OpticalLightCurveEncoder(OpticalEncoderWithCLSNoCoord):
    """
    Coordinate-free optical light-curve encoder.
    Outputs only curve-derived features for both global and token-level branches.
    """


class OpticalCoordEncoder(nn.Module):
    """
    Coordinate-only encoder for optical sky position.
    """

    def __init__(self, output_dim=128):
        super().__init__()
        self.encoder = SpatialEmbedding(output_dim=output_dim)

    def forward(self, opt_coords):
        return self.encoder(opt_coords)


class GWScalarEncoder(nn.Module):
    """
    Scalar-only GW encoder.
    """

    def __init__(self, input_dim=7, hidden_dim=256, output_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, gw_scalars):
        return self.net(gw_scalars)


class GWSkymapEncoder(nn.Module):
    """
    Skymap-only GW encoder that keeps global and token-level outputs separate.
    """

    def __init__(self, in_channels=7, output_dim=128, dropout=0.1, use_lightweight=False):
        super().__init__()
        self.use_lightweight = bool(use_lightweight)
        if self.use_lightweight:
            self.backbone = LightweightSkymapEncoder(
                in_channels=in_channels,
                output_dim=output_dim,
                dropout=dropout,
            )
            seq_in_dim = 128
        else:
            self.backbone = ResNet1D(
                in_channels=in_channels,
                output_dim=output_dim,
            )
            seq_in_dim = 512
        self.seq_proj = nn.Sequential(
            nn.Conv1d(seq_in_dim, output_dim, kernel_size=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self._init_weights()

    def _init_weights(self):
        for m in self.seq_proj.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, skymap_sequence):
        _g_sky_backbone, skymap_feat = self.backbone(skymap_sequence)
        h_tokens = self.seq_proj(skymap_feat)  # [B, D, M]
        g_sky = self.global_pool(h_tokens).squeeze(-1)  # [B, D]
        H_gw = h_tokens.permute(0, 2, 1)  # [B, M, D]
        return g_sky, H_gw


class ContrastiveFuseProj(nn.Module):
    """
    Fusion + projection block used by the contrastive branch.
    """

    def __init__(self, input_dim, proj_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden = int(hidden_dim) if hidden_dim is not None else max(int(proj_dim), int(input_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, proj_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class GWContrastiveFuseProj(nn.Module):
    def __init__(self, scalar_dim, sky_dim, proj_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        self.fuse_proj = ContrastiveFuseProj(
            input_dim=int(scalar_dim) + int(sky_dim),
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, g_scalar, g_sky):
        return self.fuse_proj(torch.cat([g_scalar, g_sky], dim=-1))


class OptContrastiveFuseProj(nn.Module):
    def __init__(self, curve_dim, coord_dim, proj_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        self.fuse_proj = ContrastiveFuseProj(
            input_dim=int(curve_dim) + int(coord_dim),
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, z_curve, coord_feat):
        return self.fuse_proj(torch.cat([z_curve, coord_feat], dim=-1))


class PhysicalDualOpticalEncoder(nn.Module):
    """
    Optical encoder with fully decoupled curve and coordinate branches.
    """

    def __init__(
        self,
        input_dim=6,
        ref_time_dim=64,
        enc_dim=128,
        num_heads=4,
        k_dim=64,
        curve_dropout=0.0,
        proj_dim=256,
        proj_dropout=0.0,
    ):
        super().__init__()
        self.curve_encoder = OpticalLightCurveEncoder(
            input_dim=input_dim,
            num_heads=num_heads,
            ref_dim=ref_time_dim,
            k_dim=k_dim,
            output_dim=enc_dim,
            dropout=curve_dropout,
        )
        self.coord_encoder = OpticalCoordEncoder(output_dim=enc_dim)
        self.contrastive_head = OptContrastiveFuseProj(
            curve_dim=enc_dim,
            coord_dim=enc_dim,
            proj_dim=proj_dim,
            hidden_dim=max(enc_dim * 2, proj_dim),
            dropout=proj_dropout,
        )

    def encode_curve_only(self, opt_t, opt_v, opt_ref_t, opt_mask, opt_err=None, errors_obs=None):
        err = opt_err if opt_err is not None else errors_obs
        return self.curve_encoder(opt_t, opt_v, opt_ref_t, opt_mask, errors_obs=err)

    def encode_coord_only(self, opt_coords):
        return self.coord_encoder(opt_coords)

    def encode_components(self, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err=None, errors_obs=None):
        z_curve, h_l = self.encode_curve_only(
            opt_t, opt_v, opt_ref_t, opt_mask, opt_err=opt_err, errors_obs=errors_obs
        )
        coord_feat = self.encode_coord_only(opt_coords)
        return z_curve, coord_feat, h_l

    def forward(self, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err=None, errors_obs=None):
        z_curve, coord_feat, h_l = self.encode_components(
            opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err=opt_err, errors_obs=errors_obs
        )
        feat_o = self.contrastive_head(z_curve, coord_feat)
        return feat_o, h_l


class PhysicalDualGWEncoder(nn.Module):
    """
    GW encoder with fully decoupled scalar and skymap branches.
    """

    def __init__(
        self,
        scalar_input_dim=7,
        skymap_channels=7,
        enc_dim=128,
        proj_dim=256,
        gw_dropout=0.1,
        proj_dropout=0.0,
        use_lightweight=False,
    ):
        super().__init__()
        scalar_hidden = 128 if use_lightweight else 256
        self.scalar_encoder = GWScalarEncoder(
            input_dim=scalar_input_dim,
            hidden_dim=scalar_hidden,
            output_dim=enc_dim,
            dropout=gw_dropout,
        )
        self.skymap_encoder = GWSkymapEncoder(
            in_channels=skymap_channels,
            output_dim=enc_dim,
            dropout=gw_dropout,
            use_lightweight=use_lightweight,
        )
        self.contrastive_head = GWContrastiveFuseProj(
            scalar_dim=enc_dim,
            sky_dim=enc_dim,
            proj_dim=proj_dim,
            hidden_dim=max(enc_dim * 2, proj_dim),
            dropout=proj_dropout,
        )

    def encode_scalar_only(self, gw_scalars):
        return self.scalar_encoder(gw_scalars)

    def encode_skymap_only(self, skymap_sequence):
        return self.skymap_encoder(skymap_sequence)

    def encode_components(self, gw_scalars, skymap_sequence):
        g_scalar = self.encode_scalar_only(gw_scalars)
        g_sky, h_gw = self.encode_skymap_only(skymap_sequence)
        return g_scalar, g_sky, h_gw

    def forward(self, gw_scalars, skymap_sequence):
        g_scalar, g_sky, h_gw = self.encode_components(gw_scalars, skymap_sequence)
        feat_g = self.contrastive_head(g_scalar, g_sky)
        return feat_g, h_gw

# ==============================================================================
# 7. End-to-End GW-Optical Contrastive Model
# ==============================================================================
class GWOpticalContrastiveModel(nn.Module):
    """
    End-to-End Model Wrapper for Contrastive Training.
    Combines:
      1. GW Encoder (Scalar MLP + Skymap ResNet)
      2. Optical Encoder (mTAN + Attention)
      3. Alignment Head (Projection + Loss)
    """
    def __init__(self, 
                 gw_scalar_dim=7, 
                 gw_skymap_channels=7, 
                 optical_input_dim=6, 
                 ref_time_dim=64,
                 enc_dim=128, 
                 proj_dim=256,
                 temp_init=0.07):
        super().__init__()
        
        # --- 1. Encoders ---
        self.gw_encoder = GWMOCResNetEncoder(
            scalar_input_dim=gw_scalar_dim,
            skymap_channels=gw_skymap_channels,
            final_output_dim=enc_dim
        )
        
        self.optical_encoder = OpticalEncoderWithCLS(
            input_dim=optical_input_dim,
            output_dim=enc_dim,
            num_heads=4,
            ref_dim=ref_time_dim
        )
        
        # --- 2. Projection Heads & Temperature ---
        # Projects Encoder Features (128) -> Latent Space (256)
        self.gw_proj = ProjectionHead(enc_dim, enc_dim, proj_dim)
        self.opt_proj = ProjectionHead(enc_dim, enc_dim, proj_dim)
        
        # We store log_temp to ensure temperature is always positive (via exp)
        self.log_temp = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(temp_init)))

        # --- 3. Standard Cross Entropy Loss ---
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, gw_s, gw_m, opt_t, opt_v, opt_ref_t, opt_mask, opt_err, opt_coords, gw_indices, mask=None):
        """
        Args:
            gw_s, gw_m: GW Inputs
            opt_t, opt_v, ...: Optical Inputs
            gw_indices: [Batch] ID of the GW event (for masking)
        """
        # --- A. Encode Features ---
        # g: [Batch, enc_dim], H_gw: spatial features (discarded for contrastive)
        g, _H_gw = self.gw_encoder(gw_s, gw_m)

        # z_l: [Batch, enc_dim] (We discard H_l for contrastive pre-training)
        z_l, _ = self.optical_encoder(opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, errors_obs=opt_err)
        
        # --- B. Project & Normalize ---
        # [Batch, proj_dim]
        feat_g = F.normalize(self.gw_proj(g), p=2, dim=1, eps=1e-8)
        feat_o = F.normalize(self.opt_proj(z_l), p=2, dim=1, eps=1e-8)
        
        # --- C. Compute Contrastive Loss ---
        loss, logits = self.compute_masked_itc_loss(feat_g, feat_o, gw_indices, mask=None)
        
        return loss, logits

    def compute_masked_itc_loss(self, feat_g, feat_o, gw_indices, mask=None):
        """
        Computes Image-Text Contrastive (ITC) loss.
        Uses gw_indices to handle potential 'false negatives'
        (though BalancedSampler avoids them, this is robust).
        """
        batch_size = feat_g.size(0)
        temperature = torch.clamp(self.log_temp.exp(), min=0.01, max=100.0)

        # 1. Similarity Matrix: [B, B]
        # Divide by temperature (smaller temp → sharper distribution)
        sim_g2o = torch.matmul(feat_g, feat_o.T) / temperature
        sim_o2g = sim_g2o.T
        
        if mask is not None:
            # 2. Ground Truth Mask: [B, B]
            # mask[i, j] = 1 if sample i and j come from the same GW event
            # If using BalancedSampler, this is just an identity matrix.
            labels_mask = (gw_indices.unsqueeze(0) == gw_indices.unsqueeze(1)).float()
            
            # 3. Compute Loss (Masked Cross Entropy)
            # We want to maximize similarity for all positive pairs (where mask == 1)
            
            # For numerical stability with Softmax
            sim_g2o_max, _ = torch.max(sim_g2o, dim=1, keepdim=True)
            sim_g2o = sim_g2o - sim_g2o_max.detach()
            
            sim_o2g_max, _ = torch.max(sim_o2g, dim=1, keepdim=True)
            sim_o2g = sim_o2g - sim_o2g_max.detach()
            
            # Log-Softmax denominator (sum over all samples in batch)
            exp_g2o = torch.exp(sim_g2o)
            exp_o2g = torch.exp(sim_o2g)
            
            # Note: If BalancedSampler is used, this simplifies to standard CE.
            # Here we implement the generic form for safety.
            
            # Log-prob of positive pairs
            # sum(exp(positives)) / sum(exp(all))
            log_prob_g2o = sim_g2o - torch.log(exp_g2o.sum(dim=1, keepdim=True))
            log_prob_o2g = sim_o2g - torch.log(exp_o2g.sum(dim=1, keepdim=True))
            
            # Compute mean loss over positive pairs
            # We only care about entries where labels_mask == 1
            loss_g = - (labels_mask * log_prob_g2o).sum(dim=1) / labels_mask.sum(dim=1)
            loss_o = - (labels_mask * log_prob_o2g).sum(dim=1) / labels_mask.sum(dim=1)

            total_loss = (loss_g.mean() + loss_o.mean()) / 2
        else:
            # 4. Standard Cross Entropy Labels
            labels = torch.arange(batch_size, device=feat_g.device)
            # Compute Symmetric Loss
            # Loss 1: Given GW, classify correct Optical
            loss_g = self.criterion(sim_g2o, labels)
            # Loss 2: Given Optical, classify correct GW
            loss_o = self.criterion(sim_o2g, labels)
            
            total_loss = (loss_g + loss_o) / 2
        
        return total_loss, sim_g2o

# ==============================================================================
# 8. Fusion Branch and Joint ALBEF-Style Model
# ==============================================================================
class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion head for GW-optical matching.

    When dual=False (legacy): GW queries optical → classifier(fused_opt).
    When dual=True:  GW queries optical + optical CLS queries GW spatial map
                     + per-pair credible level → classifier(concat).
    """
    def __init__(
        self,
        gw_dim,
        opt_dim,
        coord_dim=None,
        attn_dim=None,
        hidden_dim=None,
        dropout=0.1,
        dual=False,
        fusion_mode="legacy_g2o",
        use_cred_level_feature=False,
        use_similarity_as_cls_input=False,
    ):
        super().__init__()
        self.attn_dim = d = attn_dim if attn_dim is not None else opt_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else d * 2
        self.dual = dual
        self.fusion_mode = normalize_fusion_mode(fusion_mode, dual_fusion=dual)
        self.use_cred_level_feature = bool(use_cred_level_feature)
        self.use_similarity_as_cls_input = bool(use_similarity_as_cls_input)
        coord_dim = opt_dim if coord_dim is None else int(coord_dim)

        if self.fusion_mode in {"legacy_g2o", "legacy_dual"}:
            self.g2o_q = nn.Linear(gw_dim, d)
            self.g2o_k = nn.Linear(opt_dim, d)
            self.g2o_v = nn.Linear(opt_dim, d)
            self.g2o_norm = nn.LayerNorm(d)

            if self.fusion_mode == "legacy_dual":
                self.o2g_q = nn.Linear(opt_dim, d)
                self.o2g_k = nn.Linear(gw_dim, d)
                self.o2g_v = nn.Linear(gw_dim, d)
                self.o2g_norm = nn.LayerNorm(d)
                cls_input_dim = d * 2 + 1
            else:
                cls_input_dim = d
        else:
            self.param2opt_q = nn.Linear(gw_dim, d)
            self.param2opt_k = nn.Linear(opt_dim, d)
            self.param2opt_v = nn.Linear(opt_dim, d)
            self.param2opt_norm = nn.LayerNorm(d)

            self.coord2gw_q = nn.Linear(coord_dim, d)
            self.coord2gw_k = nn.Linear(gw_dim, d)
            self.coord2gw_v = nn.Linear(gw_dim, d)
            self.coord2gw_norm = nn.LayerNorm(d)

            cls_input_dim = d * 2
            if self.use_similarity_as_cls_input:
                cls_input_dim += 1
            if self.use_cred_level_feature:
                cls_input_dim += 1

        self.classifier = nn.Sequential(
            nn.Linear(cls_input_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 2)
        )

    def uses_cred_level_input(self):
        if self.fusion_mode == "legacy_dual":
            return True
        return bool(self.use_cred_level_feature)

    def forward(
        self,
        g_feat,
        h_l,
        z_l=None,
        H_gw=None,
        cred_level=None,
        g_param=None,
        coord_feat=None,
        sim_itc_pair=None,
    ):
        """
        Args:
            g_feat:     [B, gw_dim]     — GW global embedding
            h_l:        [B, N, opt_dim]  — optical time-series features
            z_l:        [B, opt_dim]     — optical CLS token (dual mode only)
            H_gw:       [B, M, gw_dim]  — GW spatial feature map (dual mode only)
            cred_level: [B, 1]           — per-pair credible level (dual mode only)
        Returns:
            logits: [B, 2]
            fused:  [B, cls_input_dim]
        """
        d = self.attn_dim
        scale = d ** -0.5
        aux = {}

        if self.fusion_mode == "legacy_g2o":
            q1 = self.g2o_q(g_feat).unsqueeze(1)
            k1 = self.g2o_k(h_l)
            v1 = self.g2o_v(h_l)
            fused_opt = self.g2o_norm(
                (torch.softmax(q1 @ k1.transpose(1, 2) * scale, dim=-1) @ v1).squeeze(1)
            )
            combined = fused_opt
        elif self.fusion_mode == "legacy_dual":
            q1 = self.g2o_q(g_feat).unsqueeze(1)
            k1 = self.g2o_k(h_l)
            v1 = self.g2o_v(h_l)
            fused_opt = self.g2o_norm(
                (torch.softmax(q1 @ k1.transpose(1, 2) * scale, dim=-1) @ v1).squeeze(1)
            )
            q2 = self.o2g_q(z_l).unsqueeze(1)
            k2 = self.o2g_k(H_gw)
            v2 = self.o2g_v(H_gw)
            fused_gw = self.o2g_norm(
                (torch.softmax(q2 @ k2.transpose(1, 2) * scale, dim=-1) @ v2).squeeze(1)
            )
            if cred_level is None:
                cred_level = torch.zeros((g_feat.size(0), 1), dtype=g_feat.dtype, device=g_feat.device)
            combined = torch.cat([fused_opt, fused_gw, cred_level], dim=-1)
        else:
            if g_param is None or coord_feat is None:
                raise ValueError("physical_dual_hgw requires g_param and coord_feat.")
            if H_gw is None:
                raise ValueError("physical_dual_hgw requires H_gw.")

            q1 = self.param2opt_q(g_param).unsqueeze(1)
            k1 = self.param2opt_k(h_l)
            v1 = self.param2opt_v(h_l)
            fused_opt = self.param2opt_norm(
                (torch.softmax(q1 @ k1.transpose(1, 2) * scale, dim=-1) @ v1).squeeze(1)
            )

            q2 = self.coord2gw_q(coord_feat).unsqueeze(1)
            k2 = self.coord2gw_k(H_gw)
            v2 = self.coord2gw_v(H_gw)
            scores = (q2 @ k2.transpose(1, 2)) * scale
            attn = torch.softmax(scores, dim=-1)
            fused_gw = self.coord2gw_norm((attn @ v2).squeeze(1))

            pieces = [fused_opt, fused_gw]
            if self.use_similarity_as_cls_input:
                if sim_itc_pair is None:
                    sim_itc_pair = torch.zeros((g_feat.size(0), 1), dtype=g_feat.dtype, device=g_feat.device)
                pieces.append(sim_itc_pair)
            if self.use_cred_level_feature:
                if cred_level is None:
                    cred_level = torch.zeros((g_feat.size(0), 1), dtype=g_feat.dtype, device=g_feat.device)
                pieces.append(cred_level)
            combined = torch.cat(pieces, dim=-1)
            aux["max_attention"] = attn.squeeze(1).max(dim=-1).values

        logits = self.classifier(combined)
        return logits, combined, aux


class GWOpticalALBEFModel(nn.Module):
    """
    Joint model for alignment (contrastive) and fusion (classification).
    """
    def __init__(
        self,
        gw_scalar_dim=7,
        gw_skymap_channels=7,
        optical_input_dim=6,
        ref_time_dim=64,
        enc_dim=128,
        proj_dim=256,
        fusion_attn_dim=None,
        fusion_hidden_dim=None,
        temp_init=0.07,
        temp_min=0.01,
        temp_max=100.0,
        fusion_dropout=0.1,
        gw_dropout=0.1,
        opt_dropout=0.1,
        proj_dropout=0.0,
        feature_dropout=0.0,
        label_smoothing=0.0,
        itc_label_smoothing=0.0,
        use_lightweight_gw=False,
        dual_fusion=False,
        fusion_mode=None,
        use_cred_level_feature=False,
        use_similarity_as_cls_input=False,
        time_compat_weight=0.6,
        time_compat_tau_days=30.0,
        time_compat_power=2.0,
        time_compat_max_penalty=8.0,
    ):
        super().__init__()
        self.fusion_mode = normalize_fusion_mode(fusion_mode, dual_fusion=dual_fusion)
        self.dual_fusion = self.fusion_mode != "legacy_g2o"
        self.use_cred_level_feature = bool(use_cred_level_feature)
        self.use_similarity_as_cls_input = bool(use_similarity_as_cls_input)

        if self.fusion_mode == "physical_dual_hgw":
            self.gw_encoder = PhysicalDualGWEncoder(
                scalar_input_dim=gw_scalar_dim,
                skymap_channels=gw_skymap_channels,
                enc_dim=enc_dim,
                proj_dim=proj_dim,
                gw_dropout=gw_dropout,
                proj_dropout=proj_dropout,
                use_lightweight=use_lightweight_gw,
            )
            self.optical_encoder = PhysicalDualOpticalEncoder(
                input_dim=optical_input_dim,
                ref_time_dim=ref_time_dim,
                enc_dim=enc_dim,
                num_heads=4,
                k_dim=64,
                curve_dropout=opt_dropout,
                proj_dim=proj_dim,
                proj_dropout=proj_dropout,
            )
            self.gw_proj = None
            self.opt_proj = None
        else:
            # 根据参数选择GW编码器类型
            if use_lightweight_gw:
                self.gw_encoder = LightweightGWEncoder(
                    scalar_input_dim=gw_scalar_dim,
                    skymap_channels=gw_skymap_channels,
                    final_output_dim=enc_dim,
                    dropout=gw_dropout,
                    need_param_head=False,
                )
            else:
                self.gw_encoder = GWMOCResNetEncoder(
                    scalar_input_dim=gw_scalar_dim,
                    skymap_channels=gw_skymap_channels,
                    final_output_dim=enc_dim,
                    dropout=gw_dropout,
                    need_param_head=False,
                )
            self.optical_encoder = OpticalEncoderWithCLS(
                input_dim=optical_input_dim,
                output_dim=enc_dim,
                num_heads=4,
                ref_dim=ref_time_dim,
                dropout=opt_dropout
            )
            self.gw_proj = ProjectionHead(enc_dim, enc_dim, proj_dim, dropout=proj_dropout)
            self.opt_proj = ProjectionHead(enc_dim, enc_dim, proj_dim, dropout=proj_dropout)
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.itc_label_smoothing = float(itc_label_smoothing)
        self.log_temp = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(temp_init)))
        self.temp_min = float(temp_min)
        self.temp_max = float(temp_max)
        self.time_compat_weight = float(time_compat_weight)
        self.time_compat_tau_days = float(time_compat_tau_days)
        self.time_compat_power = float(time_compat_power)
        self.time_compat_max_penalty = float(time_compat_max_penalty)
        self.itc_criterion = nn.CrossEntropyLoss(label_smoothing=itc_label_smoothing)

        self.fusion = CrossAttentionFusion(
            gw_dim=enc_dim,
            opt_dim=enc_dim,
            coord_dim=enc_dim,
            attn_dim=fusion_attn_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
            dual=self.dual_fusion,
            fusion_mode=self.fusion_mode,
            use_cred_level_feature=use_cred_level_feature,
            use_similarity_as_cls_input=use_similarity_as_cls_input,
        )
        self.cls_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def encode(self, gw_s, gw_m, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err):
        g, H_gw = self.gw_encoder(gw_s, gw_m)  # g: [B, enc_dim], H_gw: [B, SeqLen, enc_dim]
        if self.feature_dropout.p > 0:
            g = self.feature_dropout(g)
        z_l, h_l = self.optical_encoder(
            opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, errors_obs=opt_err
        )
        if self.feature_dropout.p > 0:
            z_l = self.feature_dropout(z_l)
            h_l = self.feature_dropout(h_l)
        return g, z_l, h_l, H_gw

    def encode_optical(self, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err):
        z_l, h_l = self.optical_encoder(
            opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, errors_obs=opt_err
        )
        if self.feature_dropout.p > 0:
            z_l = self.feature_dropout(z_l)
            h_l = self.feature_dropout(h_l)
        return z_l, h_l

    def uses_cred_level_input(self):
        return bool(self.fusion.uses_cred_level_input())

    def encode_coord_query(self, opt_coords):
        if self.fusion_mode == "physical_dual_hgw":
            coord_feat = self.optical_encoder.encode_coord_only(opt_coords)
        else:
            coord_feat = self.optical_encoder.encode_coordinates(opt_coords, return_xyz=False)
        if self.feature_dropout.p > 0:
            coord_feat = self.feature_dropout(coord_feat)
        return coord_feat

    def encode_gw_param_query(self, gw_s):
        if not hasattr(self.gw_encoder, "encode_scalar_only"):
            raise AttributeError("GW encoder does not support scalar-only encoding.")
        g_param = self.gw_encoder.encode_scalar_only(gw_s)
        if self.feature_dropout.p > 0:
            g_param = self.feature_dropout(g_param)
        return g_param

    def project_gw_features(self, g):
        if self.fusion_mode == "physical_dual_hgw":
            return F.normalize(g, p=2, dim=1, eps=1e-8)
        return F.normalize(self.gw_proj(g), p=2, dim=1, eps=1e-8)

    def project_optical_features(self, z_l):
        if self.fusion_mode == "physical_dual_hgw":
            return F.normalize(z_l, p=2, dim=1, eps=1e-8)
        return F.normalize(self.opt_proj(z_l), p=2, dim=1, eps=1e-8)

    def get_contrastive_embeddings(self, g, z_l):
        return self.project_gw_features(g), self.project_optical_features(z_l)

    def _build_time_compat_bias(self, gw_event_time_mjd=None, opt_event_time_mjd=None):
        """
        Build pairwise time-compatibility bias matrix [B, B] for GW->Optical logits.
        bias_ij = -w * (|t_opt_j - t_gw_i| / tau)^power, clamped to [-max_penalty, 0].

        Invalid time pairs (NaN/Inf) receive zero bias.
        """
        if gw_event_time_mjd is None:
            return None
        if self.time_compat_weight <= 0 or self.time_compat_tau_days <= 0:
            return None

        gw_t = gw_event_time_mjd.reshape(-1).to(torch.float32)
        opt_t = gw_t if opt_event_time_mjd is None else opt_event_time_mjd.reshape(-1).to(torch.float32)

        valid = torch.isfinite(gw_t).unsqueeze(1) & torch.isfinite(opt_t).unsqueeze(0)
        delta = opt_t.unsqueeze(0) - gw_t.unsqueeze(1)
        bias = -self.time_compat_weight * torch.pow(
            torch.abs(delta) / self.time_compat_tau_days,
            self.time_compat_power,
        )

        if self.time_compat_max_penalty > 0:
            bias = torch.clamp(bias, min=-self.time_compat_max_penalty, max=0.0)
        else:
            bias = torch.clamp_max(bias, 0.0)

        bias = torch.where(valid, bias, torch.zeros_like(bias))
        return bias

    def compute_itc_loss(
        self,
        g,
        z_l,
        gw_indices=None,
        mask=False,
        gw_event_time_mjd=None,
        opt_event_time_mjd=None,
    ):
        feat_g, feat_o = self.get_contrastive_embeddings(g, z_l)

        temperature = torch.clamp(self.log_temp.exp(), min=self.temp_min, max=self.temp_max)
        sim_g2o = torch.matmul(feat_g, feat_o.T) / temperature
        sim_o2g = sim_g2o.T
        time_bias = self._build_time_compat_bias(gw_event_time_mjd, opt_event_time_mjd)
        if time_bias is not None:
            time_bias = time_bias.to(device=sim_g2o.device, dtype=sim_g2o.dtype)
            sim_g2o = sim_g2o + time_bias
            sim_o2g = sim_o2g + time_bias.T

        if mask and gw_indices is not None:
            labels_mask = (gw_indices.unsqueeze(0) == gw_indices.unsqueeze(1)).float()
            batch_size = feat_g.size(0)
            target = labels_mask / labels_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            if self.itc_label_smoothing > 0:
                smooth = self.itc_label_smoothing
                target = target * (1.0 - smooth) + smooth / float(batch_size)

            log_prob_g2o = F.log_softmax(sim_g2o, dim=1)
            log_prob_o2g = F.log_softmax(sim_o2g, dim=1)
            loss_g = -(target * log_prob_g2o).sum(dim=1).mean()
            loss_o = -(target * log_prob_o2g).sum(dim=1).mean()
            total_loss = (loss_g + loss_o) / 2.0
        else:
            batch_size = feat_g.size(0)
            labels = torch.arange(batch_size, device=feat_g.device)
            loss_g = self.itc_criterion(sim_g2o, labels)
            loss_o = self.itc_criterion(sim_o2g, labels)
            total_loss = (loss_g + loss_o) / 2

        return total_loss, sim_g2o

    def compute_supcon_loss(
        self,
        g,
        z_l,
        gw_indices,
        margin=0.0,
        gw_event_time_mjd=None,
        opt_event_time_mjd=None,
    ):
        """
        Supervised Contrastive Loss for many-to-many GW-optical matching.

        Reference: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)

        Key difference from InfoNCE:
        - Treats all samples with same gw_index as positives
        - Normalizes loss by number of positives per anchor
        - More stable with multiple positives per class

        Uses the model's learned/scheduled temperature (self.log_temp) so that
        temperature scheduling and learned temperature mode work consistently.

        Args:
            g: GW embeddings [batch, dim]
            z_l: Optical embeddings [batch, dim]
            gw_indices: GW event indices for each sample
            margin: Margin to enforce between positive and negative similarities.
                    When margin > 0, negatives are penalized by subtracting margin
                    from their similarity, pushing positives to be more similar
                    than negatives by at least this margin.
        """
        feat_g, feat_o = self.get_contrastive_embeddings(g, z_l)

        batch_size = feat_g.size(0)
        device = feat_g.device

        temperature = torch.clamp(self.log_temp.exp(), min=self.temp_min, max=self.temp_max)

        features = torch.cat([feat_g, feat_o], dim=0)
        labels = torch.cat([gw_indices, gw_indices], dim=0)

        sim_matrix = torch.matmul(features, features.T) / temperature
        time_bias = self._build_time_compat_bias(gw_event_time_mjd, opt_event_time_mjd)
        if time_bias is not None:
            time_bias = time_bias.to(device=sim_matrix.device, dtype=sim_matrix.dtype)
            # Only apply to cross-modal blocks (GW->Optical and Optical->GW)
            sim_matrix[:batch_size, batch_size:] = sim_matrix[:batch_size, batch_size:] + time_bias
            sim_matrix[batch_size:, :batch_size] = sim_matrix[batch_size:, :batch_size] + time_bias.T

        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_pos = labels_eq.clone()
        mask_pos[:batch_size, :batch_size] = False
        mask_pos[batch_size:, batch_size:] = False
        mask_pos.fill_diagonal_(False)
        mask_pos = mask_pos.float()
        mask_neg = 1.0 - labels_eq.float()

        mask_self = torch.eye(2 * batch_size, device=device, dtype=torch.bool)

        # Apply margin: subtract margin from negative similarities
        # This encourages positives to be more similar than negatives by margin
        if margin > 0:
            margin_matrix = margin * mask_neg / temperature
            sim_matrix = sim_matrix - margin_matrix

        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        exp_logits = torch.exp(logits)
        exp_logits = exp_logits.masked_fill(mask_self, 0)
        log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        log_prob = logits - log_sum_exp

        num_positives = mask_pos.sum(dim=1).clamp_min(1)
        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / num_positives

        # Only average over anchors that have at least one positive
        has_pos = mask_pos.sum(dim=1) > 0
        pos_rate = has_pos.float().mean().item()
        # print(f"Supervised Contrastive Loss - Positive Rate: {pos_rate*100:.2f}%")
        if has_pos.any():
            loss = -mean_log_prob_pos[has_pos].mean()
        else:
            loss = torch.zeros((), device=device)

        sim_g2o = torch.matmul(feat_g, feat_o.T) / temperature
        if time_bias is not None:
            sim_g2o = sim_g2o + time_bias

        return loss, sim_g2o

    def fusion_logits(
        self,
        g_feat,
        h_l,
        z_l=None,
        H_gw=None,
        cred_level=None,
        gw_s=None,
        gw_m=None,
        opt_coords=None,
    ):
        fusion_kwargs = {
            "z_l": z_l,
            "H_gw": H_gw,
            "cred_level": cred_level,
        }
        if self.fusion_mode == "physical_dual_hgw":
            if gw_s is None or opt_coords is None:
                raise ValueError("physical_dual_hgw fusion requires gw_s and opt_coords.")
            g_param = self.encode_gw_param_query(gw_s)
            coord_feat = self.encode_coord_query(opt_coords)
            sim_itc_pair = None
            if self.use_similarity_as_cls_input:
                feat_g, feat_o = self.get_contrastive_embeddings(g_feat, z_l)
                sim_itc_pair = (feat_g * feat_o).sum(dim=-1, keepdim=True)
            fusion_kwargs.update(
                {
                    "g_param": g_param,
                    "coord_feat": coord_feat,
                    "sim_itc_pair": sim_itc_pair,
                }
            )
        logits, _, _ = self.fusion(g_feat, h_l, **fusion_kwargs)
        return logits

    @staticmethod
    def sample_hard_negatives(sim_g2o, gw_indices=None):
        sim = sim_g2o.detach()
        batch_size = sim.size(0)
        if gw_indices is not None:
            same_event = gw_indices.unsqueeze(0) == gw_indices.unsqueeze(1)
        else:
            same_event = torch.eye(batch_size, device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(same_event, -1e9)
        hard_idx = sim.argmax(dim=1)
        return hard_idx

    @staticmethod
    def sample_semi_hard_negatives(sim_g2o, gw_indices=None, margin=0.2):
        """
        Sample semi-hard negatives using relative margin around positive median.

        For each anchor, compute the median similarity among its positive
        (same-event) pairs, then select negatives from the band:
            sim in [(1 - margin) * pos_median, pos_median]

        Fallback: if no candidates in the band, pick the negative with
        similarity closest to (but below) pos_median. If all negatives
        exceed pos_median, fall back to a random negative.

        Args:
            sim_g2o: Similarity matrix [batch, batch]
            gw_indices: GW event indices to identify same-event pairs
            margin: Relative margin in (0, 1). Band width = margin * pos_median.

        Returns:
            semi_hard_idx: Selected negative indices [batch]
        """
        sim = sim_g2o.detach()
        batch_size = sim.size(0)
        device = sim.device

        if gw_indices is not None:
            same_event = gw_indices.unsqueeze(0) == gw_indices.unsqueeze(1)
        else:
            same_event = torch.eye(batch_size, device=device, dtype=torch.bool)

        # Median positive similarity per anchor
        pos_sim = sim.clone()
        pos_sim.masked_fill_(~same_event, float('nan'))
        pos_med = pos_sim.nanmedian(dim=1).values  # [batch]

        # Mask positives out of negative candidates
        neg_sim = sim.masked_fill(same_event, -1e9)

        # Semi-hard band: [(1-margin)*pos_med, pos_med]
        lower = ((1.0 - margin) * pos_med).unsqueeze(1)  # [batch, 1]
        upper = pos_med.unsqueeze(1)                       # [batch, 1]
        semi_hard_mask = (neg_sim >= lower) & (neg_sim <= upper)

        result = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i in range(batch_size):
            candidates = semi_hard_mask[i].nonzero(as_tuple=False).squeeze(-1)
            if candidates.numel() > 0:
                pick = torch.randint(0, candidates.numel(), (1,), device=device)
                result[i] = candidates[pick]
            else:
                # Fallback: closest negative below pos_med
                below_mask = (neg_sim[i] <= pos_med[i]) & (neg_sim[i] > -1e8)
                below_idx = below_mask.nonzero(as_tuple=False).squeeze(-1)
                if below_idx.numel() > 0:
                    best = neg_sim[i, below_idx].argmax()
                    result[i] = below_idx[best]
                else:
                    # All negatives exceed pos_med — pick random negative
                    neg_idx = (~same_event[i]).nonzero(as_tuple=False).squeeze(-1)
                    pick = torch.randint(0, neg_idx.numel(), (1,), device=device)
                    result[i] = neg_idx[pick]

        return result

# ==============================================================================
# 9. Lightweight GW Encoder (解决过拟合问题)
# ==============================================================================
class LightweightSkymapEncoder(nn.Module):
    """
    轻量级1D CNN编码器，专门设计用于小样本GW数据集。
    参数量：~100K（相比ResNet-18的~11M）

    设计原则：
    - 减少参数量以匹配有限的GW训练样本（~400个）
    - 使用更强的dropout正则化
    - 保持足够的表达能力提取skymap特征
    """
    def __init__(self, in_channels=7, output_dim=128, dropout=0.5):
        super().__init__()
        # Conv blocks (without pooling) to preserve spatial feature map
        self.conv_blocks = nn.Sequential(
            # Block 1: [B, 7, 19200] -> [B, 32, 4800]
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Block 2: [B, 32, 4800] -> [B, 64, 1200]
            nn.Conv1d(32, 64, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Block 3: [B, 64, 1200] -> [B, 128, 300]
            nn.Conv1d(64, 128, kernel_size=3, stride=4, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Separate pooling for flexibility
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [Batch, in_channels, Length]
        x = self.conv_blocks(x)        # [Batch, 128, 300]
        feature_map = x                # preserve spatial features
        pooled = self.pool(x).squeeze(-1)  # [Batch, 128]
        return self.fc(pooled), feature_map  # (pooled_output, spatial_features)


class LightweightGWEncoder(nn.Module):
    """
    轻量级GW双流编码器，替代原GWMOCResNetEncoder。

    总参数量：~100K（相比原~11M）
    适用于小样本GW数据集（<500个唯一事件）
    """
    def __init__(self,
                 scalar_input_dim=7,
                 skymap_channels=7,
                 scalar_hidden_dim=128,
                 final_output_dim=128,
                 dropout=0.5,
                 need_param_head=False):
        super().__init__()

        # 简化的Scalar编码器
        self.scalar_enc = nn.Sequential(
            nn.Linear(scalar_input_dim, scalar_hidden_dim),
            nn.BatchNorm1d(scalar_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(scalar_hidden_dim, scalar_hidden_dim),
            nn.BatchNorm1d(scalar_hidden_dim),
            nn.ReLU()
        )
        self.scalar_head = None
        if need_param_head:
            self.scalar_head = nn.Sequential(
                nn.Linear(scalar_hidden_dim, final_output_dim),
                nn.BatchNorm1d(final_output_dim),
                nn.ReLU(),
            )

        # 轻量级Skymap编码器
        self.skymap_enc = LightweightSkymapEncoder(
            in_channels=skymap_channels,
            output_dim=scalar_hidden_dim,
            dropout=dropout
        )

        # 融合层
        self.fusion_head = nn.Sequential(
            nn.Linear(scalar_hidden_dim * 2, final_output_dim),
            nn.BatchNorm1d(final_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(final_output_dim, final_output_dim)
        )

        # Skymap sequence projection for dual fusion: 128 -> final_output_dim
        # The conv_blocks output 128-dim features, but enc_dim may differ
        self.skymap_seq_proj = nn.Sequential(
            nn.Conv1d(128, final_output_dim, kernel_size=1),
            nn.BatchNorm1d(final_output_dim),
            nn.ReLU()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.Conv1d)):
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def encode_scalar_only(self, gw_scalars):
        h_scalar = self.scalar_enc(gw_scalars)
        if self.scalar_head is None:
            return h_scalar
        return self.scalar_head(h_scalar)

    def encode_features(self, gw_scalars, skymap_sequence):
        """
        Args:
            gw_scalars: [Batch, D_scalar]
            skymap_sequence: [Batch, skymap_channels, Length]
        Returns:
            g: [Batch, final_output_dim]
            g_param: [Batch, final_output_dim] or [Batch, scalar_hidden_dim]
            H_gw: [Batch, SeqLen, final_output_dim] — spatial feature map for dual fusion
        """
        h_scalar = self.scalar_enc(gw_scalars)  # [B, scalar_hidden]
        g_param = self.scalar_head(h_scalar) if self.scalar_head is not None else h_scalar
        h_skymap, skymap_feat = self.skymap_enc(skymap_sequence)  # h_skymap: [B, scalar_hidden], skymap_feat: [B, 128, 300]
        # Project spatial feature map: [B, 128, 300] -> [B, final_output_dim, 300] -> [B, 300, final_output_dim]
        H_gw = self.skymap_seq_proj(skymap_feat).permute(0, 2, 1)  # [B, 300, final_output_dim]
        combined = torch.cat([h_scalar, h_skymap], dim=1)  # [B, scalar_hidden * 2]
        g = self.fusion_head(combined)  # [B, final_output_dim]
        return g, g_param, H_gw

    def forward(self, gw_scalars, skymap_sequence):
        g, _g_param, H_gw = self.encode_features(gw_scalars, skymap_sequence)
        return g, H_gw


class OpticalKNClassifier(nn.Module):
    """
    Optical-only KN classifier built from the existing optical encoder.
    """
    def __init__(
        self,
        optical_input_dim=6,
        ref_time_dim=64,
        enc_dim=128,
        num_heads=4,
        k_dim=64,
        opt_dropout=0.1,
        feature_dropout=0.0,
        head_hidden_dim=None,
        head_dropout=0.2,
        universal_aux_enable=False,
        proj_dim=64,
        adv_hidden_dim=None,
        n_det_bucket_classes=5,
        n_bands_bucket_classes=4,
        t_span_bucket_classes=5,
        grl_lambda=1.0,
    ):
        super().__init__()
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.universal_aux_enable = bool(universal_aux_enable)
        self.optical_encoder_init_info = None

        self.optical_encoder = OpticalEncoderWithCLSNoCoord(
            input_dim=optical_input_dim,
            output_dim=enc_dim,
            num_heads=num_heads,
            ref_dim=ref_time_dim,
            k_dim=k_dim,
            dropout=opt_dropout,
        )

        hidden = head_hidden_dim if head_hidden_dim is not None else enc_dim
        self.feature_dim = enc_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(hidden, 1),
        )
        self.proj_dim = int(proj_dim)
        self.adv_hidden_dim = int(hidden if adv_hidden_dim is None else adv_hidden_dim)
        self.n_det_bucket_classes = int(n_det_bucket_classes)
        self.n_bands_bucket_classes = int(n_bands_bucket_classes)
        self.t_span_bucket_classes = int(t_span_bucket_classes)
        self.grl_lambda = float(grl_lambda)
        if self.universal_aux_enable:
            self.projection_head = _build_mlp(
                self.feature_dim,
                self.adv_hidden_dim,
                self.proj_dim,
                dropout=head_dropout,
                final_bias=True,
            )
            self.grl = GradientReversalLayer(lambda_=self.grl_lambda)
            self.adv_head_n_det = _build_mlp(
                self.feature_dim,
                self.adv_hidden_dim,
                self.n_det_bucket_classes,
                dropout=head_dropout,
            )
            self.adv_head_n_bands = _build_mlp(
                self.feature_dim,
                self.adv_hidden_dim,
                self.n_bands_bucket_classes,
                dropout=head_dropout,
            )
            self.adv_head_t_span = _build_mlp(
                self.feature_dim,
                self.adv_hidden_dim,
                self.t_span_bucket_classes,
                dropout=head_dropout,
            )
        self._init_head_weights()

    def _init_head_weights(self):
        modules = [self.classifier]
        if self.universal_aux_enable:
            modules.extend(
                [
                    self.projection_head,
                    self.adv_head_n_det,
                    self.adv_head_n_bands,
                    self.adv_head_t_span,
                ]
            )
        for module in modules:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def encode_optical(self, opt_t, opt_v, opt_ref_t, opt_mask, opt_err):
        z_l, h_l = self.optical_encoder(
            opt_t, opt_v, opt_ref_t, opt_mask, errors_obs=opt_err
        )
        if self.feature_dropout.p > 0:
            z_l = self.feature_dropout(z_l)
            h_l = self.feature_dropout(h_l)
        return z_l, h_l

    def compute_joint_features(self, opt_t, opt_v, opt_ref_t, opt_mask, opt_err):
        z_l, h_l = self.encode_optical(opt_t, opt_v, opt_ref_t, opt_mask, opt_err)
        h_pool = h_l.mean(dim=1)
        feat = torch.cat([z_l, h_pool], dim=1)
        return feat, z_l, h_l

    def forward(self, opt_t, opt_v, opt_ref_t, opt_mask, opt_err):
        feat, _, _ = self.compute_joint_features(opt_t, opt_v, opt_ref_t, opt_mask, opt_err)
        logits = self.classifier(feat)
        return logits

    def forward_with_aux(self, opt_t, opt_v, opt_ref_t, opt_mask, opt_err, return_aux=True):
        feat, z_l, h_l = self.compute_joint_features(opt_t, opt_v, opt_ref_t, opt_mask, opt_err)
        logits = self.classifier(feat)
        if not return_aux or (not self.universal_aux_enable):
            return {
                "logits": logits,
                "joint_feat": feat,
                "proj_feat": None,
                "adv_logits_n_det": None,
                "adv_logits_n_bands": None,
                "adv_logits_t_span": None,
                "cls_feat": z_l,
                "temporal_feat": h_l,
            }
        proj_feat = F.normalize(self.projection_head(feat), dim=-1)
        feat_adv = self.grl(feat)
        return {
            "logits": logits,
            "joint_feat": feat,
            "proj_feat": proj_feat,
            "adv_logits_n_det": self.adv_head_n_det(feat_adv),
            "adv_logits_n_bands": self.adv_head_n_bands(feat_adv),
            "adv_logits_t_span": self.adv_head_t_span(feat_adv),
            "cls_feat": z_l,
            "temporal_feat": h_l,
        }

    def set_grl_lambda(self, value: float):
        self.grl_lambda = float(value)
        if self.universal_aux_enable:
            self.grl.set_lambda(value)

    def set_encoder_trainable(self, trainable):
        for p in self.optical_encoder.parameters():
            p.requires_grad = bool(trainable)

    def load_optical_encoder_from_albef_state_dict(self, state_dict, strict=False):
        """
        Load only optical encoder weights from an ALBEF checkpoint state_dict.
        """
        if not isinstance(state_dict, dict):
            raise TypeError("state_dict must be a dict.")

        cleaned = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        prefix = "optical_encoder."
        optical_state = {
            k[len(prefix):]: v for k, v in cleaned.items() if k.startswith(prefix)
        }
        if not optical_state:
            raise KeyError("No optical_encoder.* keys found in provided state_dict.")
        source_mode = "full_optical_encoder"
        if any(k.startswith("curve_encoder.") for k in optical_state):
            curve_prefix = "curve_encoder."
            optical_state = {
                k[len(curve_prefix):]: v
                for k, v in optical_state.items()
                if k.startswith(curve_prefix)
            }
            if not optical_state:
                raise KeyError("No optical_encoder.curve_encoder.* keys found in provided state_dict.")
            source_mode = "curve_encoder_only"

        target_state = self.optical_encoder.state_dict()
        matched_keys = []
        shape_mismatch = []
        for key, value in optical_state.items():
            if key not in target_state:
                continue
            if tuple(target_state[key].shape) != tuple(value.shape):
                shape_mismatch.append(
                    {
                        "key": key,
                        "expected": tuple(target_state[key].shape),
                        "got": tuple(value.shape),
                    }
                )
                continue
            matched_keys.append(key)

        if not matched_keys:
            raise KeyError(
                "No compatible optical curve-encoder keys matched the optical-only backbone. "
                f"source_mode={source_mode}"
            )
        if shape_mismatch:
            preview = ", ".join(
                f"{entry['key']} expected={entry['expected']} got={entry['got']}"
                for entry in shape_mismatch[:5]
            )
            raise RuntimeError(
                "Optical encoder initialization aborted due to shape mismatch: "
                f"{preview}"
            )

        missing, unexpected = self.optical_encoder.load_state_dict(optical_state, strict=False)
        self.optical_encoder_init_info = {
            "source_mode": str(source_mode),
            "matched_key_count": int(len(matched_keys)),
            "missing_key_count": int(len(missing)),
            "unexpected_key_count": int(len(unexpected)),
            "matched_keys_preview": matched_keys[:10],
            "missing_keys_preview": list(missing[:10]),
            "unexpected_keys_preview": list(unexpected[:10]),
        }
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"Optical encoder load strict check failed. Missing={missing}, Unexpected={unexpected}"
            )
        return missing, unexpected
