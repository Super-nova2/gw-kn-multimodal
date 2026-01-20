import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

    def forward(self, coordinates):
        # coordinates: [Batch, 2] (RA in degrees, Dec in degrees)
        # convert (RA, Dec) in degrees to radians
        ra = coordinates[:, 0]   # [Batch]
        dec = coordinates[:, 1]  # [Batch]
        theta = (90.0 - dec) * (math.pi / 180.0)  # polar angle
        phi = ra * (math.pi / 180.0)               # azimuthal angle

        # Convert Spherical to Cartesian Coordinates
        x = torch.cos(theta) * torch.cos(phi)
        y = torch.cos(theta) * torch.sin(phi)
        z = torch.sin(theta)
        
        coords = torch.stack([x, y, z], dim=1) # [Batch, 3]
        return self.fc(coords)

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
        spatial_feat = self.spatial_embedding(ra_dec_obs)   # [Batch, ref_dim]
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

        x = self.avgpool(x) # [Batch, 512, 1]
        x = x.flatten(1)    # [Batch, 512]
        x = self.fc(x)      # [Batch, Output_Dim]
        
        return x

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
                 dropout=0.1):
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
    

    def forward(self, gw_scalars, skymap_sequence):
        """
        Args:
            gw_scalars: [Batch, D_scalar]
            skymap_sequence: [Batch, skymap_channels, Length] 
                             (e.g., [B, 7, 19200])
        Returns:
            g: [Batch, Final_Dim]
        """
        # Encode Scalars
        h_scalar = self.scalar_enc(gw_scalars)  # [B, scalar_hidden]
        
        # Encode Skymap Sequence
        h_skymap = self.skymap_enc(skymap_sequence) # [B, resnet_output]
        
        # Concatenate
        combined = torch.cat([h_scalar, h_skymap], dim=1)   # [B, fusion_input_dim]
        
        # Fuse
        g = self.fusion_head(combined)  # [B, final_output_dim]
        
        return g

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
        # g: [Batch, enc_dim]
        g = self.gw_encoder(gw_s, gw_m)
        
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
        logit_scale = torch.clamp(self.log_temp.exp(), min=0.01, max=100.0)
        
        # 1. Similarity Matrix: [B, B]
        sim_g2o = torch.matmul(feat_g, feat_o.T) * logit_scale
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
    Cross-attention fusion head for GW (query) and optical sequence (key/value).
    Outputs binary classification logits for matching.
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
        """
        Args:
            g_feat: [Batch, gw_dim]
            h_l: [Batch, N, opt_dim]
        Returns:
            logits: [Batch, 2]
            fused: [Batch, attn_dim]
        """
        q = self.q_proj(g_feat).unsqueeze(1)  # [B, 1, d]
        k = self.k_proj(h_l)                 # [B, N, d]
        v = self.v_proj(h_l)                 # [B, N, d]

        attn_scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.attn_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        fused = torch.matmul(attn_weights, v).squeeze(1)
        fused = self.out_norm(fused)

        logits = self.classifier(fused)
        return logits, fused


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
        use_lightweight_gw=False
    ):
        super().__init__()

        # 根据参数选择GW编码器类型
        if use_lightweight_gw:
            # 轻量级编码器：~100K参数，适用于小样本GW数据集
            self.gw_encoder = LightweightGWEncoder(
                scalar_input_dim=gw_scalar_dim,
                skymap_channels=gw_skymap_channels,
                final_output_dim=enc_dim,
                dropout=gw_dropout
            )
        else:
            # 原始ResNet编码器：~11M参数
            self.gw_encoder = GWMOCResNetEncoder(
                scalar_input_dim=gw_scalar_dim,
                skymap_channels=gw_skymap_channels,
                final_output_dim=enc_dim,
                dropout=gw_dropout
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
        self.itc_criterion = nn.CrossEntropyLoss(label_smoothing=itc_label_smoothing)

        self.fusion = CrossAttentionFusion(
            gw_dim=enc_dim,
            opt_dim=enc_dim,
            attn_dim=fusion_attn_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout
        )
        self.cls_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def encode(self, gw_s, gw_m, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err):
        g = self.gw_encoder(gw_s, gw_m)
        if self.feature_dropout.p > 0:
            g = self.feature_dropout(g)
        z_l, h_l = self.optical_encoder(
            opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, errors_obs=opt_err
        )
        if self.feature_dropout.p > 0:
            z_l = self.feature_dropout(z_l)
            h_l = self.feature_dropout(h_l)
        return g, z_l, h_l

    def encode_optical(self, opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, opt_err):
        z_l, h_l = self.optical_encoder(
            opt_coords, opt_t, opt_v, opt_ref_t, opt_mask, errors_obs=opt_err
        )
        if self.feature_dropout.p > 0:
            z_l = self.feature_dropout(z_l)
            h_l = self.feature_dropout(h_l)
        return z_l, h_l

    def compute_itc_loss(self, g, z_l, gw_indices=None, mask=False):
        feat_g = F.normalize(self.gw_proj(g), p=2, dim=1, eps=1e-8)
        feat_o = F.normalize(self.opt_proj(z_l), p=2, dim=1, eps=1e-8)

        logit_scale = torch.clamp(self.log_temp.exp(), min=self.temp_min, max=self.temp_max)
        sim_g2o = torch.matmul(feat_g, feat_o.T) * logit_scale
        sim_o2g = sim_g2o.T

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

    def fusion_logits(self, g_feat, h_l):
        logits, _ = self.fusion(g_feat, h_l)
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
        self.conv = nn.Sequential(
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

            # Global pooling: [B, 128, 300] -> [B, 128, 1]
            nn.AdaptiveAvgPool1d(1)
        )
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
        x = self.conv(x).squeeze(-1)  # [Batch, 128]
        return self.fc(x)  # [Batch, output_dim]


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
                 dropout=0.5):
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

    def forward(self, gw_scalars, skymap_sequence):
        """
        Args:
            gw_scalars: [Batch, D_scalar]
            skymap_sequence: [Batch, skymap_channels, Length]
        Returns:
            g: [Batch, final_output_dim]
        """
        h_scalar = self.scalar_enc(gw_scalars)  # [B, scalar_hidden]
        h_skymap = self.skymap_enc(skymap_sequence)  # [B, scalar_hidden]
        combined = torch.cat([h_scalar, h_skymap], dim=1)  # [B, scalar_hidden * 2]
        return self.fusion_head(combined)  # [B, final_output_dim]
