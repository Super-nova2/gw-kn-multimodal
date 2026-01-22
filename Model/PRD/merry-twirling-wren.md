# Plan: Add Negative GW Events to Dataset

## Overview

**Goal**: Add BNS events without detectable kilonova (KN) as "negative GW" events. These represent real GW detections that should NOT match any optical transient.

**Use Case**: During training, pair a negative GW with a random optical sample → model should predict "no match" (cls_label=0).

## Data Summary

| Category | Count | Source |
|----------|-------|--------|
| BNS total | 1822 | `/fred/oz016/bgao_kn/ML+GW+KN/dataset/O5_sim_bns/injections_final.csv` |
| BNS with KN (positive) | 491 | Events with valid optical data in SNANA |
| BNS without KN (negative) | ~1331 | Events where simulation failed or KN too faint |
| Skymaps available | 2307 | `/fred/oz016/bgao_kn/data/bns_skymap/` |

**Key insight**: The preprocessing notebook already identifies positive vs negative events by checking if `HEAD.FITS` has data. We just need to save the negative events instead of discarding them.

---

## Implementation Plan

### Step 0: Define Split + Labeling Rules

- Split by GW event ID before sampling (train/val), so all optical samples from the same GW stay together.
- Build split-local `gw_to_lc_map` and `neg_gw_indices`, and only pair negatives with optical samples from the same split.
- Decide negative sampling fallback when `neg_gw_ratio` cannot be met (sample with replacement vs reduce ratio).
- Standardize labels: `cls_label=1` for positive pairs (parent GW), `cls_label=0` for negative GW overrides.
- Fix RNG seeds for reproducible splits and batch composition.

### Step 1: Create New Dataset with Negative GW ✅ DONE

Modified [data_preprocess.ipynb](ML+GW+KN/Model/notebook/data_preprocess.ipynb) to include negative GW events.
Added cells after existing dataset generation to create `combined_dataset_with_neg_gw.h5`.

**Key changes:**
1. Load the **full** BNS catalog (1822 events) instead of just the filtered 491
2. For each event, check if skymap exists AND if optical data exists
3. Save ALL events with skymaps to `gw_data/`, with a new `has_kn` flag
4. Only save optical data for `has_kn=1` events

**New HDF5 structure:**
```
events/
├── gw_data/
│   ├── scalars [N_gw_total, 7]      # ALL BNS with skymaps
│   ├── skymaps [N_gw_total, 7, 19200]
│   ├── ids [N_gw_total]
│   └── has_kn [N_gw_total]          # NEW: 1=has KN, 0=no KN
└── optical_data/
    ├── values [N_opt, 200, 6]       # Only from has_kn=1 events
    ├── parent_gw_idx [N_opt]        # Links to gw_data indices
    └── ...
```

### Step 2: Add Function to data_loader.py

Add new function `create_dataset_with_neg_gw()` to [data_loader.py](ML+GW+KN/Model/data_loader.py):

```python
def create_dataset_with_neg_gw(
    full_catalog_path,      # Full BNS catalog (1822 events)
    skymap_dir,             # /fred/oz016/bgao_kn/data/bns_skymap/
    fits_dir,               # SNANA SIM directory
    output_h5_path,
    max_neg_gw=None         # Optional: limit negative count for balance
):
    """Creates HDF5 with both positive and negative GW events."""
```

### Step 3: Modify RelationalHDF5Dataset

Update [data_loader.py](ML+GW+KN/Model/data_loader.py) `RelationalHDF5Dataset.__init__()`:

```python
def __init__(self, h5_path, ..., use_neg_gw=False):
    # Existing code...

    # NEW: Load negative GW data
    self.use_neg_gw = use_neg_gw
    self.neg_gw_indices = None

    if use_neg_gw:
        with h5py.File(h5_path, 'r') as f:
            if 'events/gw_data/has_kn' in f:
                has_kn = f['events/gw_data/has_kn'][:]
                self.neg_gw_indices = np.where(has_kn == 0)[0]
                if cache_in_memory:
                    self.data_cache["neg_gw_scalar"] = f['events/gw_data/scalars'][self.neg_gw_indices]
                    self.data_cache["neg_gw_skymap"] = f['events/gw_data/skymaps'][self.neg_gw_indices]
```

Add new method:
```python
def sample_neg_gw(self):
    """Sample a random negative GW event (scalar, skymap)."""
    idx = np.random.randint(len(self.neg_gw_indices))
    if self.data_cache:
        return (self.data_cache["neg_gw_scalar"][idx],
                self.data_cache["neg_gw_skymap"][idx])
    # else: load from file
```

### Step 4: Create MixedGWBatchedSampler

Add new sampler to [data_loader.py](ML+GW+KN/Model/data_loader.py):

```python
class MixedGWBatchedSampler(Sampler):
    """
    Batch sampler that includes both positive and negative GW events.

    Structure per batch (batch_size=128, neg_gw_ratio=0.2):
    - 102 positive GW-optical pairs (80%)
    - 26 negative GW paired with random optical (20%)
    """
    def __init__(
        self,
        gw_to_lc_map: dict,
        neg_gw_indices: np.ndarray,
        batch_size: int,
        steps_per_epoch: int,
        neg_gw_ratio: float = 0.2,
        samples_per_gw: int = 1
    ):
        self.gw_to_lc_map = gw_to_lc_map
        self.neg_gw_indices = neg_gw_indices
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.neg_gw_ratio = neg_gw_ratio
        self.samples_per_gw = samples_per_gw

        self.pos_gw_ids = list(gw_to_lc_map.keys())
        self.n_neg_per_batch = int(batch_size * neg_gw_ratio)
        self.n_pos_per_batch = batch_size - self.n_neg_per_batch

        # Collect all optical indices for negative pairing
        self.all_opt_indices = []
        for lcs in gw_to_lc_map.values():
            self.all_opt_indices.extend(lcs)
        self.all_opt_indices = np.array(self.all_opt_indices)

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            batch_opt_indices = []
            batch_neg_gw_indices = []  # -1 for positive, actual index for negative

            # 1. Sample positive GW-optical pairs
            batch_gw_ids = np.random.choice(
                self.pos_gw_ids, self.n_pos_per_batch, replace=False
            )
            for gw_id in batch_gw_ids:
                lcs = self.gw_to_lc_map[gw_id]
                batch_opt_indices.append(np.random.choice(lcs))
                batch_neg_gw_indices.append(-1)  # -1 = use parent GW

            # 2. Sample negative GW pairs
            neg_opt = np.random.choice(self.all_opt_indices, self.n_neg_per_batch)
            neg_gw = np.random.choice(self.neg_gw_indices, self.n_neg_per_batch)
            batch_opt_indices.extend(neg_opt.tolist())
            batch_neg_gw_indices.extend(neg_gw.tolist())

            yield batch_opt_indices, batch_neg_gw_indices

    def __len__(self):
        return self.steps_per_epoch
```

Add factory function `create_mixed_gw_dataloaders()`:
```python
def create_mixed_gw_dataloaders(
    h5_path: str,
    batch_size: int = 128,
    neg_gw_ratio: float = 0.2,
    val_split: float = 0.1,
    # ... other params
):
    """Create dataloaders with mixed positive/negative GW sampling."""
```

### Step 5: Modify Training Loop

Update [ALBEF_train.py](ML+GW+KN/Model/ALBEF_train.py):

**New arguments:**
```python
parser.add_argument("--use_neg_gw", action='store_true',
                    help="Include negative GW events (BNS without KN)")
parser.add_argument("--neg_gw_ratio", type=float, default=0.2,
                    help="Ratio of negative GW samples per batch")
```

**Modify dataloader creation (~line 380):**
```python
if args.use_neg_gw:
    train_loader, val_loader, steps_per_epoch, val_steps = create_mixed_gw_dataloaders(
        h5_path=args.data_path,
        batch_size=args.batch_size,
        neg_gw_ratio=args.neg_gw_ratio,
        val_split=args.val_split,
        # ... other params
    )
```

**Modify batch unpacking in training loop (~line 540):**
```python
# The sampler yields (opt_indices, neg_gw_indices)
# DataLoader collate needs to handle this
# For samples with neg_gw_indices[i] >= 0, use that GW instead of parent

# In collate function or __getitem__:
if neg_gw_idx >= 0:
    # Use negative GW data
    gw_scalar = neg_gw_scalars[neg_gw_idx]
    gw_skymap = neg_gw_skymaps[neg_gw_idx]
    is_match = 0  # This pair should NOT match
else:
    # Use parent GW data (existing behavior)
    is_match = 1  # This pair SHOULD match
```

**Modify CLS loss computation (~line 580):**
```python
# labels_pos now comes from the batch (1 for positive pairs, 0 for negative GW pairs)
# No change needed if collate function sets labels correctly
```

---

## Files to Modify

| File | Changes | Lines (approx) |
|------|---------|----------------|
| [data_loader.py](ML+GW+KN/Model/data_loader.py) | Add `create_dataset_with_neg_gw()`, `MixedGWBatchedSampler`, `create_mixed_gw_dataloaders()`, modify `RelationalHDF5Dataset` | +200 |
| [ALBEF_train.py](ML+GW+KN/Model/ALBEF_train.py) | Add `--use_neg_gw`, `--neg_gw_ratio` args, conditional dataloader creation | +30 |
| [ALBEF_train.sh](ML+GW+KN/Model/ALBEF_train.sh) | Add `USE_NEG_GW` and `NEG_GW_RATIO` parsing and command line flags | +15 |
| [ALBEF_supcon.json](ML+GW+KN/Model/args/ALBEF_supcon.json) | Add `use_neg_gw`, `neg_gw_ratio`, update `data_path` | +3 |

### Step 6: Modify ALBEF_train.sh

Add parsing for new parameters (~line 146):
```bash
USE_NEG_GW=$(jq -r '.use_neg_gw // false' "$args_file")
NEG_GW_RATIO=$(jq -r '.neg_gw_ratio // empty' "$args_file")
```

Add command line flags (~line 391):
```bash
if [[ "$USE_NEG_GW" == "true" ]]; then
    cmd+=(--use_neg_gw)
fi
if [[ -n "$NEG_GW_RATIO" && "$NEG_GW_RATIO" != "null" ]]; then
    cmd+=(--neg_gw_ratio "$NEG_GW_RATIO")
fi
```

### Step 7: Update ALBEF_supcon.json

Update config to use new dataset and enable negative GW:
```json
{
    "data_path": "/fred/oz016/bgao_kn/data/LSST_KN_BNS/combined_dataset_with_neg_gw.h5",
    "use_neg_gw": true,
    "neg_gw_ratio": 0.2,
    // ... rest unchanged
}
```

### Step 8: Dataset QA + Metadata

Add lightweight checks after HDF5 creation and log summary stats:
- `len(has_kn) == len(gw_data/ids)` and shapes match expected dimensions
- all `parent_gw_idx` point to `has_kn=1` entries
- report counts: `n_pos_gw`, `n_neg_gw`, `n_opt`, and the realized neg ratio
- optional: write these into a `meta/` group in the HDF5 for provenance

### Step 9: Metrics + Monitoring

Track negative/positive behavior separately in training logs:
- CLS accuracy (pos vs neg), FPR on negatives, and confusion matrix snapshots
- distribution of match scores for neg vs pos pairs to verify separation
- quick ablation sweep for `neg_gw_ratio` (e.g., 0.1/0.2/0.3)

---

## Verification

1. **Run preprocessing**:
   ```python
   from data_loader import create_dataset_with_neg_gw
   create_dataset_with_neg_gw(
       full_catalog_path="/fred/oz016/bgao_kn/ML+GW+KN/dataset/O5_sim_bns/injections_final.csv",
       skymap_dir="/fred/oz016/bgao_kn/data/bns_skymap",
       fits_dir="/fred/oz016/bgao_kn/SNANA/SNDATA_ROOT/SIM/LSST_KN_BNS",
       output_h5_path="/fred/oz016/bgao_kn/data/LSST_KN_BNS/combined_dataset_with_neg_gw.h5"
   )
   ```

2. **Verify dataset**:
   ```python
   import h5py
   with h5py.File("/fred/oz016/bgao_kn/data/LSST_KN_BNS/combined_dataset_with_neg_gw.h5", 'r') as f:
       has_kn = f['events/gw_data/has_kn'][:]
       print(f"Positive GW: {sum(has_kn == 1)}")  # Expected: 491
       print(f"Negative GW: {sum(has_kn == 0)}")  # Expected: ~1300+
   ```

3. **Submit job to server**:
   ```bash
   cd /fred/oz016/bgao_kn/ML+GW+KN/Model
   ./ALBEF_train.sh args/ALBEF_supcon.json
   ```

4. **Monitor job**:
   ```bash
   # Check job status
   squeue -u $USER

   # View logs
   tail -f logs/train/ALBEF_*.out

   # Monitor TensorBoard
   tensorboard --logdir /fred/oz016/bgao_kn/data/model/checkpoints/supcon
   ```

5. **Success criteria**:
   - CLS accuracy on negative GW pairs > 80%
   - Overall CLS accuracy improved vs baseline without negative GW

---

# Previous Plan: Supervised Contrastive Learning (SupCon)

**Selected Approach**: Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)

---

## Problem Analysis

### Current Implementation
The code in [model.py:849-877](ML+GW+KN/Model/model.py#L849-L877) already attempts to handle multiple optical samples per GW event using **soft labels**:

```python
# Current approach: Multi-positive soft labels
labels_mask = (gw_indices.unsqueeze(0) == gw_indices.unsqueeze(1)).float()
target = labels_mask / labels_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
```

This treats all optical samples from the same GW event as equally weighted positives.

### Why It's Failing
1. **Softmax saturation**: With batch_size=128 and temperature scaling, the softmax pushes all negatives to near-zero probability, making the loss insensitive to meaningful differences
2. **Equal weighting assumption**: Not all optical transients from the same GW event may be equally informative
3. **Bidirectional asymmetry**: GW→optical and optical→GW mappings have different cardinalities (1:N vs N:1)

---

## Proven Techniques for Many-to-Many Contrastive Learning

### 1. Supervised Contrastive Learning (SupCon)
**Paper**: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)

**Key idea**: Use class labels to define positive sets, with a modified loss that handles multiple positives naturally.

**Implementation**:
```python
def supcon_loss(features, labels, temperature=0.07):
    # Normalize features
    features = F.normalize(features, dim=1)
    # Similarity matrix
    sim = torch.matmul(features, features.T) / temperature
    # Mask for positive pairs (same label, excluding self)
    mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    mask.fill_diagonal_(0)
    # Log-sum-exp for denominator (all pairs except self)
    logits_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - logits_max.detach()  # numerical stability
    exp_sim = torch.exp(sim)
    exp_sim.fill_diagonal_(0)
    # Loss: -log(sum_pos / sum_all)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
    # Mean over positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    loss = -mean_log_prob_pos.mean()
    return loss
```

**Pros**: Well-established, handles multiple positives correctly
**Cons**: Requires all positives in the same batch

---

### 2. Prototype-Based Contrastive Learning
**Papers**: "Prototypical Networks" (Snell et al.), "SwAV" (Caron et al.)

**Key idea**: Instead of instance-to-instance matching, learn a **prototype** for each GW event and match optical samples to prototypes.

**Implementation approach**:
```python
# For each GW event, aggregate optical embeddings into a prototype
gw_prototypes = {}  # gw_idx -> mean optical embedding
for gw_idx in unique_gw_indices:
    mask = (gw_indices == gw_idx)
    gw_prototypes[gw_idx] = optical_features[mask].mean(dim=0)

# Contrastive loss: GW embedding vs optical prototypes
```

**Pros**: Reduces many-to-many to many-to-one, more stable training
**Cons**: Loses individual optical sample information

---

### 3. Asymmetric Contrastive Learning
**Key idea**: Use different loss formulations for GW→optical and optical→GW directions.

**For GW→optical (1:N)**: Use multi-positive loss (current implementation)
**For optical→GW (N:1)**: Use standard single-positive loss since each optical has exactly one GW parent

```python
# GW → Optical: soft multi-positive
target_g2o = labels_mask / labels_mask.sum(dim=1, keepdim=True)
loss_g2o = -(target_g2o * F.log_softmax(sim_g2o, dim=1)).sum(dim=1).mean()

# Optical → GW: single positive (diagonal only for each optical sample)
labels = torch.arange(batch_size, device=device)
loss_o2g = F.cross_entropy(sim_o2g, labels)

# Weighted combination
loss = 0.7 * loss_g2o + 0.3 * loss_o2g  # weight more towards the 1:N direction
```

---

### 4. Multi-Instance Learning (MIL) + Contrastive
**Papers**: "Attention-based MIL" (Ilse et al.)

**Key idea**: Treat each GW event as a "bag" containing multiple optical "instances". Use attention to weight instances.

**Implementation**:
```python
class MILContrastive(nn.Module):
    def __init__(self, embed_dim):
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, optical_features, gw_indices):
        # For each GW event, compute attention-weighted optical representation
        aggregated = []
        for gw_idx in unique_gw_indices:
            mask = (gw_indices == gw_idx)
            bag_features = optical_features[mask]  # [N_i, dim]
            attn_weights = F.softmax(self.attention(bag_features), dim=0)
            weighted_sum = (attn_weights * bag_features).sum(dim=0)
            aggregated.append(weighted_sum)
        return torch.stack(aggregated)  # [num_gw, dim]
```

**Pros**: Learns which optical samples are most informative
**Cons**: More complex, requires careful implementation

---

### 5. Decoupled Contrastive Learning (DCL)
**Paper**: "Decoupled Contrastive Learning" (Yeh et al., 2022)

**Key idea**: Separate the positive and negative terms to reduce the coupling effect that causes temperature sensitivity.

```python
def dcl_loss(sim, labels_mask, temperature=0.07):
    # Positive term: only considers positive pairs
    pos_sim = sim * labels_mask
    pos_loss = -pos_sim.sum(dim=1) / labels_mask.sum(dim=1).clamp_min(1)

    # Negative term: push away negatives independently
    neg_mask = 1 - labels_mask
    neg_sim = torch.exp(sim / temperature) * neg_mask
    neg_loss = torch.log(neg_sim.sum(dim=1) + 1)

    return (pos_loss + neg_loss).mean()
```

**Pros**: More robust to temperature, handles multi-positive naturally
**Cons**: May converge slower

---

### 6. InfoNCE with Hard Negative Mining (Current + Improvements)

**Key improvements to current implementation**:

1. **Temperature capping** (already attempted):
   ```python
   temp_max = 1.0  # Prevent explosion
   ```

2. **Margin-based loss** to prevent collapse:
   ```python
   margin = 0.2
   loss = F.relu(margin - pos_sim + neg_sim.max(dim=1)[0]).mean()
   ```

3. **Memory bank** for more negatives without larger batch:
   ```python
   # Maintain a queue of past optical embeddings
   memory_bank = deque(maxlen=4096)
   ```

---

---

## Implementation Plan: Supervised Contrastive Loss (SupCon)

### Step 0: Modify Dataloader for Multi-Positive Sampling

**Current behavior** ([data_loader.py:195-213](ML+GW+KN/Model/data_loader.py#L195-L213)):
- `BalancedGWBatchedSampler` selects B unique GW events
- Picks **ONE** optical sample per GW event
- Each batch: B samples with B unique `gw_indices`

**Required for SupCon**:
- Select **fewer** GW events (e.g., B/K events)
- Pick **K** optical samples per GW event
- Add extra negative samples from `neg_data_path`
- Batch size remains B, but with grouped positives

#### New Sampler: `MultiPositiveGWBatchedSampler`

Add to [data_loader.py](ML+GW+KN/Model/data_loader.py) after line 216:

```python
class MultiPositiveGWBatchedSampler(Sampler):
    """
    Batch sampler for Supervised Contrastive Learning.
    Ensures multiple optical samples per GW event in each batch.

    Structure per batch:
    - Select (batch_size // samples_per_gw) unique GW events
    - For each GW, sample 'samples_per_gw' optical light curves
    - Total batch size = n_gw_per_batch * samples_per_gw
    """
    def __init__(
        self,
        gw_to_lc_map: dict,
        batch_size: int,
        samples_per_gw: int,
        steps_per_epoch: int,
        min_lc_per_gw: int = 2
    ):
        """
        Args:
            gw_to_lc_map: Dictionary mapping GW_ID -> [LC_ID_1, LC_ID_2, ...]
            batch_size: Total samples per batch
            samples_per_gw: Number of optical samples to draw per GW event
            steps_per_epoch: Number of batches per epoch
            min_lc_per_gw: Minimum light curves required for a GW to be eligible
        """
        self.gw_to_lc_map = gw_to_lc_map
        self.samples_per_gw = samples_per_gw
        self.steps_per_epoch = steps_per_epoch

        # Filter GW events with enough light curves
        self.eligible_gw_ids = [
            gw_id for gw_id, lcs in gw_to_lc_map.items()
            if len(lcs) >= min_lc_per_gw
        ]

        # Number of unique GW events per batch
        self.n_gw_per_batch = batch_size // samples_per_gw

        if self.n_gw_per_batch < 2:
            raise ValueError(
                f"batch_size ({batch_size}) / samples_per_gw ({samples_per_gw}) "
                f"must be >= 2 for contrastive learning"
            )
        if self.n_gw_per_batch > len(self.eligible_gw_ids):
            raise ValueError(
                f"Not enough GW events with >= {min_lc_per_gw} light curves. "
                f"Need {self.n_gw_per_batch}, have {len(self.eligible_gw_ids)}"
            )

        print(f"MultiPositiveSampler: {len(self.eligible_gw_ids)} eligible GW events, "
              f"{self.n_gw_per_batch} GW/batch, {samples_per_gw} samples/GW")

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.steps_per_epoch):
            # 1. Sample unique GW IDs for this batch
            batch_gw_ids = np.random.choice(
                self.eligible_gw_ids,
                size=self.n_gw_per_batch,
                replace=False
            )

            batch_lc_indices = []

            # 2. For each GW ID, sample multiple light curves
            for gw_id in batch_gw_ids:
                possible_lcs = self.gw_to_lc_map[gw_id]
                # Sample with replacement if not enough unique LCs
                replace = len(possible_lcs) < self.samples_per_gw
                chosen_lcs = np.random.choice(
                    possible_lcs,
                    size=self.samples_per_gw,
                    replace=replace
                )
                batch_lc_indices.extend(chosen_lcs.tolist())

            yield batch_lc_indices

    def __len__(self):
        return self.steps_per_epoch
```

#### New Factory Function: `create_supcon_dataloaders`

Add to [data_loader.py](ML+GW+KN/Model/data_loader.py):

```python
def create_supcon_dataloaders(
    h5_path: str,
    batch_size: int = 128,
    samples_per_gw: int = 4,
    val_batch_size: int = None,
    steps_per_epoch: int = None,
    val_steps_per_epoch: int = None,
    val_split: float = 0.1,
    split_seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    negative_h5_path: str = None,
    negative_group: str = "events/optical_data",
    cache_in_memory: bool = False,
    min_lc_per_gw: int = 2
):
    """
    Create dataloaders for Supervised Contrastive Learning.

    Each batch contains:
    - (batch_size // samples_per_gw) unique GW events
    - samples_per_gw optical samples per GW (positives for each other)
    """
    gw_map = build_gw_to_lc_mapping(h5_path)
    train_map, val_map = split_gw_map(gw_map, val_split, split_seed)

    # Calculate steps
    if steps_per_epoch is None:
        n_gw_per_batch = batch_size // samples_per_gw
        steps_per_epoch = max(1, len(train_map) // n_gw_per_batch)

    if val_batch_size is None:
        val_batch_size = batch_size
    if val_steps_per_epoch is None:
        n_gw_per_batch_val = val_batch_size // samples_per_gw
        val_steps_per_epoch = max(1, len(val_map) // n_gw_per_batch_val)

    # Create datasets
    if cache_in_memory:
        shared_dataset = RelationalHDF5Dataset(
            h5_path,
            negative_h5_path=negative_h5_path,
            negative_group=negative_group,
            cache_in_memory=cache_in_memory
        )
        train_dataset = shared_dataset
        val_dataset = shared_dataset
    else:
        train_dataset = RelationalHDF5Dataset(
            h5_path, negative_h5_path=negative_h5_path,
            negative_group=negative_group, cache_in_memory=cache_in_memory
        )
        val_dataset = RelationalHDF5Dataset(
            h5_path, negative_h5_path=negative_h5_path,
            negative_group=negative_group, cache_in_memory=cache_in_memory
        )

    # Create multi-positive samplers
    train_sampler = MultiPositiveGWBatchedSampler(
        gw_to_lc_map=train_map,
        batch_size=batch_size,
        samples_per_gw=samples_per_gw,
        steps_per_epoch=steps_per_epoch,
        min_lc_per_gw=min_lc_per_gw
    )
    val_sampler = MultiPositiveGWBatchedSampler(
        gw_to_lc_map=val_map,
        batch_size=val_batch_size,
        samples_per_gw=samples_per_gw,
        steps_per_epoch=val_steps_per_epoch,
        min_lc_per_gw=min_lc_per_gw
    )

    train_loader = _build_dataloader(
        train_dataset, train_sampler, num_workers,
        pin_memory, persistent_workers, prefetch_factor
    )
    val_loader = _build_dataloader(
        val_dataset, val_sampler, num_workers,
        pin_memory, persistent_workers, prefetch_factor
    )

    return train_loader, val_loader, steps_per_epoch, len(val_sampler)
```

#### Batch Structure Example

With `batch_size=128` and `samples_per_gw=4`:
- 32 unique GW events per batch
- 4 optical samples per GW
- 128 total samples
- Each GW has 3 other positive pairs within the batch

```
Batch gw_indices: [0,0,0,0, 1,1,1,1, 2,2,2,2, ..., 31,31,31,31]
                   ↑______↑  ↑______↑
                   4 positives for GW 0
```

---

### Step 1: Add SupCon Loss Function to model.py

Add new method to `GWOpticalALBEFModel` class in [model.py:849](ML+GW+KN/Model/model.py#L849):

```python
def compute_supcon_loss(self, g, z_l, gw_indices, temperature=0.07):
    """
    Supervised Contrastive Loss for many-to-many GW-optical matching.

    Reference: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)

    Key difference from InfoNCE:
    - Treats ALL samples with same gw_index as positives
    - Normalizes loss by number of positives per anchor
    - More stable with multiple positives per class
    """
    # Project and normalize features
    feat_g = F.normalize(self.gw_proj(g), p=2, dim=1, eps=1e-8)
    feat_o = F.normalize(self.opt_proj(z_l), p=2, dim=1, eps=1e-8)

    batch_size = feat_g.size(0)
    device = feat_g.device

    # Concatenate GW and optical features for unified contrastive learning
    # This allows both modalities to learn from each other
    features = torch.cat([feat_g, feat_o], dim=0)  # [2*B, dim]
    labels = torch.cat([gw_indices, gw_indices], dim=0)  # [2*B]

    # Compute similarity matrix
    sim_matrix = torch.matmul(features, features.T) / temperature  # [2*B, 2*B]

    # Mask for positive pairs (same gw_index, excluding self)
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [2*B, 2*B]
    mask_pos = labels_eq.float()
    mask_pos.fill_diagonal_(0)  # Exclude self-similarity

    # Mask to exclude self from denominator
    mask_self = torch.eye(2 * batch_size, device=device, dtype=torch.bool)

    # Numerical stability: subtract max
    logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()

    # Compute log-softmax denominator (all pairs except self)
    exp_logits = torch.exp(logits)
    exp_logits = exp_logits.masked_fill(mask_self, 0)
    log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    # Log probabilities
    log_prob = logits - log_sum_exp

    # Compute mean log-probability over positive pairs
    num_positives = mask_pos.sum(dim=1)
    # Avoid division by zero for samples with no positives
    num_positives = num_positives.clamp_min(1)

    mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / num_positives

    # Loss is negative mean log-probability
    loss = -mean_log_prob_pos.mean()

    # Return original sim_g2o for hard negative mining compatibility
    sim_g2o = torch.matmul(feat_g, feat_o.T) * (1.0 / temperature)

    return loss, sim_g2o
```

### Step 2: Add Arguments for SupCon Mode

In [ALBEF_train.py:795](ML+GW+KN/Model/ALBEF_train.py#L795), add new arguments:

```python
parser.add_argument("--itc_loss_type", type=str, default="infonce",
                    choices=["infonce", "supcon"],
                    help="ITC loss type: 'infonce' (original) or 'supcon' (supervised contrastive)")
parser.add_argument("--supcon_temperature", type=float, default=0.1,
                    help="Temperature for SupCon loss (typically 0.07-0.2)")
parser.add_argument("--samples_per_gw", type=int, default=4,
                    help="Number of optical samples per GW event for SupCon (default: 4)")
parser.add_argument("--min_lc_per_gw", type=int, default=2,
                    help="Minimum light curves required for a GW to be eligible for SupCon")
```

### Step 3: Modify Training Script

#### 3a. Update Dataloader Creation

In [ALBEF_train.py:364-380](ML+GW+KN/Model/ALBEF_train.py#L364-L380), update to use SupCon dataloader:

```python
# Add import at top:
from data_loader import (
    create_training_dataloader,
    create_train_val_dataloaders,
    create_supcon_dataloaders  # NEW
)

# Replace dataloader creation logic:
if args.val_split is not None and 0 < args.val_split < 1:
    if args.itc_loss_type == "supcon":
        # Use multi-positive sampler for SupCon
        train_loader, val_loader, steps_per_epoch, val_steps = create_supcon_dataloaders(
            h5_path=args.data_path,
            batch_size=args.batch_size,
            samples_per_gw=args.samples_per_gw,
            val_batch_size=args.val_batch_size,
            steps_per_epoch=args.steps_per_epoch,
            val_steps_per_epoch=args.val_steps_per_epoch,
            val_split=args.val_split,
            split_seed=args.split_seed,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=args.prefetch_factor,
            cache_in_memory=bool(args.cache_in_memory),
            negative_h5_path=args.neg_data_path,
            negative_group=args.neg_group,
            min_lc_per_gw=args.min_lc_per_gw
        )
        print(f"SupCon mode: {args.samples_per_gw} samples/GW, "
              f"{args.batch_size // args.samples_per_gw} GW/batch")
    else:
        # Original balanced sampler
        train_loader, val_loader, steps_per_epoch, val_steps = create_train_val_dataloaders(...)
```

#### 3b. Update Loss Computation

In [ALBEF_train.py:553-555](ML+GW+KN/Model/ALBEF_train.py#L553-L555), update to use selected loss:

```python
# Replace:
itc_loss, sim_g2o = model.compute_itc_loss(g, z_l, gw_indices, mask=args.mask_itc)

# With:
if args.itc_loss_type == "supcon":
    itc_loss, sim_g2o = model.compute_supcon_loss(
        g, z_l, gw_indices, temperature=args.supcon_temperature
    )
else:
    itc_loss, sim_g2o = model.compute_itc_loss(
        g, z_l, gw_indices, mask=args.mask_itc
    )
```

Same change in the validation function at [ALBEF_train.py:261-262](ML+GW+KN/Model/ALBEF_train.py#L261-L262).

### Step 4: Create New Config File

Create `ALBEF_supcon.json`:

```json
{
    "data_path": "/fred/oz016/bgao_kn/data/LSST_KN_BNS/combined_dataset.h5",
    "neg_data_path": "/fred/oz016/bgao_kn/data/ELASTICC2_TRAIN/negative_dataset.h5",
    "neg_group": "ELASTICC2_TRAIN/optical_data",
    "epochs": 150,
    "batch_size": 128,
    "steps_per_epoch": null,
    "ckpt_path": "/fred/oz016/bgao_kn/data/model/checkpoints/supcon",
    "stage_to_jobfs": false,
    "cache_in_memory": true,
    "val_split": 0.2,
    "val_batch_size": 64,
    "val_steps_per_epoch": 500,
    "split_seed": 42,
    "early_stop_patience": 20,
    "early_stop_min_delta": 0.001,
    "resume": null,
    "lr": 5e-5,
    "lr_scheduler": "cosine",
    "weight_decay": 1e-3,
    "grad_clip_norm": 1.0,
    "label_smoothing": 0.1,
    "warmup_epochs": 5,
    "min_lr": 1e-6,
    "num_workers": 8,
    "pin_memory": 1,
    "persistent_workers": 1,
    "prefetch_factor": 4,
    "n_ref": 64,
    "ref_start": -0.3,
    "ref_end": 0.6,
    "ref_dim": 64,
    "enc_dim": 64,
    "proj_dim": 128,
    "fusion_attn_dim": null,
    "fusion_hidden_dim": null,
    "fusion_dropout": 0.4,
    "temp_init": 0.1,
    "temp_final": 0.1,
    "temp_min": 0.01,
    "temp_max": 1.0,
    "temp_schedule": "fixed",
    "gw_dropout": 0.5,
    "opt_dropout": 0.3,
    "proj_dropout": 0.2,
    "itc_loss_type": "supcon",
    "supcon_temperature": 0.1,
    "samples_per_gw": 4,
    "min_lc_per_gw": 2,
    "itc_weight": 1.0,
    "cls_weight": 1.0,
    "cls_pos_weight": 1.0,
    "cls_neg_weight": 1.0,
    "cls_extra_neg_weight": 1.0,
    "cls_ramp_epochs": 10,
    "itc_decay_start_epoch": 0,
    "itc_decay_epochs": 0,
    "itc_decay_ratio": 0.0,
    "itc_label_smoothing": 0.0,
    "hard_neg_start_epoch": 0,
    "hard_neg_ramp_epochs": 10,
    "cls_start_epoch": 0,
    "mask_itc": false,
    "use_lightweight_gw": true,
    "gw_aug_noise": 0.05,
    "gw_aug_jitter": 0.02,
    "gw_aug_dropout": 0.1,
    "feature_dropout": 0.1,
    "opt_aug_noise": 0.05,
    "opt_aug_time_jitter": 0.01,
    "opt_aug_dropout": 0.05,
    "opt_aug_band_dropout": 0.05
}
```

---

## Files to Modify

| File | Changes |
|------|---------|
| [data_loader.py](ML+GW+KN/Model/data_loader.py) | Add `MultiPositiveGWBatchedSampler` + `create_supcon_dataloaders()` (~120 lines) |
| [model.py](ML+GW+KN/Model/model.py) | Add `compute_supcon_loss()` method (~50 lines) |
| [ALBEF_train.py](ML+GW+KN/Model/ALBEF_train.py) | Add args + conditional dataloader/loss selection (~30 lines) |
| [args/ALBEF_supcon.json](ML+GW+KN/Model/args/) | New config file |

---

## Why SupCon Works Better

| Aspect | InfoNCE (Current) | SupCon |
|--------|-------------------|--------|
| Positive handling | Soft labels, equal weight | Normalized by # positives |
| Temperature | Learned, tends to explode | Fixed, stable |
| Loss formulation | log(pos / sum_all) | mean(log(pos_i / sum_all)) |
| Multi-positive | Approximated | Native support |

---

## Verification

1. **Test dataloader batch structure**:
   ```python
   # Quick test to verify multi-positive sampling
   from data_loader import build_gw_to_lc_mapping, MultiPositiveGWBatchedSampler
   gw_map = build_gw_to_lc_mapping("/fred/oz016/bgao_kn/data/LSST_KN_BNS/combined_dataset.h5")
   sampler = MultiPositiveGWBatchedSampler(gw_map, batch_size=128, samples_per_gw=4, steps_per_epoch=1)
   batch = next(iter(sampler))
   print(f"Batch size: {len(batch)}")  # Should be 128
   # Check gw_indices grouping
   ```

2. **Run training** with new config:
   ```bash
   python ML+GW+KN/Model/ALBEF_train.py \
       --data_path /fred/oz016/bgao_kn/data/LSST_KN_BNS/combined_dataset.h5 \
       --neg_data_path /fred/oz016/bgao_kn/data/ELASTICC2_TRAIN/negative_dataset.h5 \
       --ckpt_path /fred/oz016/bgao_kn/data/model/checkpoints/supcon \
       --itc_loss_type supcon \
       --samples_per_gw 4 \
       --supcon_temperature 0.1 \
       --epochs 50 \
       --val_split 0.2
   ```

3. **Monitor TensorBoard**:
   - Val ITC Loss should remain stable (not explode)
   - Train/Val ITC accuracy gap should narrow (key metric!)
   - Temperature stays fixed at 0.1
   - Each batch should have ~32 unique GW events × 4 samples = 128 total

4. **Compare with baselines**:
   | Config | Expected Behavior |
   |--------|-------------------|
   | `ALBEF_cls_only.json` (no ITC) | CLS-only baseline |
   | `ALBEF_train.json` (InfoNCE) | Val ITC loss explodes |
   | `ALBEF_supcon.json` (SupCon) | Val ITC loss stable, better generalization |

5. **Success criteria**:
   - Val ITC Loss < 5.0 after 50 epochs (vs >8.0 with InfoNCE)
   - Val ITC Accuracy > 5% (vs ~2% with InfoNCE)
   - Val CLS Loss comparable to CLS-only baseline
