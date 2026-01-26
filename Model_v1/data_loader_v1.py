import h5py
import numpy as np
import torch
from collections import defaultdict
from typing import List, Iterator
from torch.utils.data import Dataset, DataLoader, Sampler


def build_gw_to_lc_mapping(h5_path: str):
    """
    Build mapping from GW event index to optical sample indices.
    """
    print(f"Building GW-to-optical index mapping from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        all_parent_indices = f['events/optical_data/parent_gw_idx'][:]

    gw_to_lc_map = defaultdict(list)
    for lc_idx, gw_idx in enumerate(all_parent_indices):
        gw_to_lc_map[int(gw_idx)].append(lc_idx)

    final_map = {k: np.array(v, dtype=np.int64) for k, v in gw_to_lc_map.items()}
    print(f"Mapping complete. Found {len(final_map)} unique GW events.")
    return final_map


class PixelGWRelationalDataset(Dataset):
    """
    Dataset that returns GW scalar + matched skymap pixel features per optical sample.
    """
    def __init__(
        self,
        h5_path: str,
        negative_h5_path: str = None,
        negative_group: str = "events/optical_data",
        cache_in_memory: bool = False,
        use_neg_gw: bool = False
    ):
        super().__init__()
        self.h5_path = h5_path
        self.h5_file = None
        self.negative_h5_path = negative_h5_path
        self.negative_group = negative_group
        self.neg_file = None
        self.neg_length = None
        self.cache_in_memory = cache_in_memory
        self.data_cache = None
        self.neg_cache = None
        self.use_neg_gw = use_neg_gw
        self.neg_gw_indices = None
        self.n_pos_gw = None
        self._credible_cache = {}

        with h5py.File(h5_path, 'r') as f:
            self.length = f['events/optical_data/values'].shape[0]
            if 'events/optical_data/gw_pixel_features' not in f:
                raise KeyError(
                    "Missing events/optical_data/gw_pixel_features. "
                    "Run preprocess_gw_pixel.py to create the dataset."
                )
            if self.use_neg_gw and 'events/gw_data/has_kn' in f:
                has_kn = f['events/gw_data/has_kn'][:]
                self.neg_gw_indices = np.where(has_kn == 0)[0]
                self.n_pos_gw = int(np.sum(has_kn == 1))
                print(f"Loaded {len(self.neg_gw_indices)} negative GW events (no KN)")
            if self.cache_in_memory:
                print(f"Caching dataset in memory from {h5_path}...")
                self.data_cache = {
                    "opt_val": f['events/optical_data/values'][:],
                    "opt_err": f['events/optical_data/errors'][:],
                    "opt_mask": f['events/optical_data/masks'][:],
                    "opt_time": f['events/optical_data/times'][:],
                    "opt_coords": f['events/optical_data/coordinates'][:],
                    "parent_gw_idx": f['events/optical_data/parent_gw_idx'][:],
                    "gw_scalar": f['events/gw_data/scalars'][:],
                    "gw_pixel": f['events/optical_data/gw_pixel_features'][:]
                }

        if self.negative_h5_path is not None:
            with h5py.File(self.negative_h5_path, 'r') as f:
                if self.negative_group not in f:
                    raise KeyError(
                        f"Negative group '{self.negative_group}' not found in {self.negative_h5_path}"
                    )
                self.neg_length = f[f"{self.negative_group}/values"].shape[0]
                if self.cache_in_memory:
                    print(f"Caching negative dataset in memory from {self.negative_h5_path}...")
                    self.neg_cache = {
                        "neg_val": f[f"{self.negative_group}/values"][:],
                        "neg_err": f[f"{self.negative_group}/errors"][:],
                        "neg_mask": f[f"{self.negative_group}/masks"][:],
                        "neg_time": f[f"{self.negative_group}/times"][:],
                        "neg_coords": f[f"{self.negative_group}/coordinates"][:]
                    }

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        neg_gw_local_idx = None
        if isinstance(idx, (tuple, list)):
            if len(idx) != 2:
                raise TypeError("Expected index int or (opt_idx, neg_gw_local_idx) tuple.")
            opt_idx, neg_gw_local_idx = idx
        elif isinstance(idx, np.ndarray):
            if idx.shape == ():
                opt_idx = int(idx)
            elif idx.size == 2:
                opt_idx, neg_gw_local_idx = idx.tolist()
            else:
                raise TypeError("Expected index int or (opt_idx, neg_gw_local_idx) tuple.")
        else:
            opt_idx = idx

        if self.data_cache is None:
            if self.h5_file is None:
                self.h5_file = h5py.File(self.h5_path, 'r')

            opt_val = torch.from_numpy(self.h5_file['events/optical_data/values'][opt_idx])
            opt_err = torch.from_numpy(self.h5_file['events/optical_data/errors'][opt_idx])
            opt_mask = torch.from_numpy(self.h5_file['events/optical_data/masks'][opt_idx])
            opt_time = torch.from_numpy(self.h5_file['events/optical_data/times'][opt_idx])
            opt_coords = torch.from_numpy(self.h5_file['events/optical_data/coordinates'][opt_idx])

            gw_idx = int(self.h5_file['events/optical_data/parent_gw_idx'][opt_idx])
            gw_scalar = torch.from_numpy(self.h5_file['events/gw_data/scalars'][gw_idx])
            gw_pixel = torch.from_numpy(self.h5_file['events/optical_data/gw_pixel_features'][opt_idx])
        else:
            opt_val = torch.from_numpy(self.data_cache['opt_val'][opt_idx])
            opt_err = torch.from_numpy(self.data_cache['opt_err'][opt_idx])
            opt_mask = torch.from_numpy(self.data_cache['opt_mask'][opt_idx])
            opt_time = torch.from_numpy(self.data_cache['opt_time'][opt_idx])
            opt_coords = torch.from_numpy(self.data_cache['opt_coords'][opt_idx])

            gw_idx = int(self.data_cache['parent_gw_idx'][opt_idx])
            gw_scalar = torch.from_numpy(self.data_cache['gw_scalar'][gw_idx])
            gw_pixel = torch.from_numpy(self.data_cache['gw_pixel'][opt_idx])

        is_neg_gw = False
        if (
            neg_gw_local_idx is not None
            and self.use_neg_gw
            and int(neg_gw_local_idx) >= 0
        ):
            neg_actual_idx = int(self.neg_gw_indices[int(neg_gw_local_idx)])
            if self.data_cache is not None:
                gw_scalar = torch.from_numpy(self.data_cache['gw_scalar'][neg_actual_idx])
            else:
                if self.h5_file is None:
                    self.h5_file = h5py.File(self.h5_path, 'r')
                gw_scalar = torch.from_numpy(self.h5_file['events/gw_data/scalars'][neg_actual_idx])
            gw_pixel = self._get_gw_pixel_for_coords(neg_actual_idx, opt_coords)
            gw_idx = neg_actual_idx
            is_neg_gw = True

        gw_input = torch.cat([gw_scalar, gw_pixel], dim=0)

        if self.negative_h5_path is not None:
            neg_idx = np.random.randint(0, self.neg_length)
            if self.neg_cache is None:
                if self.neg_file is None:
                    self.neg_file = h5py.File(self.negative_h5_path, 'r')
                neg_val = torch.from_numpy(self.neg_file[f"{self.negative_group}/values"][neg_idx])
                neg_err = torch.from_numpy(self.neg_file[f"{self.negative_group}/errors"][neg_idx])
                neg_mask = torch.from_numpy(self.neg_file[f"{self.negative_group}/masks"][neg_idx])
                neg_time = torch.from_numpy(self.neg_file[f"{self.negative_group}/times"][neg_idx])
                neg_coords = torch.from_numpy(self.neg_file[f"{self.negative_group}/coordinates"][neg_idx])
            else:
                neg_val = torch.from_numpy(self.neg_cache['neg_val'][neg_idx])
                neg_err = torch.from_numpy(self.neg_cache['neg_err'][neg_idx])
                neg_mask = torch.from_numpy(self.neg_cache['neg_mask'][neg_idx])
                neg_time = torch.from_numpy(self.neg_cache['neg_time'][neg_idx])
                neg_coords = torch.from_numpy(self.neg_cache['neg_coords'][neg_idx])

            if self.use_neg_gw:
                return (
                    gw_input, opt_time, opt_val, opt_mask, opt_err, opt_coords, gw_idx,
                    neg_time, neg_val, neg_mask, neg_err, neg_coords, is_neg_gw
                )
            return (
                gw_input, opt_time, opt_val, opt_mask, opt_err, opt_coords, gw_idx,
                neg_time, neg_val, neg_mask, neg_err, neg_coords
            )

        if self.use_neg_gw:
            return gw_input, opt_time, opt_val, opt_mask, opt_err, opt_coords, gw_idx, is_neg_gw
        return gw_input, opt_time, opt_val, opt_mask, opt_err, opt_coords, gw_idx

    @staticmethod
    def _coords_to_unit_vector(coords):
        coords_np = coords.detach().cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
        ra = np.deg2rad(coords_np[0])
        dec = np.deg2rad(coords_np[1])
        x = np.sin(dec) * np.cos(ra)
        y = np.sin(dec) * np.sin(ra)
        z = np.cos(dec)
        return np.array([x, y, z], dtype=np.float32)

    def _get_credible_levels(self, gw_idx, prob_mass):
        if gw_idx in self._credible_cache:
            return self._credible_cache[gw_idx]
        prob = prob_mass.astype(np.float64)
        prob = prob / prob.sum()
        order = np.argsort(prob)[::-1]
        cumulative = np.cumsum(prob[order])
        credible = np.empty_like(prob)
        credible[order] = cumulative
        credible = credible.astype(np.float32)
        self._credible_cache[gw_idx] = credible
        return credible

    def _get_gw_pixel_for_coords(self, gw_idx, opt_coords):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
        skymap = self.h5_file['events/gw_data/skymaps'][gw_idx]
        pix_vectors = skymap[0:3]
        opt_vec = self._coords_to_unit_vector(opt_coords)
        scores = np.dot(pix_vectors.T, opt_vec)
        pix_idx = int(np.argmax(scores))
        credible = self._get_credible_levels(gw_idx, skymap[4])
        pixel_feats = skymap[:, pix_idx].astype(np.float32)
        gw_pixel = np.concatenate([pixel_feats, np.array([credible[pix_idx]], dtype=np.float32)], axis=0)
        return torch.from_numpy(gw_pixel)


class BalancedGWBatchedSampler(Sampler):
    """
    Batch sampler that selects unique GW events, one optical sample per event.
    """
    def __init__(self, gw_to_lc_map: dict, batch_size: int, steps_per_epoch: int):
        self.gw_to_lc_map = gw_to_lc_map
        self.unique_gw_ids = list(gw_to_lc_map.keys())
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

        if self.batch_size > len(self.unique_gw_ids):
            raise ValueError(
                f"Batch size ({batch_size}) > Unique GW events ({len(self.unique_gw_ids)})."
            )

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.steps_per_epoch):
            batch_gw_ids = np.random.choice(
                self.unique_gw_ids,
                size=self.batch_size,
                replace=False
            )
            batch_lc_indices = []
            for gw_id in batch_gw_ids:
                possible_lcs = self.gw_to_lc_map[gw_id]
                chosen_lc = np.random.choice(possible_lcs)
                batch_lc_indices.append(int(chosen_lc))
            yield batch_lc_indices

    def __len__(self):
        return self.steps_per_epoch


class MixedGWBatchedSampler(Sampler):
    """
    Batch sampler that includes both positive and negative GW events.

    Structure per batch (batch_size=128, neg_gw_ratio=0.2):
    - 102 positive GW-optical pairs (80%)
    - 26 negative GW paired with random optical (20%)

    The sampler yields lists of (opt_idx, neg_gw_local_idx) tuples:
    - opt_idx: Optical sample index
    - neg_gw_local_idx: -1 means "use parent GW", >= 0 means "use negative GW at this local index"
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
        self.neg_gw_local_indices = np.array(neg_gw_indices, dtype=int)
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.neg_gw_ratio = neg_gw_ratio
        self.samples_per_gw = samples_per_gw

        self.pos_gw_ids = list(gw_to_lc_map.keys())
        self.n_neg_per_batch = int(batch_size * neg_gw_ratio)
        self.n_pos_per_batch = batch_size - self.n_neg_per_batch

        self.all_opt_indices = []
        for lcs in gw_to_lc_map.values():
            self.all_opt_indices.extend(lcs)
        self.all_opt_indices = np.array(self.all_opt_indices)

        if samples_per_gw > 1:
            self.n_gw_per_batch = self.n_pos_per_batch // samples_per_gw
            self.n_pos_per_batch = self.n_gw_per_batch * samples_per_gw
            self.n_neg_per_batch = self.batch_size - self.n_pos_per_batch
        else:
            self.n_gw_per_batch = self.n_pos_per_batch

        if self.n_gw_per_batch > len(self.pos_gw_ids):
            raise ValueError(
                f"Not enough positive GW events. Need {self.n_gw_per_batch}, have {len(self.pos_gw_ids)}"
            )
        if len(self.neg_gw_local_indices) == 0 and self.n_neg_per_batch > 0:
            raise ValueError("No negative GW indices provided for mixed sampling.")

        print(
            f"MixedGWBatchedSampler: {len(self.pos_gw_ids)} positive GW, "
            f"{len(self.neg_gw_local_indices)} negative GW, "
            f"{self.n_pos_per_batch} pos/batch, {self.n_neg_per_batch} neg/batch"
        )

    def __iter__(self) -> Iterator[list]:
        for _ in range(self.steps_per_epoch):
            batch_opt_indices = []
            batch_neg_gw_local_indices = []

            if self.samples_per_gw > 1:
                batch_gw_ids = np.random.choice(
                    self.pos_gw_ids, self.n_gw_per_batch, replace=False
                )
                for gw_id in batch_gw_ids:
                    lcs = self.gw_to_lc_map[gw_id]
                    replace = len(lcs) < self.samples_per_gw
                    chosen_lcs = np.random.choice(lcs, self.samples_per_gw, replace=replace)
                    batch_opt_indices.extend(chosen_lcs.tolist())
                    batch_neg_gw_local_indices.extend([-1] * self.samples_per_gw)
            else:
                batch_gw_ids = np.random.choice(
                    self.pos_gw_ids, self.n_pos_per_batch, replace=False
                )
                for gw_id in batch_gw_ids:
                    lcs = self.gw_to_lc_map[gw_id]
                    batch_opt_indices.append(np.random.choice(lcs))
                    batch_neg_gw_local_indices.append(-1)

            if self.n_neg_per_batch > 0:
                opt_replace = self.n_neg_per_batch > len(self.all_opt_indices)
                neg_opt = np.random.choice(self.all_opt_indices, self.n_neg_per_batch, replace=opt_replace)
                neg_replace = self.n_neg_per_batch > len(self.neg_gw_local_indices)
                neg_gw_local = np.random.choice(
                    self.neg_gw_local_indices, self.n_neg_per_batch, replace=neg_replace
                )
                batch_opt_indices.extend(neg_opt.tolist())
                batch_neg_gw_local_indices.extend(neg_gw_local.tolist())

            batch_pairs = list(zip(batch_opt_indices, batch_neg_gw_local_indices))
            yield batch_pairs

    def __len__(self):
        return self.steps_per_epoch


def split_gw_map(gw_to_lc_map: dict, val_split: float, seed: int):
    gw_ids = np.array(list(gw_to_lc_map.keys()))
    rng = np.random.default_rng(seed)
    rng.shuffle(gw_ids)

    val_size = max(1, int(len(gw_ids) * val_split))
    val_ids = set(gw_ids[:val_size])
    train_ids = set(gw_ids[val_size:])

    train_map = {gw_id: gw_to_lc_map[gw_id] for gw_id in train_ids}
    val_map = {gw_id: gw_to_lc_map[gw_id] for gw_id in val_ids}
    return train_map, val_map


def _build_dataloader(
    dataset: Dataset,
    sampler: Sampler,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int
):
    if num_workers > 0:
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=pin_memory
    )


def create_training_dataloader(
    h5_path: str,
    batch_size: int = 32,
    steps_per_epoch: int = 1000,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 4,
    negative_h5_path: str = None,
    negative_group: str = "events/optical_data",
    cache_in_memory: bool = False
):
    if cache_in_memory and num_workers > 0:
        print("cache_in_memory=True with num_workers>0 may increase RAM usage.")

    gw_map = build_gw_to_lc_mapping(h5_path)
    dataset = PixelGWRelationalDataset(
        h5_path,
        negative_h5_path=negative_h5_path,
        negative_group=negative_group,
        cache_in_memory=cache_in_memory
    )
    sampler = BalancedGWBatchedSampler(
        gw_to_lc_map=gw_map,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch
    )
    return _build_dataloader(
        dataset,
        sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor
    )


def create_train_val_dataloaders(
    h5_path: str,
    batch_size: int = 32,
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
    cache_in_memory: bool = False
):
    if cache_in_memory and num_workers > 0:
        print("cache_in_memory=True with num_workers>0 may increase RAM usage.")

    gw_map = build_gw_to_lc_mapping(h5_path)
    train_map, val_map = split_gw_map(gw_map, val_split, split_seed)

    if steps_per_epoch is None:
        train_optical = sum(len(v) for v in train_map.values())
        steps_per_epoch = max(1, train_optical // batch_size)
    if val_batch_size is None:
        val_batch_size = batch_size
    val_batch_size = min(val_batch_size, len(val_map))
    if val_batch_size < 1:
        raise ValueError("val_batch_size must be >= 1.")
    if val_steps_per_epoch is None:
        val_optical = sum(len(v) for v in val_map.values())
        val_steps_per_epoch = max(1, val_optical // val_batch_size)

    if cache_in_memory:
        shared_dataset = PixelGWRelationalDataset(
            h5_path,
            negative_h5_path=negative_h5_path,
            negative_group=negative_group,
            cache_in_memory=cache_in_memory
        )
        train_dataset = shared_dataset
        val_dataset = shared_dataset
    else:
        train_dataset = PixelGWRelationalDataset(
            h5_path,
            negative_h5_path=negative_h5_path,
            negative_group=negative_group,
            cache_in_memory=cache_in_memory
        )
        val_dataset = PixelGWRelationalDataset(
            h5_path,
            negative_h5_path=negative_h5_path,
            negative_group=negative_group,
            cache_in_memory=cache_in_memory
        )

    train_sampler = BalancedGWBatchedSampler(
        gw_to_lc_map=train_map,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch
    )
    val_sampler = BalancedGWBatchedSampler(
        gw_to_lc_map=val_map,
        batch_size=val_batch_size,
        steps_per_epoch=val_steps_per_epoch
    )

    train_loader = _build_dataloader(
        train_dataset,
        train_sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor
    )
    val_loader = _build_dataloader(
        val_dataset,
        val_sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor
    )

    return train_loader, val_loader, steps_per_epoch, len(val_sampler)


def create_mixed_gw_dataloaders(
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
    neg_gw_ratio: float = 0.2
):
    if cache_in_memory and num_workers > 0:
        print("cache_in_memory=True with num_workers>0 may increase RAM usage.")

    gw_map = build_gw_to_lc_mapping(h5_path)
    train_map, val_map = split_gw_map(gw_map, val_split, split_seed)

    if steps_per_epoch is None:
        total_train_optical = sum(len(lcs) for lcs in train_map.values())
        steps_per_epoch = max(1, total_train_optical // batch_size)
    if val_batch_size is None:
        val_batch_size = batch_size
    val_batch_size = min(val_batch_size, max(1, len(val_map) * samples_per_gw))
    if val_steps_per_epoch is None:
        total_val_optical = sum(len(lcs) for lcs in val_map.values())
        val_steps_per_epoch = max(1, total_val_optical // val_batch_size)

    train_dataset = PixelGWRelationalDataset(
        h5_path,
        negative_h5_path=negative_h5_path,
        negative_group=negative_group,
        cache_in_memory=cache_in_memory,
        use_neg_gw=True
    )
    val_dataset = PixelGWRelationalDataset(
        h5_path,
        negative_h5_path=negative_h5_path,
        negative_group=negative_group,
        cache_in_memory=cache_in_memory,
        use_neg_gw=True
    )

    neg_gw_indices = train_dataset.neg_gw_indices
    if neg_gw_indices is None or len(neg_gw_indices) == 0:
        raise ValueError("No negative GW indices found; cannot use mixed GW sampling.")

    neg_local_indices = np.arange(len(neg_gw_indices))
    rng = np.random.default_rng(split_seed)
    rng.shuffle(neg_local_indices)
    val_neg_count = max(1, int(len(neg_local_indices) * val_split))
    val_neg_local = neg_local_indices[:val_neg_count]
    train_neg_local = neg_local_indices[val_neg_count:]

    train_sampler = MixedGWBatchedSampler(
        gw_to_lc_map=train_map,
        neg_gw_indices=train_neg_local,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        neg_gw_ratio=neg_gw_ratio,
        samples_per_gw=samples_per_gw
    )
    val_sampler = MixedGWBatchedSampler(
        gw_to_lc_map=val_map,
        neg_gw_indices=val_neg_local,
        batch_size=val_batch_size,
        steps_per_epoch=val_steps_per_epoch,
        neg_gw_ratio=neg_gw_ratio,
        samples_per_gw=samples_per_gw
    )

    train_loader = _build_dataloader(
        train_dataset,
        train_sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor
    )
    val_loader = _build_dataloader(
        val_dataset,
        val_sampler,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor
    )

    return train_loader, val_loader, steps_per_epoch, len(val_sampler)
