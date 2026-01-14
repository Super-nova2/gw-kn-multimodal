import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
import os
import healpy as hp
from ligo.skymap.io.fits import read_sky_map
from ligo.skymap.moc import uniq2nest, uniq2pixarea
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import torch
from collections import defaultdict
from typing import List, Iterator
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
import subprocess


def build_gw_to_lc_mapping(h5_path: str):
    """
    Scans the HDF5 file to build a mapping from GW Event Index to Light Curve Indices.
    This is required for the Balanced Sampler.
    
    Args:
        h5_path: Path to the HDF5 file.
        
    Returns:
        gw_to_lc_map: Dictionary {gw_idx: np.array([lc_idx_1, lc_idx_2, ...])}
    """
    print(f"Building GW-to-Optical index mapping from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        # Load the parent_gw_idx array into memory (it's essentially a list of integers)
        # Shape: [Total_Optical_Samples]
        all_parent_indices = f['events/optical_data/parent_gw_idx'][:]
        
    gw_to_lc_map = defaultdict(list)
    for lc_idx, gw_idx in enumerate(all_parent_indices):
        gw_to_lc_map[gw_idx].append(lc_idx)
        
    # Convert lists to numpy arrays for faster random sampling later
    final_map = {k: np.array(v) for k, v in gw_to_lc_map.items()}
    
    print(f"Mapping complete. Found {len(final_map)} unique GW events.")
    return final_map


class RelationalHDF5Dataset(Dataset):
    """
    PyTorch Dataset for the relational HDF5 structure.
    Reads optical data by index and fetches the corresponding unique GW data.
    Optional negative optical samples can be drawn from a separate HDF5 file.
    """
    def __init__(
        self,
        h5_path: str,
        negative_h5_path: str = None,
        negative_group: str = "events/optical_data",
        cache_in_memory: bool = False
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
        
        # Open file temporarily to get dataset length
        with h5py.File(h5_path, 'r') as f:
            self.length = f['events/optical_data/values'].shape[0]
            if self.cache_in_memory:
                print(f"Caching positive dataset in memory from {h5_path}...")
                self.data_cache = {
                    "opt_val": f['events/optical_data/values'][:],
                    "opt_err": f['events/optical_data/errors'][:],
                    "opt_mask": f['events/optical_data/masks'][:],
                    "opt_time": f['events/optical_data/times'][:],
                    "opt_coords": f['events/optical_data/coordinates'][:],
                    "parent_gw_idx": f['events/optical_data/parent_gw_idx'][:],
                    "gw_scalar": f['events/gw_data/scalars'][:],
                    "gw_skymap": f['events/gw_data/skymaps'][:]
                }
        
        if self.negative_h5_path is not None:
            with h5py.File(self.negative_h5_path, 'r') as f:
                if self.negative_group not in f:
                    raise KeyError(f"Negative group '{self.negative_group}' not found in {self.negative_h5_path}")
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
        """
        Args:
            idx: Index of the light curve (optical data).
        """
        if self.data_cache is None:
            # Lazy loading: Open file only when needed (crucial for num_workers > 0)
            if self.h5_file is None:
                self.h5_file = h5py.File(self.h5_path, 'r')

            # 1. Retrieve Optical Data (Values, Errors, Masks, Times)
            #    HDF5 structure: events/optical_data/...
            opt_val = torch.from_numpy(self.h5_file['events/optical_data/values'][idx])
            opt_err = torch.from_numpy(self.h5_file['events/optical_data/errors'][idx])
            opt_mask = torch.from_numpy(self.h5_file['events/optical_data/masks'][idx])
            opt_time = torch.from_numpy(self.h5_file['events/optical_data/times'][idx])
            opt_coords = torch.from_numpy(self.h5_file['events/optical_data/coordinates'][idx])

            # 2. Retrieve Parent GW Index
            gw_idx = self.h5_file['events/optical_data/parent_gw_idx'][idx]

            # 3. Retrieve Unique GW Data using gw_idx
            #    HDF5 structure: events/gw/...
            gw_scalar = torch.from_numpy(self.h5_file['events/gw_data/scalars'][gw_idx])
            gw_skymap = torch.from_numpy(self.h5_file['events/gw_data/skymaps'][gw_idx])
        else:
            opt_val = torch.from_numpy(self.data_cache["opt_val"][idx])
            opt_err = torch.from_numpy(self.data_cache["opt_err"][idx])
            opt_mask = torch.from_numpy(self.data_cache["opt_mask"][idx])
            opt_time = torch.from_numpy(self.data_cache["opt_time"][idx])
            opt_coords = torch.from_numpy(self.data_cache["opt_coords"][idx])
            gw_idx = self.data_cache["parent_gw_idx"][idx]
            gw_scalar = torch.from_numpy(self.data_cache["gw_scalar"][gw_idx])
            gw_skymap = torch.from_numpy(self.data_cache["gw_skymap"][gw_idx])
        
        # Optional: Retrieve Negative Optical Data (non-KN or unrelated transient)
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
                neg_val = torch.from_numpy(self.neg_cache["neg_val"][neg_idx])
                neg_err = torch.from_numpy(self.neg_cache["neg_err"][neg_idx])
                neg_mask = torch.from_numpy(self.neg_cache["neg_mask"][neg_idx])
                neg_time = torch.from_numpy(self.neg_cache["neg_time"][neg_idx])
                neg_coords = torch.from_numpy(self.neg_cache["neg_coords"][neg_idx])

            # Return tuple: (GW_Inputs, Optical_Inputs, Metadata, Negative_Optical_Inputs)
            # gw_idx is returned for masking the contrastive loss (handling same-source negatives)
            return (
                gw_scalar, gw_skymap, opt_time, opt_val, opt_mask, opt_err, opt_coords, int(gw_idx),
                neg_time, neg_val, neg_mask, neg_err, neg_coords
            )

        # Return tuple: (GW_Inputs, Optical_Inputs, Metadata)
        # gw_idx is returned for masking the contrastive loss (handling same-source negatives)
        return gw_scalar, gw_skymap, opt_time, opt_val, opt_mask, opt_err, opt_coords, int(gw_idx)
    
class BalancedGWBatchedSampler(Sampler):
    """
    Custom Batch Sampler that ensures:
    1. Each batch contains 'batch_size' UNIQUE GW events.
    2. For each selected GW event, ONE light curve is randomly sampled.
    
    This prevents "false negatives" in contrastive learning where multiple LCs 
    from the same GW event appear in the same batch.
    """
    def __init__(self, gw_to_lc_map: dict, batch_size: int, steps_per_epoch: int):
        """
        Args:
            gw_to_lc_map: Dictionary mapping GW_ID -> [LC_ID_1, LC_ID_2, ...]
            batch_size: Number of unique GW events per batch.
            steps_per_epoch: Number of batches to yield per 'epoch'. (N_LC // batch_size)
        """
        self.gw_to_lc_map = gw_to_lc_map
        self.unique_gw_ids = list(gw_to_lc_map.keys())
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        
        # Validation: Batch size cannot exceed total unique GW events
        if self.batch_size > len(self.unique_gw_ids):
            raise ValueError(f"Batch size ({batch_size}) > Unique GW events ({len(self.unique_gw_ids)}).")

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.steps_per_epoch):
            # 1. Sample unique GW IDs for this batch (without replacement)
            batch_gw_ids = np.random.choice(
                self.unique_gw_ids, 
                size=self.batch_size, 
                replace=False
            )
            
            batch_lc_indices = []
            
            # 2. For each GW ID, sample ONE random light curve index
            for gw_id in batch_gw_ids:
                possible_lcs = self.gw_to_lc_map[gw_id]
                chosen_lc = np.random.choice(possible_lcs)
                batch_lc_indices.append(chosen_lc)
                
            # Yield the list of optical indices for the DataLoader to fetch
            yield batch_lc_indices

    def __len__(self):
        return self.steps_per_epoch

class GWBatchedSampler(Sampler):
    """
    Batch sampler that iterates over GW IDs once per epoch.
    """
    def __init__(self, gw_to_lc_map: dict, batch_size: int, shuffle: bool = True, max_steps: int = None):
        self.gw_to_lc_map = gw_to_lc_map
        self.gw_ids = np.array(sorted(gw_to_lc_map.keys()))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_steps = max_steps

        if self.batch_size > len(self.gw_ids):
            raise ValueError(f"Batch size ({batch_size}) > Unique GW events ({len(self.gw_ids)}).")

    def __iter__(self) -> Iterator[List[int]]:
        gw_ids = self.gw_ids.copy()
        if self.shuffle:
            np.random.shuffle(gw_ids)

        steps = 0
        for i in range(0, len(gw_ids), self.batch_size):
            if self.max_steps is not None and steps >= self.max_steps:
                break
            batch_gw_ids = gw_ids[i:i + self.batch_size]
            if len(batch_gw_ids) < self.batch_size:
                break

            batch_lc_indices = []
            for gw_id in batch_gw_ids:
                possible_lcs = self.gw_to_lc_map[gw_id]
                chosen_lc = np.random.choice(possible_lcs)
                batch_lc_indices.append(chosen_lc)

            yield batch_lc_indices
            steps += 1

    def __len__(self):
        total_steps = len(self.gw_ids) // self.batch_size
        if self.max_steps is None:
            return total_steps
        return min(total_steps, self.max_steps)

def split_gw_map(gw_to_lc_map: dict, val_split: float, seed: int):
    if val_split <= 0 or val_split >= 1:
        raise ValueError("val_split must be in (0, 1).")

    gw_ids = np.array(sorted(gw_to_lc_map.keys()))
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
    """
    Factory function to initialize the Dataset, Sampler, and DataLoader.
    """
    if cache_in_memory and num_workers > 0:
        print("cache_in_memory=True with num_workers>0 may increase RAM usage.")
    # 1. Build Index Map (Once)
    gw_map = build_gw_to_lc_mapping(h5_path)
    
    # 2. Initialize Dataset
    dataset = RelationalHDF5Dataset(
        h5_path,
        negative_h5_path=negative_h5_path,
        negative_group=negative_group,
        cache_in_memory=cache_in_memory
    )
    
    # 3. Initialize Custom Sampler
    # Note: 'steps_per_epoch' defines how many batches constitute one epoch loop
    sampler = BalancedGWBatchedSampler(
        gw_to_lc_map=gw_map,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch
    )

    # 4. Initialize DataLoader
    # IMPORTANT: batch_sampler is used, so batch_size/shuffle/sampler/drop_last 
    # arguments in DataLoader constructor must not be provided.
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
    if val_batch_size < batch_size:
        print(f"Validation batch size adjusted to {val_batch_size} based on unique GW events.")
    if val_steps_per_epoch is None:
        val_optical = sum(len(v) for v in val_map.values())
        val_steps_per_epoch = max(1, val_optical // val_batch_size)

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
            h5_path,
            negative_h5_path=negative_h5_path,
            negative_group=negative_group,
            cache_in_memory=cache_in_memory
        )
        val_dataset = RelationalHDF5Dataset(
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


# Constants
BAND_MAP = {'LSST-u': 0, 'LSST-g': 1, 'LSST-r': 2, 'LSST-i': 3, 'LSST-z': 4, 'LSST-Y': 5}
NEG_BAND_MAP = {'u': 0, 'g': 1, 'r': 2, 'i': 3, 'z': 4, 'Y': 5}
NUM_BANDS = 6
MAX_LC_LENGTH = 200  # Maximum length of light curves all band

# Functions for parsing SNANA FITS files, and sampling MOC skymaps
def parse_snana_fits(event_id, sim_dir, sim_name="LSST_KN_BNS"):
    """
    Parses {event_id}_HEAD.fits and {event_id}_PHOT.fits.
    Extracts multiple light curve realizations for a single GW event.
    
    Args:
        event_id: String ID of the event.
        sim_dir: Directory containing FITS files.
        
    Returns:
        List of tuples: [(values, masks, times), ...]
        Returns empty list if files are missing.
    """
    if type(event_id) == str:
        event_id = int(float(event_id))
    head_path = os.path.join(sim_dir, f"{sim_name}_{event_id}",f"{sim_name}_{event_id}_HEAD.FITS")
    phot_path = os.path.join(sim_dir, f"{sim_name}_{event_id}",f"{sim_name}_{event_id}_PHOT.FITS")

    if not os.path.exists(head_path) or not os.path.exists(phot_path):
        print(f"Warning: FITS files not found for {event_id}")
        return []

    try:
        # open readme file and get MJD explode value
        with open(os.path.join(sim_dir, f"{sim_name}_{event_id}",f"{sim_name}_{event_id}.README")) as f:
            readme_lines = f.readlines()
            mjd_explode = readme_lines[27].split(":")[1].split()[0]
            mjd_explode = float(mjd_explode)
        # Open FITS files
        with fits.open(head_path) as hdul_head, fits.open(phot_path) as hdul_phot:
            # Usually data is in extension 1
            data_head = hdul_head[1].data
            data_phot = hdul_phot[1].data
            
            # Use columns directly (Astropy FITS columns are case-insensitive usually)
            # HEAD columns
            ptrobs_min = data_head['PTROBS_MIN']
            ptrobs_max = data_head['PTROBS_MAX']
            
            # PHOT columns
            mjd_all = data_phot['MJD']
            flux_all = data_phot['FLUXCAL']
            fluxerr_all = data_phot['FLUXCALERR'] # Optional usage
            flt_all = data_phot['BAND'] # Filters

            extracted_lcs = []
            
            # Iterate over each realization in HEAD
            for i in range(len(data_head)):
                # SNANA uses 1-based indexing for pointers, Python uses 0-based
                # Start index: value - 1
                # End index: value (exclusive in python slicing)
                start_idx = ptrobs_min[i] - 1
                end_idx = ptrobs_max[i]

                # get coordinates
                ra = data_head['RA'][i]
                dec = data_head['DEC'][i]
                coordinates = np.array([ra, dec], dtype=np.float32)
                
                # Slicing the PHOT data
                lc_mjd = mjd_all[start_idx : end_idx]
                lc_flux = flux_all[start_idx : end_idx]
                lc_fluxerr = fluxerr_all[start_idx : end_idx]
                lc_flt = flt_all[start_idx : end_idx]

                # Normalization
                std = np.std(lc_flux)
                mean = np.mean(lc_flux)
                lc_flux = (lc_flux - mean) / (std + 1e-8)
                lc_fluxerr = lc_fluxerr / (std + 1e-8)
                
                # --- Format Conversion (to Tensor-ready numpy) ---
                val_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)    # Values matrix (flux)
                err_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)    # Errors matrix (flux errors)
                mask_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)
                time_vec = np.zeros((MAX_LC_LENGTH,), dtype=np.float32)
                
                # 1. Time Normalization (Relative to BNS merger time)
                if len(lc_mjd) > 0:
                    rel_times = (lc_mjd - mjd_explode) / 100  # Scale down to manageable range[-0.3, 0.6]
                else:
                    continue # Skip empty light curves

                # 2. Fill Matrices
                # Truncate if longer than MAX_LC_LENGTH
                seq_len = min(len(lc_mjd), MAX_LC_LENGTH)
                if len(lc_mjd) > MAX_LC_LENGTH:
                    print(f"Warning: Light curve for event {event_id} exceeds MAX_LC_LENGTH. Truncating.")
                    # Keep the MAX_LC_LENGTH points with smallest absolute rel_times
                    sorted_indices = np.argsort(np.abs(rel_times))[:MAX_LC_LENGTH]
                    sorted_indices = np.sort(sorted_indices)  # Sort back to chronological order
                    lc_mjd = lc_mjd[sorted_indices]
                    lc_flux = lc_flux[sorted_indices]
                    lc_fluxerr = lc_fluxerr[sorted_indices]
                    lc_flt = lc_flt[sorted_indices]
                    rel_times = rel_times[sorted_indices]
                
                for t in range(seq_len):
                    band_char = lc_flt[t].strip() # Remove whitespace
                    if band_char in BAND_MAP:
                        b_idx = BAND_MAP[band_char]
                        
                        val_mat[t, b_idx] = lc_flux[t]
                        err_mat[t, b_idx] = lc_fluxerr[t]
                        mask_mat[t, b_idx] = 1.0
                        time_vec[t] = rel_times[t]
                
                extracted_lcs.append((val_mat, err_mat, mask_mat, time_vec, coordinates))
                
            return extracted_lcs

    except Exception as e:
        print(f"Error processing FITS for {event_id}: {e}")
        return []

def sample_moc_skymap(map_file):
    """
    Convert UNIQ to sky coordinates and calculte pixel areas.
    Note: Do not include DISTNORM in the output. For inf values in DISTMU, relace with distmean and diststd from metadata.
    """
    
    # read moc skymap and metadata
    moc_map = read_sky_map(map_file, moc=True, distances=True)
    _, meta = read_sky_map(map_file, nest=True)

    # extract distance meta info
    dist_mean = meta.get('distmean', None)
    dist_std = meta.get('diststd', None)

    uniq = moc_map['UNIQ']
    probdensity = moc_map['PROBDENSITY']
    distmu = moc_map['DISTMU']
    distsigma = moc_map['DISTSIGMA']
    # distnorm = moc_map['DISTNORM']  # not used

    # 1) UNIQ -> order, ipix, nside
    order, ipix = uniq2nest(uniq)

    # 2) caculate pixel area
    dA = uniq2pixarea(uniq)
    # 3) calculate pixel probability
    dP = probdensity * dA
    # 4) calculate theta, phi
    xs  = np.zeros_like(ipix, dtype=np.float32)
    ys = np.zeros_like(ipix, dtype=np.float32)
    zs = np.zeros_like(ipix, dtype=np.float32)
    for k in np.unique(order):
        m = (order == k)
        this_ipix  = ipix[m]
        this_nside = 2 ** k
        theta, phi = hp.pix2ang(this_nside, this_ipix, nest=True)
        # ras[m]  = np.degrees(phi)
        # decs[m] = 90.0 - np.degrees(theta)
        xs[m] = np.cos(theta) * np.cos(phi)   # dec,[0, pi]
        ys[m] = np.cos(theta) * np.sin(phi)   # ra,[0, 2pi]
        zs[m] = np.sin(theta)
    
    # 5) return torch tensors
    gw_mocmap = torch.tensor(np.vstack([xs, ys, zs, dA, 100 * dP, distmu, distsigma]), dtype=torch.float32)   # [7, N_pixels], no distnorm

    # 6) process unnormal distance values
    inf_dist_mu = torch.where(torch.isinf(gw_mocmap[5]))[0]
    gw_mocmap[5, inf_dist_mu] = dist_mean  # set inf to mean value
    gw_mocmap[6, inf_dist_mu] = dist_std   # set inf to std value
    gw_mocmap[5,:] = gw_mocmap[5,:] / 1000.0  # scale down
    gw_mocmap[6,:] = gw_mocmap[6,:] / 1000.0 # scale down

    return gw_mocmap  # [7, N_pixels]

# Function to create relational dataset
def create_relational_dataset(
    gw_catalog_path, 
    fits_dir, 
    output_h5_path
):
    """
    Main function to process all data and save to HDF5.
    """
    # 1. Load GW Catalog
    print(f"Loading GW Catalog from {gw_catalog_path}...")
    # Assuming CSV has columns: event_id, m1, m2, ..., skymap_path
    gw_df = pd.read_csv(gw_catalog_path)
    
    n_unique = len(gw_df)
    
    # 2. Initialize HDF5 File
    with h5py.File(output_h5_path, 'w') as f:
        # --- Group A: Unique GW Events ---
        grp_gw = f.create_group('events/gw_data')
        
        # Pre-allocate GW datasets (we know exact size N_unique)
        ds_gw_scalars = grp_gw.create_dataset('scalars', (n_unique, 7), dtype='f4')
        ds_gw_skymaps = grp_gw.create_dataset('skymaps', (n_unique, 6, 19200), dtype='f4') # 6 channels after cleaning
        # Store IDs as fixed-length ASCII strings
        dt_str = h5py.special_dtype(vlen=str) 
        ds_gw_ids = grp_gw.create_dataset('ids', (n_unique,), dtype=dt_str)
        
        # --- Group B: All Optical Data ---
        # We don't know total optical count yet, so we use resizable datasets (chunked)
        grp_opt = f.create_group('events/optical_data')
        
        chunk_size = 1024
        ds_opt_vals = grp_opt.create_dataset('values', (0, MAX_LC_LENGTH, NUM_BANDS), 
                                             maxshape=(None, MAX_LC_LENGTH, NUM_BANDS), dtype='f4', chunks=(chunk_size, MAX_LC_LENGTH, NUM_BANDS))
        ds_opt_errs = grp_opt.create_dataset('errors', (0, MAX_LC_LENGTH, NUM_BANDS),
                                             maxshape=(None, MAX_LC_LENGTH, NUM_BANDS), dtype='f4', chunks=(chunk_size, MAX_LC_LENGTH, NUM_BANDS))
        ds_opt_masks = grp_opt.create_dataset('masks', (0, MAX_LC_LENGTH, NUM_BANDS), 
                                              maxshape=(None, MAX_LC_LENGTH, NUM_BANDS), dtype='f4', chunks=(chunk_size, MAX_LC_LENGTH, NUM_BANDS))
        ds_opt_times = grp_opt.create_dataset('times', (0, MAX_LC_LENGTH), 
                                              maxshape=(None, MAX_LC_LENGTH), dtype='f4', chunks=(chunk_size, MAX_LC_LENGTH))
        
        # Parent Index Mapping (The Relation)
        ds_parent_idx = grp_opt.create_dataset('parent_gw_idx', (0,), maxshape=(None,), dtype='i4', chunks=(chunk_size,))
        
        # --- Processing Loop ---
        print("Starting processing loop...")
        total_optical_count = 0
        
        # Buffer for optical data to reduce HDF5 resize calls (optimization)
        opt_buffer_vals = []
        opt_buffer_errs = []
        opt_buffer_masks = []
        opt_buffer_times = []
        opt_buffer_p_idx = []
        BUFFER_LIMIT = 5000 

        def flush_buffer():
            nonlocal total_optical_count, opt_buffer_vals, opt_buffer_errs, opt_buffer_masks, opt_buffer_times, opt_buffer_p_idx
            if len(opt_buffer_vals) == 0: return
            
            n_new = len(opt_buffer_vals)
            current_size = total_optical_count
            new_size = current_size + n_new
            
            # Resize datasets
            ds_opt_vals.resize(new_size, axis=0)
            ds_opt_errs.resize(new_size, axis=0)
            ds_opt_masks.resize(new_size, axis=0)
            ds_opt_times.resize(new_size, axis=0)
            ds_parent_idx.resize(new_size, axis=0)
            
            # Write data
            ds_opt_vals[current_size:new_size] = np.array(opt_buffer_vals)
            ds_opt_errs[current_size:new_size] = np.array(opt_buffer_errs)
            ds_opt_masks[current_size:new_size] = np.array(opt_buffer_masks)
            ds_opt_times[current_size:new_size] = np.array(opt_buffer_times)
            ds_parent_idx[current_size:new_size] = np.array(opt_buffer_p_idx)
            
            total_optical_count += n_new
            
            # Clear buffer
            opt_buffer_vals = []
            opt_buffer_errs = []
            opt_buffer_masks = []
            opt_buffer_times = []
            opt_buffer_p_idx = []

        # Iterate over unique GW events
        for gw_idx, row in tqdm(gw_df.iterrows(), total=n_unique):
            # if gw_idx > 1:
            #     break
            event_id = int(row['simulation_id'])
            # print(f"Processing GW Event {event_id} ({gw_idx+1}/{n_unique})...")
            
            # 1. Process & Save GW Data
            # Scalars (Columns m1...param14)
            # Adjust columns based on your CSV
            gw_params_name = ['mass1_detector', 'mass2_detector', 'spin1z', 'spin2z', 'inclination', 'distmean', 'diststd']
            scalars = row[gw_params_name].values.astype(np.float32)
            ds_gw_scalars[gw_idx] = scalars
            ds_gw_ids[gw_idx] = str(event_id)
            
            # Skymap
            # Apply robust preprocessing (Returns Tensor [6, 19200])
            gw_mocmap = sample_moc_skymap(f"/fred/oz016/bgao_kn/data/bns_skymap/{event_id}.fits")
            ds_gw_skymaps[gw_idx] = gw_mocmap.numpy() # Convert back to numpy for HDF5
            
            # 2. Process Optical Data
            # Extract light curves from SNANA FITS
            lcs = parse_snana_fits(event_id, sim_dir=fits_dir)
            
            # Add to buffer
            for (vals, errs, masks, times) in lcs:
                opt_buffer_vals.append(vals)
                opt_buffer_errs.append(errs)
                opt_buffer_masks.append(masks)
                opt_buffer_times.append(times)
                opt_buffer_p_idx.append(gw_idx) # Link to parent GW index
            
            # Flush if buffer is full
            if len(opt_buffer_vals) >= BUFFER_LIMIT:
                flush_buffer()
        
        # Final flush
        flush_buffer()
        
        # Save metadata
        f.attrs['n_unique_gw'] = n_unique
        f.attrs['n_total_optical'] = total_optical_count
        print(f"\nProcessing Complete.")
        print(f"Unique GW Events: {n_unique}")
        print(f"Total Light Curves: {total_optical_count}")
        print(f"Saved to: {output_h5_path}")

# Functions for parsing SNANA FITS files for negative samples
def parse_snana_fits_neg(head_path, phot_path, type="SN"):
    '''
    Parses HEAD.fits and PHOT.fits.
    Extracts multiple light curve realizations for a kind of transients.
    
    Args:
        head_path: Path to HEAD FITS file.
        phot_path: Path to PHOT FITS file.
        type: Type of transient ('SN', 'TDE', 'AGN', 'uLens', 'dwarf-nova')
        
    Returns:
        List of tuples: [(values, masks, times), ...]
        Returns empty list if files are missing.
    '''
    allowed_types = ['SN', 'TDE', 'AGN', 'uLens', 'dwarf-nova']
    if type not in allowed_types:
        raise ValueError(f"Type must be one of {allowed_types}")

    if not os.path.exists(head_path) or not os.path.exists(phot_path):
        print(f"Warning: FITS files not found for {head_path}")
        return []

    try:
        # Open FITS files
        with fits.open(head_path) as hdul_head, fits.open(phot_path) as hdul_phot:
            # Usually data is in extension 1
            data_head = hdul_head[1].data
            data_phot = hdul_phot[1].data
            
            # Use columns directly (Astropy FITS columns are case-insensitive usually)
            # HEAD columns
            
            ptrobs_min = data_head['PTROBS_MIN']
            ptrobs_max = data_head['PTROBS_MAX']
            
            # PHOT columns
            mjd_all = data_phot['MJD']
            flux_all = data_phot['FLUXCAL']
            fluxerr_all = data_phot['FLUXCALERR'] # Optional usage
            flt_all = data_phot['BAND'] # Filters

            extracted_lcs = []
            
            # Iterate over each realization in HEAD
            for i in range(len(data_head)):
                # SNANA uses 1-based indexing for pointers, Python uses 0-based
                # Start index: value - 1
                # End index: value (exclusive in python slicing)
                start_idx = ptrobs_min[i] - 1
                end_idx = ptrobs_max[i]
                nobs = data_head['NOBS'][i]
                if nobs < 5:
                    # print(f"Warning: Light curve for {head_path} realization {i} has less than 5 observations. Skipping.")
                    continue  # Skip light curves with less than 5 observations

                # get coordinates
                ra = data_head['RA'][i]
                dec = data_head['DEC'][i]
                coordinates = np.array([ra, dec], dtype=np.float32)
                
                # Slicing the PHOT data
                lc_mjd = mjd_all[start_idx : end_idx]
                lc_flux = flux_all[start_idx : end_idx]
                lc_fluxerr = fluxerr_all[start_idx : end_idx]
                lc_flt = flt_all[start_idx : end_idx]

                # Normalization
                std = np.std(lc_flux)
                mean = np.mean(lc_flux)
                lc_flux = (lc_flux - mean) / (std + 1e-8)
                lc_fluxerr = lc_fluxerr / (std + 1e-8)
                
                # --- Format Conversion (to Tensor-ready numpy) ---
                val_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)    # Values matrix (flux)
                err_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)    # Errors matrix (flux errors)
                mask_mat = np.zeros((MAX_LC_LENGTH, NUM_BANDS), dtype=np.float32)
                time_vec = np.zeros((MAX_LC_LENGTH,), dtype=np.float32)
                
                # 1. Time Normalization (Relative to pseudo-explosion time)
                if len(lc_mjd) > 0:
                    use_peak = type in ['SN', 'uLens', 'dwarf-nova', 'TDE']
                    if use_peak and 'PEAKMJD' in data_head.columns.names:
                        pesudo_mjd_explode = data_head['PEAKMJD'][i] - np.random.uniform(0.5, 5.0)
                    else:
                        pesudo_mjd_explode = np.random.uniform(lc_mjd.min(), lc_mjd.max())
                    rel_times = (lc_mjd - pesudo_mjd_explode) / 100  # Scale down to manageable range[-0.3, 0.6]
                    time_mask = np.where((rel_times >= -0.3) & (rel_times <= 0.6))[0]
                    lc_mjd = lc_mjd[time_mask]
                    lc_flux = lc_flux[time_mask]
                    lc_fluxerr = lc_fluxerr[time_mask]
                    lc_flt = lc_flt[time_mask]
                    rel_times = rel_times[time_mask]
                else:
                    continue # Skip empty light curves

                # 2. Fill Matrices
                # Truncate if longer than MAX_LC_LENGTH
                seq_len = min(len(lc_mjd), MAX_LC_LENGTH)
                if len(lc_mjd) > MAX_LC_LENGTH:
                    print(f"Warning: Light curve for realization {i} exceeds MAX_LC_LENGTH. Truncating.")
                    # Keep the MAX_LC_LENGTH points with smallest absolute rel_times
                    sorted_indices = np.argsort(np.abs(rel_times))[:MAX_LC_LENGTH]
                    sorted_indices = np.sort(sorted_indices)  # Sort back to chronological order
                    lc_mjd = lc_mjd[sorted_indices]
                    lc_flux = lc_flux[sorted_indices]
                    lc_fluxerr = lc_fluxerr[sorted_indices]
                    lc_flt = lc_flt[sorted_indices]
                    rel_times = rel_times[sorted_indices]
                
                for t in range(seq_len):
                    band_char = lc_flt[t].strip() # Remove whitespace
                    if band_char in NEG_BAND_MAP:
                        b_idx = NEG_BAND_MAP[band_char]
                        
                        val_mat[t, b_idx] = lc_flux[t]
                        err_mat[t, b_idx] = lc_fluxerr[t]
                        mask_mat[t, b_idx] = 1.0
                        time_vec[t] = rel_times[t]
                
                extracted_lcs.append((val_mat, err_mat, mask_mat, time_vec, coordinates))
                
            return extracted_lcs

    except Exception as e:
        print(f"Error processing FITS for {head_path}: {e}")
        return []

# Function to create negative dataset HDF5
def create_negative_dataset(sim_root, output_h5_path, buffer_limit=5000):
    sim_root = Path(sim_root)

    # Decompress FITS.gz files (exclude KN) and remove .gz
    gz_files = [p for p in sim_root.rglob("*.FITS.gz") if "ELASTICC2_TRAIN_02_KN_" not in p.as_posix()]
    if len(gz_files) > 0:
        for gz_path in tqdm(gz_files, desc="Decompressing FITS.gz"):
            subprocess.run(["gunzip", "-f", str(gz_path)], check=False)

    head_files = [p for p in sim_root.rglob("*_HEAD.FITS") if "ELASTICC2_TRAIN_02_KN_" not in p.as_posix()]
    head_files = sorted(head_files)
    if len(head_files) == 0:
        raise FileNotFoundError(f"No HEAD.FITS files found under {sim_root}")

    with h5py.File(output_h5_path, 'w') as f:
        grp_opt = f.create_group('ELASTICC2_TRAIN/optical_data')
        chunk_size = 1024
        ds_opt_vals = grp_opt.create_dataset(
            'values', (0, MAX_LC_LENGTH, NUM_BANDS),
            maxshape=(None, MAX_LC_LENGTH, NUM_BANDS),
            dtype='f4', chunks=(chunk_size, MAX_LC_LENGTH, NUM_BANDS)
        )
        ds_opt_errs = grp_opt.create_dataset(
            'errors', (0, MAX_LC_LENGTH, NUM_BANDS),
            maxshape=(None, MAX_LC_LENGTH, NUM_BANDS),
            dtype='f4', chunks=(chunk_size, MAX_LC_LENGTH, NUM_BANDS)
        )
        ds_opt_masks = grp_opt.create_dataset(
            'masks', (0, MAX_LC_LENGTH, NUM_BANDS),
            maxshape=(None, MAX_LC_LENGTH, NUM_BANDS),
            dtype='f4', chunks=(chunk_size, MAX_LC_LENGTH, NUM_BANDS)
        )
        ds_opt_times = grp_opt.create_dataset(
            'times', (0, MAX_LC_LENGTH),
            maxshape=(None, MAX_LC_LENGTH),
            dtype='f4', chunks=(chunk_size, MAX_LC_LENGTH)
        )
        ds_opt_coords = grp_opt.create_dataset(
            'coordinates', (0, 2),
            maxshape=(None, 2),
            dtype='f4', chunks=(chunk_size, 2)
        )
        dt_str = h5py.special_dtype(vlen=str)
        ds_opt_types = grp_opt.create_dataset(
            'types', (0,), maxshape=(None,), dtype=dt_str, chunks=(chunk_size,)
        )

        total_optical = 0
        opt_buffer_vals = []
        opt_buffer_errs = []
        opt_buffer_masks = []
        opt_buffer_times = []
        opt_buffer_coords = []
        opt_buffer_types = []

        def flush_buffer():
            nonlocal total_optical, opt_buffer_vals, opt_buffer_errs, opt_buffer_masks, opt_buffer_times, opt_buffer_coords, opt_buffer_types
            if len(opt_buffer_vals) == 0:
                return
            n_new = len(opt_buffer_vals)
            current_size = total_optical
            new_size = current_size + n_new

            ds_opt_vals.resize(new_size, axis=0)
            ds_opt_errs.resize(new_size, axis=0)
            ds_opt_masks.resize(new_size, axis=0)
            ds_opt_times.resize(new_size, axis=0)
            ds_opt_coords.resize(new_size, axis=0)
            ds_opt_types.resize(new_size, axis=0)

            ds_opt_vals[current_size:new_size] = np.array(opt_buffer_vals)
            ds_opt_errs[current_size:new_size] = np.array(opt_buffer_errs)
            ds_opt_masks[current_size:new_size] = np.array(opt_buffer_masks)
            ds_opt_times[current_size:new_size] = np.array(opt_buffer_times)
            ds_opt_coords[current_size:new_size] = np.array(opt_buffer_coords)
            ds_opt_types[current_size:new_size] = np.array(opt_buffer_types, dtype=object)

            total_optical += n_new
            opt_buffer_vals = []
            opt_buffer_errs = []
            opt_buffer_masks = []
            opt_buffer_times = []
            opt_buffer_coords = []
            opt_buffer_types = []

        for head_path in tqdm(head_files):
            name_lower = head_path.parent.name.lower()
            if 'kn' in name_lower:
                continue
            if 'agn' in name_lower:
                transient_type = 'AGN'
            elif 'tde' in name_lower:
                transient_type = 'TDE'
            elif 'ulens' in name_lower:
                transient_type = 'uLens'
            elif 'dwarf-nova' in name_lower:
                transient_type = 'dwarf-nova'
            elif 'sn' in name_lower or 'slsn' in name_lower or 'pisn' in name_lower:
                transient_type = 'SN'
            else:
                print(f"Skipping unsupported type folder: {head_path.parent.name}")
                continue

            phot_path = head_path.with_name(head_path.name.replace('_HEAD.FITS', '_PHOT.FITS'))
            if not phot_path.exists():
                print(f"Missing PHOT file for {head_path.name}")
                continue

            lcs = parse_snana_fits_neg(str(head_path), str(phot_path), type=transient_type)
            for (vals, errs, masks, times, coordinates) in lcs:
                opt_buffer_vals.append(vals)
                opt_buffer_errs.append(errs)
                opt_buffer_masks.append(masks)
                opt_buffer_times.append(times)
                opt_buffer_coords.append(coordinates)
                opt_buffer_types.append(transient_type)

            if len(opt_buffer_vals) >= buffer_limit:
                flush_buffer()

        flush_buffer()
        f.attrs['n_total_optical'] = total_optical
        print(f"Total negative light curves: {total_optical}")
        print(f"Saved to: {output_h5_path}")
