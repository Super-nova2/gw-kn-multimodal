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
    """
    def __init__(self, h5_path: str):
        super().__init__()
        self.h5_path = h5_path
        self.h5_file = None
        
        # Open file temporarily to get dataset length
        with h5py.File(h5_path, 'r') as f:
            self.length = f['events/optical_data/values'].shape[0]
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """
        Args:
            idx: Index of the light curve (optical data).
        """
        # Lazy loading: Open file only when needed (crucial for num_workers > 0)
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            
        # 1. Retrieve Optical Data (Values, Errors, Masks, Times)
        #    HDF5 structure: events/optical_data/...
        opt_val   = torch.from_numpy(self.h5_file['events/optical_data/values'][idx])
        opt_err   = torch.from_numpy(self.h5_file['events/optical_data/errors'][idx])
        opt_mask  = torch.from_numpy(self.h5_file['events/optical_data/masks'][idx])
        opt_time  = torch.from_numpy(self.h5_file['events/optical_data/times'][idx])
        opt_coords = torch.from_numpy(self.h5_file['events/optical_data/coordinates'][idx])
        
        # 2. Retrieve Parent GW Index
        gw_idx = self.h5_file['events/optical_data/parent_gw_idx'][idx]
        
        # 3. Retrieve Unique GW Data using gw_idx
        #    HDF5 structure: events/gw/...
        gw_scalar = torch.from_numpy(self.h5_file['events/gw_data/scalars'][gw_idx])
        gw_skymap = torch.from_numpy(self.h5_file['events/gw_data/skymaps'][gw_idx])
        
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

def create_training_dataloader(
    h5_path: str, 
    batch_size: int = 32, 
    steps_per_epoch: int = 1000, 
    num_workers: int = 4
):
    """
    Factory function to initialize the Dataset, Sampler, and DataLoader.
    """
    # 1. Build Index Map (Once)
    gw_map = build_gw_to_lc_mapping(h5_path)
    
    # 2. Initialize Dataset
    dataset = RelationalHDF5Dataset(h5_path)
    
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
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


# Constants
BAND_MAP = {'LSST-u': 0, 'LSST-g': 1, 'LSST-r': 2, 'LSST-i': 3, 'LSST-z': 4, 'LSST-Y': 5}
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
            gw_params_name = ['mass1', 'mass2', 'spin1z', 'spin2z', 'inclination', 'distmean', 'diststd']
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