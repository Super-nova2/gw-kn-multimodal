import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def coords_to_unit_vectors(coords_deg):
    """
    Convert (RA, Dec) in degrees to the same XYZ convention used in the stored skymaps.
    The skymaps were created from healpy theta/phi using:
    x = cos(theta) * cos(phi), y = cos(theta) * sin(phi), z = sin(theta).
    This implies theta is colatitude, so we match with sin(dec) for x/y and cos(dec) for z.
    """
    ra = np.deg2rad(coords_deg[:, 0])
    dec = np.deg2rad(coords_deg[:, 1])
    x = np.sin(dec) * np.cos(ra)
    y = np.sin(dec) * np.sin(ra)
    z = np.cos(dec)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def compute_credible_levels(prob_mass):
    """
    Compute per-pixel credible level based on sorted probability mass.
    Returns a vector in [0, 1], where smaller means more likely.
    """
    prob = prob_mass.astype(np.float64)
    prob = prob / prob.sum()
    order = np.argsort(prob)[::-1]
    cumulative = np.cumsum(prob[order])
    credible = np.empty_like(prob)
    credible[order] = cumulative
    return credible.astype(np.float32)


def copy_optical_dataset(src_ds, dst_grp, name, n_opt, chunk_rows):
    src_shape = src_ds.shape
    dst_shape = (n_opt,) + src_shape[1:]
    if src_ds.chunks is not None:
        chunks = (min(chunk_rows, dst_shape[0]),) + src_ds.chunks[1:]
    else:
        chunks = (min(chunk_rows, dst_shape[0]),) + dst_shape[1:]

    dst_ds = dst_grp.create_dataset(
        name,
        shape=dst_shape,
        dtype=src_ds.dtype,
        chunks=chunks,
        compression=src_ds.compression
    )

    for start in range(0, n_opt, chunk_rows):
        end = min(start + chunk_rows, n_opt)
        dst_ds[start:end] = src_ds[start:end]

    return dst_ds


def main():
    parser = argparse.ArgumentParser(
        description="Create GW pixel features and credible levels for each optical sample."
    )
    parser.add_argument("--input_h5", type=str, required=True)
    parser.add_argument("--output_h5", type=str, required=True)
    parser.add_argument("--max_optical", type=int, default=None,
                        help="Use only the first N optical samples (for quick tests).")
    parser.add_argument("--chunk_rows", type=int, default=4096,
                        help="Chunk size for copying optical datasets.")

    args = parser.parse_args()

    input_path = Path(args.input_h5)
    output_path = Path(args.output_h5)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with h5py.File(input_path, 'r') as src, h5py.File(output_path, 'w') as dst:
        events_grp = dst.create_group('events')
        src.copy('events/gw_data', events_grp)

        opt_src = src['events/optical_data']
        opt_dst = events_grp.create_group('optical_data')

        n_opt_total = opt_src['values'].shape[0]
        n_opt = n_opt_total
        if args.max_optical is not None:
            n_opt = min(n_opt_total, int(args.max_optical))

        print(f"Copying optical datasets: {n_opt} / {n_opt_total} samples")
        copied = {}
        for name in ['values', 'errors', 'masks', 'times', 'coordinates', 'parent_gw_idx']:
            copied[name] = copy_optical_dataset(
                opt_src[name], opt_dst, name, n_opt, args.chunk_rows
            )

        pixel_ds = opt_dst.create_dataset(
            'gw_pixel_features',
            shape=(n_opt, 8),
            dtype='f4',
            chunks=(min(args.chunk_rows, n_opt), 8)
        )
        pixel_ds.attrs['feature_names'] = (
            'x,y,z,dA,dP,distmu,distsigma,credible_level'
        )

        parent_idx = copied['parent_gw_idx'][:]
        coords = copied['coordinates'][:]

        n_gw = src['events/gw_data/scalars'].shape[0]
        gw_to_opt = [[] for _ in range(n_gw)]
        for opt_idx, gw_idx in enumerate(parent_idx):
            gw_to_opt[int(gw_idx)].append(opt_idx)

        print("Computing matched pixel features...")
        for gw_idx in tqdm(range(n_gw)):
            opt_indices = gw_to_opt[gw_idx]
            if not opt_indices:
                continue

            opt_indices = np.asarray(opt_indices, dtype=np.int64)
            skymap = src['events/gw_data/skymaps'][gw_idx]
            pix_vectors = skymap[0:3].T
            tree = cKDTree(pix_vectors)

            opt_vectors = coords_to_unit_vectors(coords[opt_indices])
            _, pix_idx = tree.query(opt_vectors, k=1)

            credible = compute_credible_levels(skymap[4])
            pixel_feats = skymap[:, pix_idx].T
            pixel_feats = pixel_feats.astype(np.float32)
            cred_col = credible[pix_idx].reshape(-1, 1)
            gw_pixel = np.concatenate([pixel_feats, cred_col], axis=1)
            pixel_ds[opt_indices] = gw_pixel

        dst.attrs['source_h5'] = str(input_path)
        dst.attrs['gw_pixel_feature_names'] = (
            'x,y,z,dA,dP,distmu,distsigma,credible_level'
        )
        dst.attrs['optical_samples'] = n_opt

    print(f"Done. Saved to {output_path}")


if __name__ == "__main__":
    main()
