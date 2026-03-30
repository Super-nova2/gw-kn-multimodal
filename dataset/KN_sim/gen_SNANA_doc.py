import opsimsummaryv2 as opsim
import numpy as np
import healpy as hp
# import ligo.skymap functions
import os
from ligo.skymap.io.fits import read_sky_map
from ligo.skymap.moc import uniq2nest, uniq2pixarea
import ligo.skymap.postprocess as postprocess
# import astropy functions
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from astropy.cosmology import z_at_value
import argparse
from pathlib import Path
import re
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR.parent
_BASE_DIR = os.environ.get('BASE_DIR', '/fred/oz016/bgao_kn')

# function to sample sky coordinates from MOC skymap
def r_peak(distmu, distsigma):
    """
    Calculate the peak distance from distmu and distsigma.(ansatz distribution)
    Note: (distnorm does not affect the peak position)
    """
    distmu = np.asarray(distmu, dtype=float)
    distsigma = np.asarray(distsigma, dtype=float)
    dist_peak = 0.5 * (distmu + np.sqrt(distmu**2 + 8.0 * distsigma**2))
    return dist_peak

def sample_sky_from_moc(mocmap, level=0.9, within=True):
    """
    Sample sky coordinates from a MOC skymap within/without a credible level.
    """

    uniq = mocmap['UNIQ']
    probdensity = mocmap['PROBDENSITY']
    distmu = mocmap['DISTMU']
    distsigma = mocmap['DISTSIGMA']

    # 1) UNIQ -> order, ipix, nside
    order, ipix = uniq2nest(uniq)

    # 2) caculate pixel area
    dA = uniq2pixarea(uniq)
    # 3) calculate pixel probability
    dP = probdensity * dA
    # 4) calculate credible levels
    cls = postprocess.find_greedy_credible_levels(
        dP, probdensity
    )
    cls = np.where(cls>1, 1.0, cls)

    if within:
        mask = cls <= level
        dP_cred = dP[mask]
        order_cred = order[mask]
        ipix_cred = ipix[mask]
    else:
        mask = cls >= level
        dP_cred = dP[mask]
        order_cred = order[mask]
        ipix_cred = ipix[mask]


    # 5) calculate RA, Dec for sampled pixels
    ras  = np.empty(len(dP_cred), dtype=float)
    decs = np.empty(len(dP_cred), dtype=float)

    # group by order to speed up
    for k in np.unique(order_cred):
        m = (order_cred == k)
        this_ipix  = ipix_cred[m]
        this_nside = 2 ** k
        theta, phi = hp.pix2ang(this_nside, this_ipix, nest=True)
        ras[m]  = np.degrees(phi)
        decs[m] = 90.0 - np.degrees(theta)
    
    # calculate distance with peak pdf from distmu and distsigma
    distmu = np.array(distmu[mask], dtype=float)
    distsigma = np.array(distsigma[mask], dtype=float)

    # remove nan values
    valid = np.isfinite(distmu) & np.isfinite(distsigma) & (distsigma>0)
    distmu = distmu[valid]
    distsigma = distsigma[valid]
    dist_peak = r_peak(distmu, distsigma)

    return ras[valid], decs[valid], dist_peak

def get_NLIBID(simlib_file):
    with open(simlib_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('NLIBID:'):
                nlibid = int(line.split()[1])
                break
    return nlibid

def gen_input(injections, text, sim_id, GW_type="bns"):
    """
    Generate SIMGEN INPUT file content by modifying the template text with GW parameters.
    """
    # get gw parameters
    idx = np.where(injections['simulation_id']==sim_id)[0]
    mjd_explode = injections['mjd_time'].iloc[idx].values
    costheta = injections['costheta'].iloc[idx].values
    phi = injections['phi'].iloc[idx].values
    mej_dyn = injections['mej_dyn'].iloc[idx].values
    mej_wind = injections['mej_wind'].iloc[idx].values

    # check if parameters are in model range
    if mej_dyn[0] < 0.001:
        mej_dyn[0] = 0.001
    elif mej_dyn[0] > 0.02:
        mej_dyn[0] = 0.02
    if mej_wind[0] < 0.01:
        mej_wind[0] = 0.01
    elif mej_wind[0] > 0.13:
        mej_wind[0] = 0.13

    # modify explosion time
    text = re.sub(
        r"^(MJD_EXPLODE:\s*)\S+.*$",
        rf"\1 {mjd_explode[0]}",
        text,
        flags=re.MULTILINE
    )
    # modify kilonova parameters
    text = re.sub(
        r"^(GENPEAK_COSTHETA:\s*)\S+.*$",
        rf"\1 {costheta[0]}",
        text,
        flags=re.MULTILINE
    )
    if GW_type=="bns":
        text = re.sub(
            r"^(GENPEAK_PHI:\s*)\S+.*$",
            rf"\1 {phi[0]}",
            text,
            flags=re.MULTILINE
        )
    text = re.sub(
        r"^(GENPEAK_MEJDYN:\s*)\S+.*$",
        rf"\1 {mej_dyn[0]}",
        text,
        flags=re.MULTILINE
    )
    text = re.sub(
        r"^(GENPEAK_MEJWIND:\s*)\S+.*$",
        rf"\1 {mej_wind[0]}",
        text,
        flags=re.MULTILINE
    )

    return text


parser = argparse.ArgumentParser(description="Generate SNANA SIMLIB for one GW event.")

parser.add_argument("--GW_type", type=str, default="bns", help="bns/nsbh")
parser.add_argument("--skymap_path", type=str, default=f"{_BASE_DIR}/data/bns_skymap/", help="Path to skymaps")
parser.add_argument("--sim_name", type=str, default="LSST_KN_BNS", help="Name of simulation, eg:LSST_KN_BNS/NSBH")
parser.add_argument("--sim_ids", nargs="+", type=int, required=True, help="a list of simulation_id, at most 10")
parser.add_argument("--GW_params", type=str, default=str(DATASET_DIR / "O5_sim_bns" / "injections_final.csv"), help="CSV file containing GW parameters")
parser.add_argument("--Opsim", type=str, default=f"{_BASE_DIR}/data/rubin_sim/baseline_v5.1/baseline_v5.1.1_10yrs.db", help="Opsim database file")
parser.add_argument("--within", action='store_true', help="sample within credible level")
parser.add_argument("--level", type=float, default=0.9, help="credible level to sample sky position")
parser.add_argument("--outdir", type=str, default="./data/", help="output directory for SIMLIB")
parser.add_argument("--template_input", type=str, default=str(SCRIPT_DIR / "Template_doc" / "SIMGEN_KN_LSST_TEMPLATE.INPUT"), help="template SIMGEN INPUT file")
args = parser.parse_args()

# check output directory
if not args.outdir.endswith('/'):
    args.outdir += '/'
# os.makedirs(args.outdir, exist_ok=True)
os.makedirs(args.outdir + "SIM_INPUT/", exist_ok=True)
os.makedirs(args.outdir + "SIMLIB/", exist_ok=True)
simlib_dir = args.outdir + "SIMLIB/"
input_dir = args.outdir + "SIM_INPUT/"

# derive SIMLIB filename prefix from the OpSim database filename stem
# (opsimsummaryv2 names the output file as <db_stem><file_suffix>.SIMLIB)
opsim_stem = Path(args.Opsim).stem  # e.g. "baseline_v5.1.1_10yrs"

# load GW parameters
injections = pd.read_csv(args.GW_params)
template_input_path = Path(args.template_input)
template_text = template_input_path.read_text()

# Load OpSim survey
OpSimSurv = opsim.OpSimSurvey(args.Opsim)   #take about 3minutes to load the database

# Compute the healpy representation of the survey, taking about 2 minutes
# with a cut to a minimum of 500 and a maximum of 10000 visits
nside = 256
OpSimSurv.compute_hp_rep(nside=nside, minVisits=1, maxVisits=10000)    # nest=False

sim_ids = args.sim_ids
sim_name = args.sim_name

for sim_id in sim_ids:
    print(f"\nProcessing simulation ID: {sim_id}")
    # MOC skymap for high resolution
    skymap = read_sky_map(f'{args.skymap_path}{sim_id}.fits', moc=True)
    # Load with nest map
    nest_map, meta = read_sky_map(f'{args.skymap_path}{sim_id}.fits', nest=True)
    distmean = meta.get('distmean')
    diststd = meta.get('diststd')
    print("Simulation ID:", sim_id)
    print("Distance mean:", meta.get('distmean'), "Mpc\nDistance std:", meta.get('diststd'), 'Mpc')

    # Sample sky position within 90% credible region
    ra, dec, dist_peak = sample_sky_from_moc(skymap, level=args.level, within=args.within)
    print("Range of dist_peak(Mpc):", dist_peak.min(), dist_peak.max())
    redshift = z_at_value(cosmo.luminosity_distance, dist_peak * u.Mpc).value

    OpSimSurv.sample_coordinates(ra, dec, redshift, nsides=nside, is_deg=True)

    # Writing simlib
    sim = opsim.sim_io.SNANA_Simlib(OpSimSurv, out_path=simlib_dir, file_suffix=f"_{sim_name}_{sim_id}")
    sim.write_SIMLIB()  # taking about 2 minutes
    print(f"SIMLIB for simulation ID {sim_id} written to {simlib_dir}")

    # generate corresponding SIMGEN INPUT file
    NLIBID = get_NLIBID(os.path.join(simlib_dir, f"{opsim_stem}_{sim_name}_{sim_id}.SIMLIB"))
    print(f"NLIBID for simulation ID {sim_id}: {NLIBID}")

    # replace NLIBID and GENVERSION in the template
    text = template_text
    text = re.sub(
            r"^(NGENTOT_LC:\s*)\S+.*$",
            rf"\1 {NLIBID}",
            text,
            flags=re.MULTILINE
    )
    genversion_new = f"{sim_name}_{sim_id}"
    text = re.sub(
        r"^(GENVERSION:\s*)\S+.*$",
        rf"\1{genversion_new}",
        text,
        flags=re.MULTILINE
    )
    # MODIFY SIMLIB FILE (use raw SIMLIB, not COADD — keep per-visit resolution)
    simlib_file = f"{simlib_dir}{opsim_stem}_{sim_name}_{sim_id}.SIMLIB"
    text = re.sub(
        r"^(SIMLIB_FILE:\s*)\S+.*$",
        fr"\1{simlib_file}",
        text,
        flags=re.MULTILINE
    )

    # set unique RANSEED per event to avoid correlated noise realisations
    text = re.sub(
        r"^(RANSEED:\s*)\S+.*$",
        rf"\1 {100000 + sim_id}",
        text,
        flags=re.MULTILINE,
    )

    # modify other parameters based on GW parameters and simlib file
    text = gen_input(injections, text, sim_id, GW_type=args.GW_type)
    # save to file
    with open(input_dir + f"SIMGEN_{sim_name}_{sim_id}.INPUT", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"SIMGEN INPUT file for simulation ID {sim_id} written to {input_dir}")
