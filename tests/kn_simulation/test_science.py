import pandas as pd
from config import PIPELINE_ROOT
from rubin_too import classify_event, load_too_config
from snana import gen_input


def test_low_snr_is_baseline_not_filtered():
    config = load_too_config(PIPELINE_ROOT / "config" / "rubin_too_2024.yaml")
    decision = classify_event(7.5, 20.0, config)
    assert decision.mode == "baseline"
    assert decision.strategy is None


def test_snana_input_uses_only_prepared_catalog_fields():
    catalog = pd.DataFrame(
        {
            "simulation_id": [7],
            "trigger_mjd": [62000.0],
            "viewing_costheta": [0.5],
            "phi_deg": [30.0],
            "snana_mej_dynamic": [0.01],
            "snana_mej_wind": [0.02],
        }
    )
    template = (
        "MJD_EXPLODE: 0\nGENPEAK_COSTHETA: 0\nGENPEAK_MEJDYN: 0\n"
        "GENPEAK_MEJWIND: 0\nGENPEAK_PHI: 0"
    )
    output = gen_input(catalog, template, 7, gw_type="bns")
    assert "MJD_EXPLODE:  62000.0" in output
    assert "GENPEAK_MEJDYN:  0.01" in output
    assert "GENPEAK_MEJWIND:  0.02" in output
