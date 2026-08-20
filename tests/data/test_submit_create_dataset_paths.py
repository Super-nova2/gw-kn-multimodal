from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "Model"
    / "scripts"
    / "data"
    / "submit_create_dataset_bns_nsbh.sh"
)


def test_dataset_submit_defaults_follow_current_production_layout():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "production_rubin_dual" not in text
    assert "runs_dual" not in text
    assert "_seed_42" not in text
    assert "_DUAL" not in text
    assert "production_am_bayestar/dual/bns_train_seed_1234/pos_catalog.csv" in text
    assert "production_am_bayestar/dual/nsbh_test_seed_1234/pos_catalog.csv" in text
    assert "kn_simulation/runs/bns_train/success_sim_ids.txt" in text
    assert "kn_simulation/runs/nsbh_test/success_sim_ids.txt" in text
    assert "SNDATA_ROOT/SIM/LSST_KN_BNS_TRAIN" in text
    assert "SNDATA_ROOT/SIM/LSST_KN_NSBH_TEST" in text


def test_dataset_submit_uses_all_latest_type1_test_events():
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'BNS_MAX_NEG_TYPE1_GW="${BNS_MAX_NEG_TYPE1_GW:-1000}"' in text
    assert 'NSBH_MAX_NEG_TYPE1_GW="${NSBH_MAX_NEG_TYPE1_GW:-1000}"' in text
    assert 'BNS_MAX_NEG_GW="${BNS_MAX_NEG_GW:-2000}"' in text
    assert 'NSBH_MAX_NEG_GW="${NSBH_MAX_NEG_GW:-1500}"' in text
