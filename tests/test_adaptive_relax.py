"""Tests for the adaptive lattice-relaxation fallback trigger."""

from cellnet.lattice_conf_pipeline import ConfQRSRecord, conf_sweep_had_hit


def _record(success_rate, relax_lattice=True):
    return ConfQRSRecord(
        dedup_idx=0,
        cellpar=[5.0, 6.0, 7.0, 90.0, 90.0, 90.0],
        success_rate=success_rate,
        time_min=1.0,
        workdir="/tmp/none",
        n_conformers=1,
        relax_lattice=relax_lattice,
    )


def test_empty_sweep_is_not_a_hit():
    assert conf_sweep_had_hit([]) is False


def test_all_zero_success_rates_is_not_a_hit():
    """The v9 outright-failure signature: every cell finished, none matched."""
    assert conf_sweep_had_hit([_record(0.0) for _ in range(72)]) is False


def test_errored_runs_are_not_hits():
    assert conf_sweep_had_hit([_record(None) for _ in range(4)]) is False


def test_mixed_none_and_zero_is_not_a_hit():
    assert conf_sweep_had_hit([_record(None), _record(0.0), _record(0.0)]) is False


def test_any_positive_success_rate_is_a_hit():
    assert conf_sweep_had_hit([_record(0.0), _record(0.0005), _record(None)]) is True


def test_single_weak_hit_counts():
    """OBEQIX-scale hits (~0.05 %) must still count as success."""
    assert conf_sweep_had_hit([_record(0.0005)]) is True


# --- all-errored guard and summary block -----------------------------------

from cellnet.lattice_conf_pipeline import (  # noqa: E402
    conf_sweep_all_errored,
    summarize_adaptive_sweeps,
)


def _errored(msg="CHARMM not found"):
    rec = _record(None)
    rec.error = msg
    return rec


def test_empty_sweep_is_not_all_errored():
    """An empty sweep carries no error information and must not be treated as broken."""
    assert conf_sweep_all_errored([]) is False


def test_all_errored_sweep_detected():
    assert conf_sweep_all_errored([_errored() for _ in range(5)]) is True


def test_one_finished_run_means_not_all_errored():
    """A single completed cell (even at SR = 0) means the setup works; the fallback should run."""
    assert conf_sweep_all_errored([_errored(), _errored(), _record(0.0)]) is False


def test_summary_block_recovered():
    primary = [_record(0.0) for _ in range(3)]
    fallback = [_record(0.0, relax_lattice=False), _record(0.0104, relax_lattice=False)]
    block = summarize_adaptive_sweeps(
        enabled=True,
        primary_relax_lattice=True,
        primary=primary,
        fallback=fallback,
        fallback_root="/x/adaptive_fixed_lattice",
    )
    assert block["primary_n_hit"] == 0
    assert block["fallback_ran"] is True
    assert block["fallback_relax_lattice"] is False
    assert block["fallback_n_hit"] == 1
    assert block["recovered_by_fallback"] is True
    assert block["fallback_skipped_reason"] is None


def test_summary_block_skipped_all_errored():
    primary = [_errored() for _ in range(2)]
    block = summarize_adaptive_sweeps(
        enabled=True,
        primary_relax_lattice=True,
        primary=primary,
        fallback=None,
        fallback_root=None,
        skipped_reason="every primary run errored",
    )
    assert block["primary_all_errored"] is True
    assert block["fallback_ran"] is False
    assert block["fallback_n_runs"] == 0
    assert block["recovered_by_fallback"] is False
    assert block["fallback_skipped_reason"] == "every primary run errored"


def test_summary_block_primary_hit_means_no_recovery_claim():
    primary = [_record(0.5)]
    block = summarize_adaptive_sweeps(
        enabled=True,
        primary_relax_lattice=True,
        primary=primary,
        fallback=None,
        fallback_root=None,
        skipped_reason="primary sweep matched",
    )
    assert block["primary_n_hit"] == 1
    assert block["recovered_by_fallback"] is False
