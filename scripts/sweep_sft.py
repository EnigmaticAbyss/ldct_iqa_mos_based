from __future__ import annotations

from scripts.sweep_common import main_for_sweep, run_sweep


def main() -> None:
    """
    Run the common sweep CLI configured for SFT experiments.

    This entrypoint only supplies SFT-specific help text; sweep execution is
    handled by ``scripts.sweep_common.main_for_sweep``.
    """
    main_for_sweep(
        description="Run multiple SFT trainings, evaluate each output, and compare metrics.",
        config_help="Path to SFT sweep config JSON",
    )


if __name__ == "__main__":
    main()
