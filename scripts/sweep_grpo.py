from __future__ import annotations

from scripts.sweep_common import main_for_sweep


def main() -> None:
    main_for_sweep(
        description="Run multiple GRPO trainings, evaluate each output, and compare metrics.",
        config_help="Path to GRPO sweep config JSON",
    )


if __name__ == "__main__":
    main()
