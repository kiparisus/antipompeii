#!/usr/bin/env python3

#####################################
############ ANTIPOMPEII ############
########## INITIALIZATION ###########
#####################################

import argparse
import sys
from pathlib import Path

# Add the project root to sys.path so ``python -m src.antipompeii.main`` and
# direct invocation both resolve the ``src.antipompeii.*`` package imports.
project_root = Path(__file__).parent.parent
if project_root not in sys.path:
    sys.path.append(str(project_root))

from src.antipompeii.ui.cli import antipompeiiCLI, DEFAULT_CONFIG
from src.antipompeii.utils.logger import setup_logger


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="antipompeii",
        description=(
            "ANTIPOMPEII — urban vulnerability assessment and resilience tool. "
            "Runs interactively by default; pass --mode 4 to drive a "
            "non-interactive run from a YAML configuration."
        ),
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to a YAML configuration file (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "-m", "--mode",
        type=int,
        choices=(1, 2, 3, 4, 5),
        default=None,
        help="Override the operational mode set in the YAML.",
    )
    return parser.parse_args(argv)


def main():
    """Entry point for the ANTIPOMPEII application."""
    args = _parse_args()
    logger = setup_logger()
    logger.info("Starting ANTIPOMPEII")

    try:
        cli = antipompeiiCLI(config_path=args.config, mode_override=args.mode)
        cli.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n\nExiting...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)
    finally:
        logger.info("Application shutdown")


if __name__ == "__main__":
    main()
