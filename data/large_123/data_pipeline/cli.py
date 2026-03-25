"""CLI entry point for the pipeline."""

import argparse
import sys

from .config import PipelineConfig
from .pipeline_factory import create_pipeline


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="Run the data pipeline")
    parser.add_argument("--input", type=str, default="data/input")
    parser.add_argument("--output", type=str, default="data/output")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args(argv)

    config = PipelineConfig(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
    )
    results = create_pipeline(config)
    print(f"Pipeline complete: {len(results)} records processed")


if __name__ == "__main__":
    main()
