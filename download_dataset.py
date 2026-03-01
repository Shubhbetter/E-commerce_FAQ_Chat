"""Helper script to fetch datasets from Kaggle.

This is meant to satisfy the project requirement "choose one LLM dataset from
publicly available dataset repos".  We demonstrate downloading the e-commerce
FAQ dataset used elsewhere in this repo, but the script can fetch any Kaggle
`owner/dataset` identifier and save a single file from it.

Usage examples::

    # download the CSV only
    python download_dataset.py \
        --dataset saadmakhdoom/ecommerce-faq-chatbot-dataset \
        --file "Ecommerce_FAQs.csv"

You must have the `kaggle` Python package installed and either set the
`KAGGLE_USERNAME`/`KAGGLE_KEY` environment variables or placed a
`~/.kaggle/kaggle.json` file as described in the Kaggle API docs.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download a file from a Kaggle dataset")
    parser.add_argument("--dataset", required=True,
                        help="Kaggle dataset identifier, e.g. owner/name")
    parser.add_argument("--file", required=True,
                        help="Filename within the dataset to download")
    parser.add_argument("--output", default=".",
                        help="Directory to save the file")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # run the kaggle CLI; the user must have it installed and auth configured
        cmd = [
            "kaggle", "datasets", "download", "-d", args.dataset,
            "-f", args.file,
            "-p", str(out_dir),
            "--unzip",
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"Downloaded {args.file} into {out_dir}")
    except subprocess.CalledProcessError as e:
        print("Failed to invoke kaggle CLI. Ensure the package is installed and"
              " credentials are set.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
