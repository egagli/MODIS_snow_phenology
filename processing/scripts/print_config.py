"""Print config parameters for a given config file (no secrets)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from modis_snow_phenology.config import Config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config-file", default="config/config_v1.txt")
    return p.parse_args()


def main():
    args = parse_args()
    c = Config(args.config_file)
    print(f"Config file          : {args.config_file}")
    print(f"config_name          : {c.config_name}")
    print(f"version              : {c.version}")
    print(f"azure_storage_account: {c.azure_storage_account}")
    print(f"azure_container      : {c.azure_container}")
    print(f"icechunk_prefix      : {c.icechunk_prefix}")
    print(f"tile_status_path     : {c.tile_status_path}")
    print(f"wy_start             : {c.wy_start}")
    print(f"wy_end               : {c.wy_end}")
    print(f"shard_shape          : {c.shard_shape}")
    print(f"inner_chunk_shape    : {c.inner_chunk_shape}")


if __name__ == "__main__":
    main()
