#!/usr/bin/env python3
"""
sync_data_to_r2.py — Sync the frontend data bundle to R2.

Uploads web/public/data/ (players_index.json, shards, manifest, meta, classes)
to the mlbdraftcol R2 bucket under a /data/ prefix so the Vercel deployment
can serve data from Cloudflare's CDN.

Requires rclone configured with an 'r2' remote pointing to the mlbdraftcol bucket.

Usage:
  # First-time setup (one-time):
  #   1. brew install rclone
  #   2. rclone config (choose S3-compatible, endpoint = R2 endpoint)
  #      - type: s3
  #      - provider: Cloudflare
  #      - env_auth: false
  #      - access_key_id: <your-r2-token-id>
  #      - secret_access_key: <your-r2-token-secret>
  #      - region: auto
  #      - endpoint: https://<accountid>.r2.cloudflarestorage.com
  #      - name it 'r2'

  # Then run:
  python3 scripts/sync_data_to_r2.py

  # Set the following env var in Vercel dashboard:
  #   NEXT_PUBLIC_DATA_BASE = https://pub-<r2-public-bucket-id>.r2.dev
"""

import subprocess, sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "web" / "public" / "data"
R2_PATH = "r2:mlbdraftcol/data/"

FILES = [
    "players_index.json",
    "meta.json",
    "models_manifest.json",
]
CLASSES = [f"classes/{y}.json" for y in range(2021, 2027)]
SHARDS = [f"players/shard-{i:02d}.json" for i in range(64)]


def main():
    all_files = FILES + CLASSES + SHARDS
    print(f"Syncing {len(all_files)} files from {DATA_DIR} to {R2_PATH}...")

    total_size = 0
    for f in all_files:
        fp = DATA_DIR / f
        if fp.exists():
            total_size += fp.stat().st_size

    print(f"Total data size: {total_size / 1024 / 1024:.1f} MB")
    print()

    # Build rclone copy command
    cmd = ["rclone", "copy", str(DATA_DIR) + "/", R2_PATH,
           "--include", "players_index.json",
           "--include", "meta.json",
           "--include", "models_manifest.json",
           "--include", "classes/*.json",
           "--include", "players/shard-*.json",
           "--progress"]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\nSync complete!")
        print(f"\nSet this in Vercel dashboard → Project Settings → Environment Variables:")
        print(f"  NEXT_PUBLIC_DATA_BASE = https://pub-<bucket-id>.r2.dev")
        print(f"\nThen redeploy from Vercel dashboard or push to main.")
    else:
        print(f"\nSync failed (exit code {result.returncode})")
        print("Is rclone installed and configured?")
        print("  brew install rclone && rclone config")


if __name__ == "__main__":
    main()
