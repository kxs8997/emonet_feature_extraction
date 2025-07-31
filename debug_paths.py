#!/usr/bin/env python3
"""
Debug script to check path mappings between annotations and actual image files
"""

import json
import os
from pathlib import Path

# Load annotations
with open('annotations.json', 'r') as f:
    annotations = json.load(f)

print(f"Total annotations: {len(annotations)}")

# Check first few entries
sample_paths = list(annotations.keys())[:5]
print(f"\nSample annotation paths:")
for path in sample_paths:
    print(f"  {path}")

# Check dataset root
dataset_root = Path(os.environ.get('AFEW_VA_ROOT', 
    "/home/ksubramanian/.cache/kagglehub/datasets/hoanguyensgu/afew-va/versions/1/AFEW-VA"))

print(f"\nDataset root: {dataset_root}")
print(f"Dataset root exists: {dataset_root.exists()}")

if dataset_root.exists():
    # Check first video directory
    video_001 = dataset_root / "001"
    print(f"\nVideo 001 directory: {video_001}")
    print(f"Video 001 exists: {video_001.exists()}")
    
    if video_001.exists():
        files = list(video_001.glob("*"))[:5]
        print(f"First 5 files in 001/:")
        for f in files:
            print(f"  {f.name}")
    
    # Test path mapping
    print(f"\nTesting path mappings:")
    for sample_path in sample_paths[:3]:
        # Try original path
        full_path = dataset_root / sample_path
        print(f"  {sample_path} -> {full_path} (exists: {full_path.exists()})")
        
        # Try with .png extension
        png_path = sample_path.replace('.jpg', '.png')
        full_png_path = dataset_root / png_path
        print(f"  {png_path} -> {full_png_path} (exists: {full_png_path.exists()})")

# Count existing files
existing_count = 0
for img_path in sample_paths[:100]:  # Check first 100 for speed
    # Try .png first
    png_path = img_path.replace('.jpg', '.png')
    full_png_path = dataset_root / png_path
    if full_png_path.exists():
        existing_count += 1
        continue
    
    # Try original path
    full_path = dataset_root / img_path
    if full_path.exists():
        existing_count += 1

print(f"\nOut of first 100 annotations, {existing_count} files exist")
