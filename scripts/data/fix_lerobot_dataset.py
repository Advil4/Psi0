#!/usr/bin/env python3
"""
Comprehensive script to fix LeRobot datasets for training compatibility.

This script performs all necessary fixes to ensure datasets are compatible
with the LeRobot framework and PSI0 training pipeline.

Supported fixes:
1. Convert episodes.jsonl to LeRobot compatible format
2. Generate missing metadata files (episodes_stats.jsonl, modality.json, etc.)
3. Fix parquet file metadata (List -> Sequence type)
4. Rename parquet columns to match training configuration
5. Update info.json with correct feature names and paths
6. Rename video folders to match expected structure

Usage:
    # For Pick_up_an_apple dataset:
    python scripts/data/fix_lerobot_dataset.py --dataset Pick_up_an_apple
    
    # For custom dataset:
    python scripts/data/fix_lerobot_dataset.py --dataset Your_Dataset_Name
"""

import json
import shutil
import argparse
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np

# Configuration - will be set in main()
DATASET_DIR = None
META_DIR = None
DATA_DIR = None
VIDEO_DIR = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fix LeRobot dataset for training compatibility'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='Pick_up_an_apple',
        help='Dataset name (folder name under real/ directory)'
    )
    parser.add_argument(
        '--root-dir',
        type=str,
        default='/mnt/data0/xhx/Psi0/real',
        help='Root directory containing datasets'
    )
    return parser.parse_args()


def convert_episodes_jsonl():
    """Convert episodes.jsonl to LeRobot compatible format."""
    print("=" * 80)
    print("Step 1: Converting episodes.jsonl...")
    print("=" * 80)
    
    episodes_file = META_DIR / "episodes.jsonl"
    episodes_backup = META_DIR / "episodes_original.jsonl"
    
    # Backup original file
    if not episodes_backup.exists():
        shutil.copy(episodes_file, episodes_backup)
        print(f"  ✓ Backed up original episodes.jsonl")
    
    # Load task mappings
    tasks_file = META_DIR / "tasks.jsonl"
    task_to_index = {}
    if tasks_file.exists():
        with open(tasks_file, 'r') as f:
            for line in f:
                task_data = json.loads(line.strip())
                task_to_index[task_data['task']] = task_data['task_index']
        print(f"  ✓ Loaded {len(task_to_index)} task mappings")
    
    # Convert episodes
    converted_episodes = []
    with open(episodes_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            episode_data = json.loads(line.strip())
            
            # Extract task indices
            tasks = episode_data.get("tasks", [])
            task_indices = [task_to_index.get(task, 0) for task in tasks]
            
            converted = {
                "episode_index": episode_data.get("episode_index"),
                "tasks": task_indices,  # Use indices instead of strings
                "length": episode_data.get("length"),
                "dataset_from_index": episode_data.get("dataset_from_index", 0),
                "dataset_to_index": episode_data.get("dataset_to_index", 0),
            }
            
            # Handle chunk/file index from nested keys
            if "data/chunk_index" in episode_data:
                converted["chunk_index"] = episode_data["data/chunk_index"]
            if "data/file_index" in episode_data:
                converted["file_index"] = episode_data["data/file_index"]
            
            converted_episodes.append(converted)
            
            if line_num % 20 == 0:
                print(f"  Converted {line_num} episodes...")
    
    # Write converted episodes
    with open(episodes_file, 'w') as f:
        for episode in converted_episodes:
            f.write(json.dumps(episode) + '\n')
    
    print(f"  ✓ Converted {len(converted_episodes)} episodes")
    print(f"  ✓ Original backup saved to: {episodes_backup}")


def generate_episodes_stats():
    """Generate episodes_stats.jsonl from original episodes.jsonl."""
    print("\n" + "=" * 80)
    print("Step 2: Generating episodes_stats.jsonl...")
    print("=" * 80)
    
    episodes_backup = META_DIR / "episodes_original.jsonl"
    stats_file = META_DIR / "episodes_stats.jsonl"
    
    if not episodes_backup.exists():
        print("  ✗ No backup found, skipping stats generation")
        return
    
    episode_stats_list = []
    with open(episodes_backup, 'r') as f:
        for line in f:
            episode_data = json.loads(line.strip())
            episode_index = episode_data.get("episode_index")
            
            # Extract action and timestamp stats
            episode_stats = {
                "episode_index": episode_index,
                "stats": {
                    "action": {
                        "min": episode_data.get("stats/action/min", []),
                        "max": episode_data.get("stats/action/max", []),
                        "mean": episode_data.get("stats/action/mean", []),
                        "std": episode_data.get("stats/action/std", []),
                        "count": episode_data.get("stats/action/count", [])
                    },
                    "timestamp": {
                        "min": episode_data.get("stats/timestamp/min", []),
                        "max": episode_data.get("stats/timestamp/max", []),
                        "mean": episode_data.get("stats/timestamp/mean", []),
                        "std": episode_data.get("stats/timestamp/std", []),
                        "count": episode_data.get("stats/timestamp/count", [])
                    }
                }
            }
            episode_stats_list.append(episode_stats)
    
    with open(stats_file, 'w') as f:
        for stats in episode_stats_list:
            f.write(json.dumps(stats) + '\n')
    
    print(f"  ✓ Generated {len(episode_stats_list)} episode stats")


def generate_modality_json():
    """Generate modality.json file."""
    print("\n" + "=" * 80)
    print("Step 3: Generating modality.json...")
    print("=" * 80)
    
    modality_file = META_DIR / "modality.json"
    
    modality = {
        "state": {
            "names": ["dim"],
            "shape": [36],
            "dtype": "float32"
        },
        "action": {
            "names": ["dim"],
            "shape": [36],
            "dtype": "float32"
        },
        "video": {
            "names": ["c", "h", "w"],
            "shape": [3, 480, 640],
            "dtype": "uint8"
        }
    }
    
    with open(modality_file, 'w') as f:
        json.dump(modality, f, indent=2)
    
    print(f"  ✓ Generated modality.json")


def generate_relative_stats():
    """Generate empty relative_stats.json file."""
    print("\n" + "=" * 80)
    print("Step 4: Generating relative_stats.json...")
    print("=" * 80)
    
    relative_stats_file = META_DIR / "relative_stats.json"
    
    with open(relative_stats_file, 'w') as f:
        json.dump({}, f)
    
    print(f"  ✓ Generated relative_stats.json")


def generate_stats_psi0():
    """Generate stats_psi0.json from original episodes.jsonl."""
    print("\n" + "=" * 80)
    print("Step 5: Generating stats_psi0.json...")
    print("=" * 80)
    
    episodes_backup = META_DIR / "episodes_original.jsonl"
    stats_psi0_file = META_DIR / "stats_psi0.json"
    
    if not episodes_backup.exists():
        print("  ✗ No backup found, skipping stats_psi0 generation")
        return
    
    # Collect all action stats
    all_action_mins = []
    all_action_maxs = []
    
    with open(episodes_backup, 'r') as f:
        for line in f:
            episode_data = json.loads(line.strip())
            action_min = episode_data.get("stats/action/min", [])
            action_max = episode_data.get("stats/action/max", [])
            if action_min and action_max:
                all_action_mins.append(action_min)
                all_action_maxs.append(action_max)
    
    if not all_action_mins:
        print("  ✗ No action stats found")
        return
    
    # Calculate global min/max
    global_min = np.min(all_action_mins, axis=0).tolist()
    global_max = np.max(all_action_maxs, axis=0).tolist()
    
    stats_psi0 = {
        "action": {
            "min": global_min,
            "max": global_max
        },
        "states": {
            "min": global_min,  # Same as action for now
            "max": global_max
        }
    }
    
    with open(stats_psi0_file, 'w') as f:
        json.dump(stats_psi0, f, indent=2)
    
    print(f"  ✓ Generated stats_psi0.json")


def fix_parquet_metadata():
    """Fix parquet file metadata (List -> Sequence type)."""
    print("\n" + "=" * 80)
    print("Step 6: Fixing parquet metadata...")
    print("=" * 80)
    
    parquet_files = list(DATA_DIR.glob("**/*.parquet"))
    print(f"  Found {len(parquet_files)} parquet files")
    
    fixed_count = 0
    for i, parquet_file in enumerate(parquet_files, 1):
        try:
            table = pq.read_table(parquet_file)
            schema = table.schema
            
            if schema.metadata and b'huggingface' in schema.metadata:
                hf_metadata = json.loads(schema.metadata[b'huggingface'])
                
                modified = False
                if 'info' in hf_metadata and 'features' in hf_metadata['info']:
                    features = hf_metadata['info']['features']
                    
                    for field_name, feature_def in features.items():
                        if isinstance(feature_def, dict) and feature_def.get('_type') == 'List':
                            feature_def['_type'] = 'Sequence'
                            modified = True
                
                if modified:
                    new_hf_metadata = json.dumps(hf_metadata).encode('utf-8')
                    new_metadata = schema.metadata.copy()
                    new_metadata[b'huggingface'] = new_hf_metadata
                    new_schema = schema.with_metadata(new_metadata)
                    
                    temp_file = parquet_file.with_suffix('.tmp.parquet')
                    with pq.ParquetWriter(temp_file, new_schema) as writer:
                        writer.write_table(table)
                    temp_file.replace(parquet_file)
                    fixed_count += 1
            
            if i % 20 == 0 or i == len(parquet_files):
                print(f"  Processed {i}/{len(parquet_files)} files...")
        
        except Exception as e:
            print(f"  Error processing {parquet_file}: {e}")
    
    print(f"  ✓ Fixed {fixed_count} out of {len(parquet_files)} files")


def rename_parquet_columns():
    """Rename columns in parquet files (observation.state -> states)."""
    print("\n" + "=" * 80)
    print("Step 7: Renaming parquet columns...")
    print("=" * 80)
    
    parquet_files = list(DATA_DIR.glob("**/*.parquet"))
    print(f"  Found {len(parquet_files)} parquet files")
    
    column_mapping = {
        'observation.state': 'states',
    }
    
    fixed_count = 0
    for i, parquet_file in enumerate(parquet_files, 1):
        try:
            table = pq.read_table(parquet_file)
            
            needs_rename = any(old_name in table.column_names for old_name in column_mapping.keys())
            
            if not needs_rename:
                continue
            
            new_column_names = []
            for col_name in table.column_names:
                new_name = column_mapping.get(col_name, col_name)
                new_column_names.append(new_name)
            
            new_table = table.rename_columns(new_column_names)
            
            temp_file = parquet_file.with_suffix('.tmp.parquet')
            pq.write_table(new_table, temp_file)
            temp_file.replace(parquet_file)
            fixed_count += 1
            
            if i % 20 == 0 or i == len(parquet_files):
                print(f"  Processed {i}/{len(parquet_files)} files...")
        
        except Exception as e:
            print(f"  Error processing {parquet_file}: {e}")
    
    print(f"  ✓ Fixed {fixed_count} out of {len(parquet_files)} files")


def update_info_json():
    """Update info.json with correct feature names and paths."""
    print("\n" + "=" * 80)
    print("Step 8: Updating info.json...")
    print("=" * 80)
    
    info_file = META_DIR / "info.json"
    
    with open(info_file, 'r') as f:
        info = json.load(f)
    
    # Update data_path and video_path
    info['data_path'] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    info['video_path'] = "videos/chunk-{episode_chunk:03d}/egocentric/episode_{episode_index:06d}.mp4"
    
    # Add missing fields
    if 'total_videos' not in info:
        info['total_videos'] = info.get('total_episodes', 116)
    if 'total_chunks' not in info:
        info['total_chunks'] = 1
    
    # Update features
    features = info.get('features', {})
    
    # Rename observation.image to observation.images.egocentric
    if 'observation.image' in features:
        features['observation.images.egocentric'] = features.pop('observation.image')
        
        # Update shape and names
        features['observation.images.egocentric']['shape'] = [480, 640, 3]
        features['observation.images.egocentric']['names'] = ['height', 'width', 'channel']
        
        # Update video_info format
        if 'info' in features['observation.images.egocentric']:
            old_info = features['observation.images.egocentric'].pop('info')
            features['observation.images.egocentric']['video_info'] = {
                'video.fps': 30.0,
                'video.codec': old_info.get('video.codec', 'av1'),
                'video.pix_fmt': old_info.get('video.pix_fmt', 'yuv420p'),
                'video.is_depth_map': old_info.get('video.is_depth_map', False),
                'has_audio': old_info.get('has_audio', False)
            }
    
    # Rename observation.state to states
    if 'observation.state' in features:
        features['states'] = features.pop('observation.state')
    
    info['features'] = features
    
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"  ✓ Updated info.json")


def rename_video_folders():
    """Rename video folders to match expected structure."""
    print("\n" + "=" * 80)
    print("Step 9: Renaming video folders...")
    print("=" * 80)
    
    # Find all chunk directories
    chunk_dirs = list(VIDEO_DIR.glob("chunk-*"))
    
    renamed_count = 0
    for chunk_dir in chunk_dirs:
        old_video_dir = chunk_dir / "observation.image"
        new_video_dir = chunk_dir / "egocentric"
        
        if old_video_dir.exists() and not new_video_dir.exists():
            old_video_dir.rename(new_video_dir)
            renamed_count += 1
            print(f"  ✓ Renamed: {old_video_dir.name} -> {new_video_dir.name}")
    
    if renamed_count == 0:
        print(f"  ℹ No folders to rename (already correct)")
    else:
        print(f"  ✓ Renamed {renamed_count} video folders")


def main():
    """Run all fixes."""
    args = parse_args()
    
    # Configuration
    global DATASET_DIR, META_DIR, DATA_DIR, VIDEO_DIR
    DATASET_DIR = Path(args.root_dir) / args.dataset
    META_DIR = DATASET_DIR / "meta"
    DATA_DIR = DATASET_DIR / "data"
    VIDEO_DIR = DATASET_DIR / "videos"
    
    print("\n" + "=" * 80)
    print(f"Fixing LeRobot dataset: {args.dataset}")
    print("=" * 80)
    print(f"\nDataset directory: {DATASET_DIR}")
    print(f"This script will perform all necessary fixes...\n")
    
    # Validate dataset exists
    if not DATASET_DIR.exists():
        print(f"✗ ERROR: Dataset directory not found: {DATASET_DIR}")
        return 1
    
    try:
        # Step 1: Convert episodes.jsonl
        convert_episodes_jsonl()
        
        # Step 2: Generate episodes_stats.jsonl
        generate_episodes_stats()
        
        # Step 3: Generate modality.json
        generate_modality_json()
        
        # Step 4: Generate relative_stats.json
        generate_relative_stats()
        
        # Step 5: Generate stats_psi0.json
        generate_stats_psi0()
        
        # Step 6: Fix parquet metadata
        fix_parquet_metadata()
        
        # Step 7: Rename parquet columns
        rename_parquet_columns()
        
        # Step 8: Update info.json
        update_info_json()
        
        # Step 9: Rename video folders
        rename_video_folders()
        
        print("\n" + "=" * 80)
        print("✓ All fixes completed successfully!")
        print("=" * 80)
        print("\nGenerated/Fixed files:")
        print(f"  episodes.jsonl                     - Converted to LeRobot format")
        print(f"  episodes_original.jsonl            - Backup of original")
        print(f"  episodes_stats.jsonl               - Per-episode statistics")
        print(f"  modality.json                      - Modality configuration")
        print(f"  relative_stats.json                - Relative action stats")
        print(f"  stats_psi0.json                    - PSI0 format statistics")
        print(f"  info.json                          - Updated with correct paths/names")
        print(f"  Parquet files                      - Fixed metadata and column names")
        print(f"  Video folders                      - Renamed to 'egocentric'")
        print("\nYou can now run training:")
        print(f"  bash scripts/train/psi0/finetune-real-psi0.sh Pick_up_an_apple")
        print("=" * 80 + "\n")
    
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
