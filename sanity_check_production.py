import os
import sys
import json
import numpy as np
import faiss

# Load our updated Config class
from config import Config

print("🕵️‍♂️ STARTING 290M PRODUCTION RETRIEVAL AUDIT...")
print("-" * 60)

# 1. Verify Config Mode
if getattr(Config, 'RUN_MODE', 'development') != 'production':
    print("❌ ERROR: Your config.py is NOT set to RUN_MODE = 'production'!")
    sys.exit(1)
print("✅ Config confirmed in PRODUCTION mode.")

# 2. Check Physical File Existence
files_to_check = [
    ("FAISS Index", Config.INDEX_PATH),
    ("Taxon Mapping", Config.IMAGE_ID_TO_TAXON_ID_PATH),
    ("Geo Mapping", Config.GEO_INDEX_PATH),
    ("Month Mapping", Config.MONTH_INDEX_PATH),
    ("Extension Map", Config.PHOTO_ID_TO_EXT_MAP_PATH),
    ("Taxonomy Dictionary", Config.TAXA_FILE)
]

missing_files = 0
for name, path in files_to_check:
    if os.path.exists(path):
        size_gb = os.path.getsize(path) / (1024**3)
        print(f"✅ FOUND: {name:<20} | Size: {size_gb:.4f} GB | Path: {path}")
    else:
        print(f"❌ MISSING: {name:<20} | NOT FOUND at: {path}")
        missing_files += 1

if missing_files > 0:
    print(f"\n🛑 AUDIT FAILED: {missing_files} critical files are missing. Fix your paths first!")
    sys.exit(1)

print("-" * 60)
print("🧠 LOADING ARRAYS IN MMAP MODE FOR MATHEMTICAL ALIGNMENT CHECK...")

try:
    # 3. Load FAISS and get exact vector count
    print("Loading FAISS Index header via MMAP...")
    index = faiss.read_index(Config.INDEX_PATH, faiss.IO_FLAG_MMAP)
    expected_count = index.ntotal
    print(f"🎯 Target Production Vector Count: {expected_count:,}")
    
    # 4. Load Arrays with mmap_mode='r' (Zero memory footprint)
    taxon_arr = np.load(Config.IMAGE_ID_TO_TAXON_ID_PATH, mmap_mode='r')
    geo_arr = np.load(Config.GEO_INDEX_PATH, mmap_mode='r')
    month_arr = np.load(Config.MONTH_INDEX_PATH, mmap_mode='r')
    
    # 5. Length Validation Checks
    mismatches = 0
    for name, arr in [("Taxon Map", taxon_arr), ("Geo Map", geo_arr), ("Month Map", month_arr)]:
        if len(arr) == expected_count:
            print(f"✅ ALIGNED: {name:<12} length matches FAISS exact footprint ({len(arr):,}).")
        else:
            print(f"🔥 MISMATCH: {name:<12} has {len(arr):,} items, but FAISS expects {expected_count:,}!")
            mismatches += 1
            
    if mismatches > 0:
        print("\n🛑 AUDIT FAILED: Mismatched array shapes found. Do not boot the app.")
        sys.exit(1)
        
    print("-" * 60)
    print("🧪 RUNNING END-TO-END LOOKUP TEST (Sample Index: 10,000,000)...")
    
    test_idx = 10_000_000
    sample_taxon = int(taxon_arr[test_idx])
    sample_geo = geo_arr[test_idx]
    sample_month = int(month_arr[test_idx])
    
    print(f"  • Vector Matrix Position : {test_idx:,}")
    print(f"  • Resolved Taxon ID       : {sample_taxon}")
    print(f"  • Raw Encoded Lat/Long   : {sample_geo}")
    print(f"  • Observation Month      : {sample_month}")
    
    # Check if taxonomy dict can read it
    with open(Config.TAXA_FILE, 'r') as f:
        taxa_data = json.load(f)
    
    # Look for the taxon id inside the dict values or keys
    species_name = "Unknown Species"
    for name, tid in taxa_data.items():
        if int(tid) == sample_taxon:
            species_name = name
            break
            
    print(f"  • Resolved Species Name  : {species_name}")
    print("-" * 60)
    print("🎉 SUCCESS! All 290M datasets are physically synced, structurally aligned, and verified.")

except Exception as e:
    print(f"\n💥 CRITICAL CRASH DURING SANITY CHECK: {e}")
    import traceback
    traceback.print_exc()