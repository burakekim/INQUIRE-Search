import json
import numpy as np

MAPPINGS_DIR = "/data/vision/beery/scratch/eddie/natural_retrieval/inaturalist/notebooks"

print("🔬 VERIFYING DIRECT METADATA MAPPING...")
print("-" * 60)

# 1. Load the 300M Taxon Array in read-only mode
taxon_arr = np.load(f"{MAPPINGS_DIR}/image_id_to_taxon_id.npy", mmap_mode='r')

# 2. Pick the exact ID returned by your FAISS index mock search
faiss_returned_id = 23096840

# 3. Pull the Taxon ID directly from that position
resolved_taxon_id = int(taxon_arr[faiss_returned_id])
print(f"✅ FAISS Index Position {faiss_returned_id:,} maps to Taxon ID: {resolved_taxon_id}")

# 4. Cross-reference with your species dictionary
with open(f"{MAPPINGS_DIR}/species_names.json", 'r') as f:
    species_dict = json.load(f)

species_name = "Unknown Species"
for name, tid in species_dict.items():
    if int(tid) == resolved_taxon_id:
        species_name = name
        break

print(f"✅ Taxon ID {resolved_taxon_id} resolves to Scientific Name: {species_name}")
print("-" * 60)
print("🎉 If a real scientific name printed above, your pipeline is perfectly aligned!")