"""
One-off: build a FAISS-id-aligned license array from the production metadata mmaps.

MemoryMappedMetadataProvider (utils.py) concatenates metadata_*.mmap chunks in the
order returned by `sorted(Path(...).glob("metadata_*.mmap"))` — a *lexicographic*
sort of filenames (metadata_0, metadata_1, metadata_10, metadata_100, ...), not
numeric chunk order. That lexicographic order is what defines the "global image
id" every other aligned array (geo_index, month_index) and FAISS itself use, so
this script must concatenate in the exact same order to stay aligned.

Output:
  - image_id_to_license.npy: int16 array, one code per image, same length/order
    as month_index / geo_index / FAISS ids.
  - license_code_to_name.json: {code: license_string} for decoding.
"""
import json
from pathlib import Path

import numpy as np

from config import Config

DTYPE = np.dtype([
    ('photo_id', np.int64),
    ('photo_uuid', 'S36'),
    ('observer_id', np.int64),
    ('license', 'S64'),
    ('width', np.int32),
    ('height', np.int32),
])

OUT_DIR = Path("/data/vision/beery/scratch/burake/searchin/data/inputs/inat_data")
OUT_ARRAY = OUT_DIR / "image_id_to_license.npy"
OUT_CODEMAP = OUT_DIR / "license_code_to_name.json"


def main():
    metadata_dir = Path(Config.METADATA_PATH)
    # Same call/order as MemoryMappedMetadataProvider.__init__
    chunk_files = sorted(metadata_dir.glob("metadata_*.mmap"))
    print(f"Found {len(chunk_files)} metadata chunks", flush=True)

    license_to_code = {}
    code_chunks = []
    total = 0

    for i, chunk_file in enumerate(chunk_files):
        mm = np.memmap(chunk_file, dtype=DTYPE, mode='r')
        licenses = mm['license']
        uniq, inv = np.unique(licenses, return_inverse=True)

        local_codes = np.empty(len(uniq), dtype=np.int16)
        for j, u in enumerate(uniq):
            s = u.decode('utf-8').rstrip('\x00').strip()
            if not s:
                # Some photos in the open-data dump carry no license value at all.
                s = 'Unknown'
            if s not in license_to_code:
                license_to_code[s] = len(license_to_code)
            local_codes[j] = license_to_code[s]

        code_chunks.append(local_codes[inv].astype(np.int16))
        total += len(mm)
        del mm

        if (i + 1) % 25 == 0 or i == len(chunk_files) - 1:
            print(f"  processed {i + 1}/{len(chunk_files)} chunks, {total} records so far", flush=True)

    all_codes = np.concatenate(code_chunks)
    assert len(all_codes) == total

    print(f"Total records: {total}", flush=True)
    print(f"License vocabulary: {license_to_code}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_ARRAY, all_codes)
    code_to_name = {v: k for k, v in license_to_code.items()}
    with open(OUT_CODEMAP, 'w') as f:
        json.dump(code_to_name, f, indent=2)

    print(f"Wrote {OUT_ARRAY} ({all_codes.nbytes / 1e6:.1f} MB)")
    print(f"Wrote {OUT_CODEMAP}")


if __name__ == "__main__":
    main()
