import numpy as np
import faiss
from flask import Flask, request, jsonify
import os
import threading
from config import Config

# The pool size of the request currently being served, kept per-thread. app.py runs each search in
# a ThreadPoolExecutor with max_workers=2, so two searches can be in flight at once; a single shared
# attribute on the index let one request's pool size be read by the other (E43 -- an unfiltered
# request would reset it to None and the filtered request that asked for it logged None).
_TLS = threading.local()


class EnhancedSearchIndex:
    """
    Manages geospatial + other filtering on top of a Faiss index.
    """
    def __init__(self, faiss_index_path, emb_to_inat_index=None):
        """
        Manages geospatial + other filtering on top of a Faiss index.
        """
        import time
        import sys
        t_load_start = time.time()
        print(f"[EnhancedSearchIndex.__init__] Loading FAISS index from: {faiss_index_path}", flush=True)
        
        try:
            print(f"[EnhancedSearchIndex.__init__] Production Scale: Loading via MMAP (Read-Only)...", flush=True)
            self.index = faiss.read_index(faiss_index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
            print(f"[EnhancedSearchIndex.__init__] Mmap load successful", flush=True)
        except Exception as e:
            print(f"[EnhancedSearchIndex.__init__] Mmap load failed: {e}, falling back to regular load...", flush=True)
            self.index = faiss.read_index(faiss_index_path)

        print(f"[EnhancedSearchIndex.__init__] FAISS index loaded in {time.time() - t_load_start:.3f}s", flush=True)
        
        # --- THREAD OPTIMIZATION & NPROBE INJECTION ---
        # E48: efSearch belongs to the COARSE QUANTIZER, not to the IVF index. This block used to
        # write `target_idx.hnsw.efSearch = 8` on the IndexIVFPQ, which has no `.hnsw` at all, so
        # the guard was never true and the assignment never happened -- the value stored in the
        # index file (1034) was what ran. Measured before changing anything
        # (exps/archive/E48_efsearch.py, 264 benchmark queries, cell recall against the exact top-256
        # centroids by brute force):
        #
        #     efSearch     cells returned (asked 256)   cell recall@256   coarse ms/query
        #            8                            228             39.0%              0.58
        #          256                            256             97.8%              0.12
        #         1034 (the accident)             256             99.8%              0.58
        #         2048                            256            100.0%              1.29
        #
        # So the intended 8 would have been actively harmful: the quantizer cannot even fill the
        # request, returning 228 cells when asked for 256, and 39% of those are the wrong cells.
        # (Note it does NOT silently lift efSearch to k -- measured, not assumed.)
        #
        # 2048 is set DELIBERATELY, and the reason is not headroom: it is the smallest tested value
        # reaching 100.0% cell recall, which takes the coarse quantizer out of consideration
        # entirely as a source of retrieval error so nobody has to reason about it again. It costs
        # 0.7 ms on a ~1 s query. Secondarily, cell recall was only measured at the deployed
        # nprobe=256, and 2048 is cheap insurance if anyone raises nprobe later.
        #
        # Setting it here rather than relying on the file also means an index rebuild cannot
        # silently change it back: the stored 1034 is now overridden on every load.
        COARSE_EF_SEARCH = 2048
        try:
            faiss.omp_set_num_threads(4)
            target_idx = faiss.downcast_index(self.index.index) if type(self.index).__name__ == "IndexPreTransform" else self.index
            if hasattr(target_idx, 'nprobe'):
                target_idx.nprobe = 1
            quantizer = None
            if hasattr(target_idx, 'quantizer'):
                quantizer = faiss.downcast_index(target_idx.quantizer)
            if quantizer is not None and hasattr(quantizer, 'hnsw'):
                was = quantizer.hnsw.efSearch
                quantizer.hnsw.efSearch = COARSE_EF_SEARCH
                print(f"[EnhancedSearchIndex.__init__] coarse quantizer "
                      f"{type(quantizer).__name__}: efSearch {was} -> {quantizer.hnsw.efSearch} "
                      f"(E48; deliberate, see comment)", flush=True)
            else:
                print(f"[EnhancedSearchIndex.__init__] coarse quantizer has no HNSW; "
                      f"efSearch not applicable.", flush=True)
        except Exception as e:
            print(f"[EnhancedSearchIndex.__init__] Could not optimize inner index parameters: {e}", flush=True)
        
        self.emb_to_inat_index = emb_to_inat_index
        
        # --- FILTERS ROUTING ---
        env_filters = os.environ.get('FILTERS_DIR')
        if env_filters and os.path.exists(env_filters):
            self.filters_dir = env_filters
        elif hasattr(Config, 'FILTERS_DIR') and Config.FILTERS_DIR and os.path.exists(Config.FILTERS_DIR):
            self.filters_dir = Config.FILTERS_DIR
        else:
            self.filters_dir = '/data/vision/beery/scratch/eddie/natural_retrieval/inaturalist/notebooks'
            
        print(f"Using filters directory: {self.filters_dir}", flush=True)

        self.geo_index = None
        self._build_geo_index()
        self._build_species_index()
        self._build_month_index()
        self._build_license_index()
        
        # --- FATAL RAM BOMB BYPASS ---
        self.faiss_to_metadata = None
        self.metadata_to_faiss = None
        print("[EnhancedSearchIndex.__init__] Production Setup Complete: All OOM landmines successfully bypassed.", flush=True)
        sys.stdout.flush()

    def _build_species_index(self):
        # RAM FIX: Prevent Python from unpickling a 20GB dictionary of lists into RAM.
        # In production we instead mmap the FORWARD per-image taxon array (image_id_to_taxon_id.npy,
        # ~1.2GB, row idx == FAISS id) and compute the id subset per query on the fly (like the geo
        # filter does), so the in-index IDSelector species filter works without the 20GB reverse dict.
        self.image_taxon_index = None
        forward_path = os.path.join(self.filters_dir, 'image_id_to_taxon_id.npy')
        if os.path.exists(forward_path):
            self.image_taxon_index = np.load(forward_path, mmap_mode='r')
            print(f"[EnhancedSearchIndex] Loaded forward taxon array (mmap): {self.image_taxon_index.shape}", flush=True)
        else:
            print(f"[EnhancedSearchIndex] Forward taxon array not found at {forward_path}.", flush=True)

        # E47: the descendants map is loaded in production too. The RAM bomb this bypass exists for
        # is `map_taxon_id_to_image_ids.npy` -- 886 MB on disk that unpickles to ~20 GB of Python
        # lists -- but `taxon_descendants.npy` is a separate 2.6 MB file (38,039 keys, ~1.3 MB in
        # RAM, 0.24 s to load) that was only skipped because both were loaded behind one `if`.
        # Dropping it made the production filter species-exact, so it could not see the 83,504
        # subspecies iNat files beside its species -- which is 673 of the 683 confirming photos the
        # filter was blind to (E46 Phase 0b). The feature was dead, not missing.
        descendants_path = os.path.join(self.filters_dir, 'taxon_descendants.npy')
        self.taxon_descendants = None
        if os.path.exists(descendants_path):
            self.taxon_descendants = np.load(descendants_path, allow_pickle=True).item()
            print(f"[EnhancedSearchIndex] Loaded taxon descendants map "
                  f"({len(self.taxon_descendants):,} taxa).", flush=True)
        else:
            print(f"[EnhancedSearchIndex] Taxon descendants map not found at {descendants_path}; "
                  f"species filtering will be exact-rank only.", flush=True)

        if getattr(Config, 'RUN_MODE', 'development') == 'production':
            print("[EnhancedSearchIndex] Production Mode: Bypassing heavy inverse species dictionary load.", flush=True)
            self.taxon_index = None
            return

        taxon_path = os.path.join(self.filters_dir, 'map_taxon_id_to_image_ids.npy')
        if os.path.exists(taxon_path):
            self.taxon_index = np.load(taxon_path, allow_pickle=True).item()
            print("[EnhancedSearchIndex] Loaded species inverted index.", flush=True)
        else:
            self.taxon_index = None
            print(f"[EnhancedSearchIndex] Species inverted index not found.", flush=True)

    def _build_month_index(self):
        month_path = os.path.join(self.filters_dir, 'image_id_to_month.npy')
        if os.path.exists(month_path):
            # RAM FIX: Use memory mapping so it consumes 0 RAM
            self.month_index = np.load(month_path, mmap_mode='r')
            print("[EnhancedSearchIndex] Loaded month index (mmap).", flush=True)
        else:
            self.month_index = None
            print(f"[EnhancedSearchIndex] Month index not found.", flush=True)

    def _build_license_index(self):
        # Built via backend/build_license_index.py; lives outside filters_dir
        # (which points at a colleague's shared directory).
        license_path = getattr(Config, 'LICENSE_INDEX_PATH', None)
        code_map_path = getattr(Config, 'LICENSE_CODE_MAP_PATH', None)
        self.license_index = None
        self.license_name_to_code = {}
        if license_path and os.path.exists(license_path):
            # RAM FIX: Use memory mapping so it consumes 0 RAM
            self.license_index = np.load(license_path, mmap_mode='r')
            print("[EnhancedSearchIndex] Loaded license index (mmap).", flush=True)
        else:
            print(f"[EnhancedSearchIndex] License index not found.", flush=True)
        if code_map_path and os.path.exists(code_map_path):
            import json
            with open(code_map_path, 'r') as f:
                code_to_name = {int(k): v for k, v in json.load(f).items()}
            self.license_name_to_code = {v: k for k, v in code_to_name.items()}


    ##############################
    # Functions for array geo index
    def _encode_lat_lon(self, lat, lon):
        # Shift latitude by +90° and scale
        lat_enc = np.uint16(np.round((lat + 90.0) * 100))
        # Shift longitude by +180° and scale
        lon_enc = np.uint16(np.round((lon + 180.0) * 100))
        return (lat_enc, lon_enc)

    def _decode_lat_lon(self, lat_enc, lon_enc):
        lat = (lat_enc / 100.0) - 90.0
        lon = (lon_enc / 100.0) - 180.0
        return (lat, lon)

    def _build_geo_index(self):
        geo_path = os.path.join(self.filters_dir, 'image_id_to_latlong_int.npy')
        if os.path.exists(geo_path):
            # RAM FIX: Use memory mapping for the 300 million row array
            self.geo_index = np.load(geo_path, mmap_mode='r')
            print("[EnhancedSearchIndex] Loaded geo index (mmap).", flush=True)
        else:
            self.geo_index = None
            print(f"[EnhancedSearchIndex] Geo index not found.", flush=True)

    ##############################

    def _get_ids_in_bounding_box(self, min_lat, max_lat, min_lng, max_lng):
        """
        Return the 'row indices' (which must match the Faiss ID or be mappable to the Faiss ID)
        for images whose lat/lng is within the bounding box.
        """
        if self.geo_index is None:
            raise ValueError("Geo index not built. Call _build_geo_index() first.")

        print('[geo] Filtering by bounding box:', min_lat, max_lat, min_lng, max_lng)
        min_lat_enc, min_lon_enc = self._encode_lat_lon(min_lat, min_lng)
        max_lat_enc, max_lon_enc = self._encode_lat_lon(max_lat, max_lng)
        
        # Optimized: use numpy's efficient boolean indexing
        # Pre-extract columns to avoid repeated indexing
        lats = self.geo_index[:, 0]
        lons = self.geo_index[:, 1]
        
        # Combine masks efficiently
        mask = ((lats >= min_lat_enc) & (lats <= max_lat_enc) & 
                (lons >= min_lon_enc) & (lons <= max_lon_enc))
        subset_ids = np.flatnonzero(mask)  # More efficient than np.where()[0]

        print(f"[geo] Found {len(subset_ids)} images in bounding box")
        return subset_ids

    def _bbox_mask_for_ids(self, ids, min_lat, max_lat, min_lng, max_lng):
        """The SAME bounding-box predicate as `_get_ids_in_bounding_box`, applied to a subset of
        rows instead of to all 300M of them (E47).

        `_get_ids_in_bounding_box` builds the full in-box id array (161M ids for the North-America
        box) and the caller then intersects it with a much smaller species subset. Intersection
        commutes, so `intersect1d(all_in_box, ids) == ids[predicate(ids)]` exactly -- both are the
        rows of `ids` that satisfy the predicate, and `ids` arrives sorted and unique from
        `np.flatnonzero`, so the result stays sorted and unique too. Doing it this way reads
        len(ids) rows of the geo mmap rather than 300M, which is the whole ~25 s cost.
        """
        ids = np.asarray(ids, dtype=np.int64)
        if len(ids) == 0:
            return ids
        min_lat_enc, min_lon_enc = self._encode_lat_lon(min_lat, min_lng)
        max_lat_enc, max_lon_enc = self._encode_lat_lon(max_lat, max_lng)
        sub = np.asarray(self.geo_index[ids])
        lats, lons = sub[:, 0], sub[:, 1]
        keep = ((lats >= min_lat_enc) & (lats <= max_lat_enc) &
                (lons >= min_lon_enc) & (lons <= max_lon_enc))
        return ids[keep]

    def _is_hnsw_index(self):
        """Return True if the loaded Faiss index is an HNSW-style index."""
        try:
            cls_name = type(self.index).__name__
            return 'IndexHNSW' in cls_name or 'HNSW' in cls_name
        except Exception:
            return False

    def _metadata_to_faiss_ids(self, metadata_indices):
        """Convert metadata row indices -> Faiss vector IDs if mapping available.

        :param metadata_indices: 1D array-like of metadata row indices
        :returns: NumPy array of Faiss IDs (int64)
        """
        meta_arr = np.asarray(metadata_indices).astype(np.int64)
        if self.metadata_to_faiss is None:
            # No mapping available; assume metadata indices == Faiss IDs
            return meta_arr

        # metadata_to_faiss may be an ndarray (argsort result) or a dict
        if isinstance(self.metadata_to_faiss, np.ndarray):
            # ensure indices are within range
            mask = (meta_arr >= 0) & (meta_arr < self.metadata_to_faiss.shape[0])
            faiss_ids = self.metadata_to_faiss[meta_arr[mask]]
            return np.asarray(faiss_ids, dtype=np.int64)
        else:
            # dict
            faiss_list = []
            for m in meta_arr.tolist():
                f = self.metadata_to_faiss.get(int(m), None)
                if f is not None:
                    faiss_list.append(int(f))
            return np.asarray(faiss_list, dtype=np.int64)

    def _search_and_filter_post(self, query_vec, k, valid_ids, nprobe=10, multiplier=10):
        """
        Fallback search: For HNSW indices that don't support ID selectors,
        we use different strategies based on the size of the valid subset.
        """
        import time
        import sys
        t_start = time.time()
        print(f"[_search_and_filter_post] Starting with {len(valid_ids)} valid IDs, k={k}", flush=True)
        sys.stdout.flush()
        
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        num_valid = len(valid_ids)
        
        # Set OpenMP threads to 1 for speed
        try:
            old_threads = faiss.omp_get_max_threads()
            faiss.omp_set_num_threads(1)
            print(f"[_search_and_filter_post] Set OMP threads from {old_threads} to 1", flush=True)
        except Exception as e:
            print(f"[_search_and_filter_post] Could not set OMP threads: {e}", flush=True)
        
        # If the valid subset is very large (>50% of index), just do normal search
        # and post-filter - it's faster than reconstructing so many vectors
        selectivity = num_valid / max(self.index.ntotal, 1)
        
        # Be even more aggressive with large subsets - use simple search for >30% or >3000 images
        if selectivity > 0.3 or num_valid > 3000:
            # Large subset: use simple post-filter with minimal multiplier
            print(f"[fallback] Large subset ({num_valid} ids, {selectivity:.1%}), using minimal post-filter search", flush=True)
            # For very large subsets, use an even smaller multiplier
            if num_valid > 10000:
                k_try = min(int(k * 1.2), 300)  # Ultra-conservative for huge subsets
            else:
                k_try = min(int(k * 1.5), 500)  # Conservative multiplier
            
            print(f"[fallback] Searching for k_try={k_try} results...", flush=True)
            sys.stdout.flush()
            t_search_start = time.time()
            D_all, I_all = self.index.search(query_vec, k_try)
            t_search_elapsed = time.time() - t_search_start
            print(f"[fallback] Search took {t_search_elapsed:.3f}s", flush=True)
            
            # Convert to set for fast lookup - use frozenset for large arrays
            print(f"[fallback] Converting {num_valid} IDs to set...", flush=True)
            t_set_start = time.time()
            valid_set = set(valid_ids.tolist()) if num_valid < 100000 else set(int(x) for x in valid_ids)
            t_set_elapsed = time.time() - t_set_start
            print(f"[fallback] Set creation took {t_set_elapsed:.3f}s", flush=True)
            
            print(f"[fallback] Filtering results...", flush=True)
            t_filter_start = time.time()
            filtered_I = []
            filtered_D = []
            for dist_row, idx_row in zip(D_all, I_all):
                row_I = []
                row_D = []
                for d, idx in zip(dist_row, idx_row):
                    if int(idx) in valid_set:
                        row_I.append(int(idx))
                        row_D.append(float(d))
                        if len(row_I) >= k:
                            break
                while len(row_I) < k:
                    row_I.append(-1)
                    row_D.append(float('inf'))
                filtered_I.append(np.array(row_I, dtype=np.int64))
                filtered_D.append(np.array(row_D, dtype=np.float32))
            
            t_filter_elapsed = time.time() - t_filter_start
            print(f"[fallback] Filtering took {t_filter_elapsed:.3f}s", flush=True)
            t_total = time.time() - t_start
            print(f"[fallback] Total post-filter time: {t_total:.3f}s", flush=True)
            sys.stdout.flush()
            
            return np.vstack(filtered_D), np.vstack(filtered_I)
        
        # Small subset: build temporary index for exact search
        print(f"[fallback] Small subset ({num_valid} ids, {selectivity:.1%}), building temporary index")
        
        valid_ids_list = valid_ids.tolist()
        
        # Try to reconstruct vectors from the index
        try:
            # Get dimension from query
            d = query_vec.shape[1] if query_vec.ndim > 1 else query_vec.shape[0]
            
            # Build a flat index with just the filtered vectors
            subset_vectors = np.zeros((len(valid_ids_list), d), dtype=np.float32)
            for i, vid in enumerate(valid_ids_list):
                try:
                    subset_vectors[i] = self.index.reconstruct(int(vid))
                except Exception:
                    # If reconstruct fails, this approach won't work
                    print("[fallback] Cannot reconstruct vectors, falling back to brute force post-filter")
                    # Fall back to the old approach but with minimal over-fetch
                    k_try = min(k * 3, len(valid_ids_list), 1000)
                    D_all, I_all = self.index.search(query_vec, k_try)
                    valid_set = set(valid_ids_list)
                    
                    filtered_I = []
                    filtered_D = []
                    for dist_row, idx_row in zip(D_all, I_all):
                        row_I = []
                        row_D = []
                        for d, idx in zip(dist_row, idx_row):
                            if int(idx) in valid_set:
                                row_I.append(int(idx))
                                row_D.append(float(d))
                                if len(row_I) >= k:
                                    break
                        while len(row_I) < k:
                            row_I.append(-1)
                            row_D.append(float('inf'))
                        filtered_I.append(np.array(row_I, dtype=np.int64))
                        filtered_D.append(np.array(row_D, dtype=np.float32))
                    
                    return np.vstack(filtered_D), np.vstack(filtered_I)
            
            # Create temporary flat index for exact search on subset
            temp_index = faiss.IndexFlatIP(d)  # Use inner product (same as your main index likely uses)
            temp_index.add(subset_vectors)
            
            # Search in the subset
            D_sub, I_sub = temp_index.search(query_vec, min(k, len(valid_ids_list)))
            
            # Map back to original IDs
            result_I = []
            result_D = []
            for dist_row, idx_row in zip(D_sub, I_sub):
                row_I = []
                row_D = []
                for d, idx in zip(dist_row, idx_row):
                    if idx >= 0 and idx < len(valid_ids_list):
                        row_I.append(valid_ids_list[idx])
                        row_D.append(float(d))
                while len(row_I) < k:
                    row_I.append(-1)
                    row_D.append(float('inf'))
                result_I.append(np.array(row_I[:k], dtype=np.int64))
                result_D.append(np.array(row_D[:k], dtype=np.float32))
            
            print(f"[fallback] Completed exact search on {len(valid_ids_list)} vectors")
            return np.vstack(result_D), np.vstack(result_I)
            
        except Exception as e:
            print(f"[fallback] Error in optimized search: {e}, falling back to simple post-filter")
            # Final fallback - minimal fetch and filter
            k_try = min(k * 3, 1000)
            D_all, I_all = self.index.search(query_vec, k_try)

        valid_set = set(valid_ids.tolist())

        # Collect first k matches that are in valid_set
        filtered_I = []
        filtered_D = []
        for dist_row, idx_row in zip(D_all, I_all):
            row_I = []
            row_D = []
            for d, idx in zip(dist_row, idx_row):
                if int(idx) in valid_set:
                    row_I.append(int(idx))
                    row_D.append(float(d))
                    if len(row_I) >= k:
                        break
            # pad if not enough results
            while len(row_I) < k:
                row_I.append(-1)
                row_D.append(float('inf'))
            filtered_I.append(np.array(row_I, dtype=np.int64))
            filtered_D.append(np.array(row_D, dtype=np.float32))

        # Return stacked arrays shaped (1, k)
        return np.vstack(filtered_D), np.vstack(filtered_I)

    def search_with_geo_filter(self, query_vec, k, min_lat, max_lat, min_lng, max_lng):
        """
        Perform KNN search restricted to images in the bounding box.
        
        :param query_vec: A NumPy array of shape (1, d) or (d,) containing the query embedding.
        :param k: Number of nearest neighbors.
        :param min_lat, max_lat, min_lng, max_lng: bounding box coordinates.
        :return: (D, I) where D are distances, I are Faiss IDs.
        """
        import time
        import sys
        t_start = time.time()
        print(f"[search_with_geo_filter] Starting geo-filtered search with k={k}", flush=True)
        sys.stdout.flush()
        
        # 1. Find row indices in bounding box
        subset_ids = self._get_ids_in_bounding_box(min_lat, max_lat, min_lng, max_lng)
        if len(subset_ids) == 0:
            # No images in that bounding box
            print("[search_with_geo_filter] No images in bounding box", flush=True)
            return np.array([[]]), np.array([[]])
        
        print(f"[search_with_geo_filter] Searching with {len(subset_ids)} images in bounding box.", flush=True)
        # 2. Convert metadata row indices -> Faiss IDs (if mapping available), then build selector
        faiss_ids = self._metadata_to_faiss_ids(subset_ids)

        # For IVF-style indexes we can attach the selector via SearchParametersIVF.
        # HNSW-style indexes (IndexHNSW) don't accept SearchParametersIVF and will
        # raise the 'params type invalid' error. Detect and fallback to a
        # post-filter approach for those indexes.
        if self._is_hnsw_index():
            print("Index appears to be HNSW-style; using post-filter fallback for geo selection.", flush=True)
            sys.stdout.flush()
            return self._search_and_filter_post(query_vec, k, faiss_ids, nprobe=1)

        # Set OpenMP threads to 1 for speed
        try:
            old_threads = faiss.omp_get_max_threads()
            faiss.omp_set_num_threads(1)
            print(f"[search_with_geo_filter] Set OMP threads from {old_threads} to 1", flush=True)
        except Exception as e:
            print(f"[search_with_geo_filter] Could not set OMP threads: {e}", flush=True)

        sel = faiss.IDSelectorBatch(faiss_ids.astype(np.int64))  # must be int64 for Faiss
        params = faiss.SearchParametersIVF(sel=sel, nprobe=1)

        # 4. Run the search restricted to that subset
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        t_search_start = time.time()
        print(f"[search_with_geo_filter] Calling index.search with nprobe=1...", flush=True)
        sys.stdout.flush()
        D, I = self.index.search(query_vec, k, params=params)
        t_search_elapsed = time.time() - t_search_start

        print(f"[search_with_geo_filter] Search took {t_search_elapsed:.3f}s, found {len(I[0])} results.", flush=True)
        t_total = time.time() - t_start
        print(f"[search_with_geo_filter] Total time: {t_total:.3f}s", flush=True)
        sys.stdout.flush()

        return D, I
    

    def _get_ids_in_species(self, taxon_id, expand_descendants=False):
        """
        Return the 'row indices' (which must match the Faiss ID or be mappable to the Faiss ID)
        for images with the given species. `taxon_id` may be a single id or a list/array of ids
        (OR match, e.g. predator OR prey).

        Two backends: the reverse dict (self.taxon_index, dev only) or, in production, the forward
        mmap array (self.image_taxon_index) scanned on the fly. **Both CAN expand descendants**
        (E47) so the two paths agree when asked; before, only the dev path did, which made the
        deployed filter species-exact and blind to the subspecies iNat actually labels the photo
        with.

        Expansion is DOWNWARD ONLY -- a species plus its subspecies, never up to genus or family.
        Dilution is monotone in pool size (E46: 51k photos -> 60 edges, 21.5M -> 51), so widening
        upward is precisely the mechanism that made the 717-species scope lose.

        **Default False (E48).** Exact-rank is what the deployed filter has always done, and the
        randomized retrieval condition's species-scope arm must keep meaning what E46 measured while
        it collects its first observations -- see the note at
        backend/services/search_service.py:193. Pass expand_descendants=True to opt in per request.
        """
        import time
        t_start = time.time()

        taxon_ids = taxon_id if isinstance(taxon_id, (list, tuple, np.ndarray, set)) else [taxon_id]
        taxon_ids = [int(t) for t in taxon_ids]

        n_asked = len(taxon_ids)
        if expand_descendants and self.taxon_descendants:
            expanded = set(taxon_ids)
            for tid in taxon_ids:
                expanded.update(int(d) for d in self.taxon_descendants.get(tid, ()))
            taxon_ids = sorted(expanded)

        if self.taxon_index is not None:
            # descendants are already folded into taxon_ids above, so this walks one flat list
            # rather than nesting a second descendant loop inside it
            arrays_to_concat = [np.asarray(self.taxon_index[tid])
                                for tid in taxon_ids if tid in self.taxon_index]
            subset_ids = (np.unique(np.concatenate(arrays_to_concat)) if arrays_to_concat
                          else np.array([], dtype=np.int64))
        elif self.image_taxon_index is not None:
            # On-the-fly scan of the forward array (row idx == FAISS id), like the geo filter.
            # The array is 300M rows but the index only has ntotal (~289M) vectors -- the trailing
            # rows are unindexed padding (all zeros), so real taxon ids never match there, but clip
            # to < ntotal defensively so no out-of-range id ever reaches the IDSelector.
            mask = np.isin(self.image_taxon_index, taxon_ids)
            subset_ids = np.flatnonzero(mask)
            ntotal = getattr(self.index, "ntotal", None)
            if ntotal is not None:
                subset_ids = subset_ids[subset_ids < ntotal]
        else:
            raise ValueError("No taxon index available (neither reverse dict nor forward array).")

        t_elapsed = time.time() - t_start
        print(f"[species] Found {len(subset_ids)} images for {n_asked} requested taxa "
              f"(+{len(taxon_ids) - n_asked} descendants, expand={expand_descendants}) "
              f"in {t_elapsed:.3f}s", flush=True)
        return subset_ids
    
    def search_with_species_filter(self, query_vec, k, taxon_id, nprobe=1):
        """
        Perform KNN search restricted to images with the given species.
        
        :param query_vec: A NumPy array of shape (1, d) or (d,) containing the query embedding.
        :param k: Number of nearest neighbors.
        :return: (D, I) where D are distances, I are Faiss IDs.
        """
        import time
        import sys
        t_start = time.time()
        print(f"[search_with_species_filter] Starting search for taxon_id={taxon_id}, k={k}", flush=True)
        sys.stdout.flush()
        
        # 1. Find row indices in species
        t_species_start = time.time()
        subset_ids = self._get_ids_in_species(taxon_id)
        t_species_elapsed = time.time() - t_species_start
        print(f"[search_with_species_filter] Got species IDs in {t_species_elapsed:.3f}s", flush=True)
        
        if len(subset_ids) == 0:
            print("[search_with_species_filter] No images found for taxon_id", flush=True)
            return [np.array([])], [np.array([])]
        
        # 2. Convert to FAISS IDs
        t_map_start = time.time()
        faiss_ids = self._metadata_to_faiss_ids(subset_ids)
        t_map_elapsed = time.time() - t_map_start
        print(f"[search_with_species_filter] Mapped to {len(faiss_ids)} FAISS IDs in {t_map_elapsed:.3f}s", flush=True)

        if self._is_hnsw_index():
            print("Index appears to be HNSW-style; using post-filter fallback for species selection.", flush=True)
            sys.stdout.flush()
            return self._search_and_filter_post(query_vec, k, faiss_ids, nprobe=nprobe)

        # Set OpenMP threads to 1 for speed
        try:
            old_threads = faiss.omp_get_max_threads()
            faiss.omp_set_num_threads(1)
            print(f"[search_with_species_filter] Set OMP threads from {old_threads} to 1", flush=True)
        except Exception as e:
            print(f"[search_with_species_filter] Could not set OMP threads: {e}", flush=True)
        
        # Build selector
        t_selector_start = time.time()
        sel = faiss.IDSelectorBatch(faiss_ids.astype(np.int64))
        params = faiss.SearchParametersIVF(sel=sel, nprobe=nprobe)
        t_selector_elapsed = time.time() - t_selector_start
        print(f"[search_with_species_filter] Built selector in {t_selector_elapsed:.3f}s", flush=True)

        t_search_start = time.time()
        print(f"[search_with_species_filter] Calling index.search with nprobe={nprobe}...", flush=True)
        sys.stdout.flush()
        D, I = self.index.search(query_vec, k, params=params)
        t_search_elapsed = time.time() - t_search_start
        
        print(f"[search_with_species_filter] FAISS search took {t_search_elapsed:.3f}s, found {len(I[0])} results.", flush=True)
        t_total = time.time() - t_start
        print(f"[search_with_species_filter] Total time: {t_total:.3f}s", flush=True)
        sys.stdout.flush()
        return D, I
    

    @property
    def last_filter_pool_size(self):
        """Candidate-pool size of the search this thread just ran, or None if it was unfiltered.
        Read by SearchService.search to report `filter_pool_size` to the caller."""
        return getattr(_TLS, 'pool_size', None)

    def search_with_filters(self, query_vec, k, geo_filter=None, taxon_id=None, months=None,
                            licenses=None, nprobe=1, expand_descendants=False):
        """
        Perform KNN search restricted to images in the bounding box.

        :param query_vec: A NumPy array of shape (1, d) or (d,) containing the query embedding.
        :param k: Number of nearest neighbors.
        :param geo_filter (optional): bounding box coordinate tuple (min_lat, max_lat, min_lng, max_lng)
        :param taxon_id (optional): species ID
        :param months (optional): list of months (numbered 1-12) to filter by
        :param licenses (optional): list of license codes (ints, from Config.LICENSE_CODE_MAP_PATH) to filter by
        :param nprobe: number of IVF probes (i.e. search strength)
        :return: (D, I) where D are distances, I are Faiss IDs.
        """
        import time
        import sys
        t_start = time.time()
        print(f"[search_with_filters] Starting filtered search with k={k}, nprobe={nprobe}", flush=True)
        sys.stdout.flush()

        subset_ids = None

        # E47: species FIRST, then the bounding box applied to what survives. The old order built
        # the full 161M-id in-box array and then intersected it down to a ~50k species subset,
        # paying ~25 s for a pool three orders of magnitude smaller. Intersection commutes, so the
        # id set is unchanged (see _bbox_mask_for_ids); only the cost changes. A geo-only request
        # still has to scan, because there is nothing to narrow it with first.
        if taxon_id is not None:
            subset_ids = self._get_ids_in_species(taxon_id, expand_descendants)
            print("[filter] Have ", len(subset_ids), "images in filter after taxon filter.", flush=True)

        if geo_filter is not None:
            min_lat, max_lat, min_lng, max_lng = geo_filter
            if subset_ids is None:
                subset_ids = self._get_ids_in_bounding_box(min_lat, max_lat, min_lng, max_lng)
            else:
                subset_ids = self._bbox_mask_for_ids(subset_ids, min_lat, max_lat, min_lng, max_lng)
            print("[filter] Have ", len(subset_ids), "images in filter after geo filter.", flush=True)

        # Apply month filtering. Previously this only ran when subset_ids was
        # already populated (from geo or taxon filters). That made "month-only"
        # filters no-ops. Support both cases:
        #  - If subset_ids is None, build it from the month_index (month-only filter)
        #  - If subset_ids exists, intersect it with the months
        if months is not None:
            if self.month_index is None:
                print("Month filter provided but month index not available; ignoring months.", flush=True)
            else:
                if subset_ids is None:
                    # Build subset of all image ids matching the requested months
                    month_subset = np.where(np.isin(self.month_index, months))[0]
                    subset_ids = month_subset
                    print("[filter] Have ", len(subset_ids), "images in filter after month-only filter.", flush=True)
                else:
                    # Intersect the existing subset with the months
                    # Keep the previous safety check for extremely large subsets
                    if len(subset_ids) < 500 * 1000:
                        subset_ids = subset_ids[np.isin(self.month_index[subset_ids], months)]
                        print("[filter] Have ", len(subset_ids), "images in filter after month filter.", flush=True)
                    else:
                        # If the existing subset is huge, intersection may be expensive
                        # but still perform it; warn for visibility.
                        subset_ids = subset_ids[np.isin(self.month_index[subset_ids], months)]
                        print("[filter] (large) Have ", len(subset_ids), "images in filter after month filter.", flush=True)

        #     month_subset_ids = self._get_ids_in_months(months)
        #     if subset_ids is None:
        #         subset_ids = month_subset_ids
        #     else:
        #         subset_ids = np.intersect1d(subset_ids, month_subset_ids)

        # Apply license filtering, same shape as the month-filter block above.
        if licenses is not None:
            if self.license_index is None:
                print("License filter provided but license index not available; ignoring licenses.", flush=True)
            else:
                if subset_ids is None:
                    license_subset = np.where(np.isin(self.license_index, licenses))[0]
                    subset_ids = license_subset
                    print("[filter] Have ", len(subset_ids), "images in filter after license-only filter.", flush=True)
                else:
                    subset_ids = subset_ids[np.isin(self.license_index[subset_ids], licenses)]
                    print("[filter] Have ", len(subset_ids), "images in filter after license filter.", flush=True)

        # Size of the candidate pool this request actually searched, so a draw can be logged with
        # it (E37 Phase 5). None when the request was unfiltered.
        _TLS.pool_size = None if subset_ids is None else int(len(subset_ids))

        if subset_ids is not None:
            if len(subset_ids) == 0:
                print("No images match the filters.", flush=True)
                return [np.array([])], [np.array([])]
            print(f"[search_with_filters] Searching with {len(subset_ids)} images that match filters.", flush=True)
            
            # 2. Convert metadata indices -> Faiss IDs and build an IDSelectorBatch
            faiss_ids = self._metadata_to_faiss_ids(subset_ids)

            if self._is_hnsw_index():
                print("Index appears to be HNSW-style; using post-filter fallback for combined filters.", flush=True)
                sys.stdout.flush()
                return self._search_and_filter_post(query_vec, k, faiss_ids, nprobe=nprobe)

            # Set OpenMP threads to 1 for speed
            try:
                old_threads = faiss.omp_get_max_threads()
                faiss.omp_set_num_threads(1)
                print(f"[search_with_filters] Set OMP threads from {old_threads} to 1", flush=True)
            except Exception as e:
                print(f"[search_with_filters] Could not set OMP threads: {e}", flush=True)

            sel = faiss.IDSelectorBatch(faiss_ids.astype(np.int64))
            params = faiss.SearchParametersIVF(sel=sel, nprobe=nprobe)

            t_search_start = time.time()
            print(f"[search_with_filters] Calling index.search with nprobe={nprobe}...", flush=True)
            sys.stdout.flush()
            result = self.index.search(query_vec, k, params=params)
            t_search_elapsed = time.time() - t_search_start
            
            print(f"[search_with_filters] Search took {t_search_elapsed:.3f}s", flush=True)
            t_total = time.time() - t_start
            print(f"[search_with_filters] Total time: {t_total:.3f}s", flush=True)
            sys.stdout.flush()
            return result
        else:
            # No id subset. Honor a raised nprobe here too (the §3.32 CONTROL: unfiltered search
            # at high nprobe, to isolate whether the in-index win is species filtering vs just
            # scanning more IVF cells). nprobe=1 keeps the original fast unfiltered default.
            if nprobe and nprobe != 1:
                print(f"[search_with_filters] No filters, unfiltered search at nprobe={nprobe}", flush=True)
                sys.stdout.flush()
                try:
                    faiss.omp_set_num_threads(1)
                except Exception:
                    pass
                if query_vec.ndim == 1:
                    query_vec = query_vec.reshape(1, -1)
                params = faiss.SearchParametersIVF(nprobe=nprobe)
                return self.index.search(query_vec, k, params=params)
            print("[search_with_filters] No filters applied, using unfiltered search", flush=True)
            sys.stdout.flush()
            return self.search(query_vec, k)

    
    def search(self, query_vec, k):
        """
        Perform KNN search without any geospatial filtering.
        
        :param query_vec: A NumPy array of shape (1, d) or (d,) containing the query embedding.
        :param k: Number of nearest neighbors.
        :return: (D, I) where D are distances, I are Faiss IDs.
        """
        import time
        import sys
        t_start = time.time()
        print(f"[EnhancedSearchIndex.search] Starting unfiltered search for k={k} neighbors", flush=True)
        sys.stdout.flush()
        
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        
        print(f"[EnhancedSearchIndex.search] Query shape: {query_vec.shape}", flush=True)
        print(f"[EnhancedSearchIndex.search] Index type: {type(self.index).__name__}", flush=True)
        print(f"[EnhancedSearchIndex.search] Index ntotal: {self.index.ntotal}", flush=True)
        sys.stdout.flush()
        
        # Check if this is an IVF index that needs training
        if hasattr(self.index, 'nprobe'):
            print(f"[EnhancedSearchIndex.search] IVF index detected, current nprobe: {self.index.nprobe}", flush=True)
            # Use very aggressive nprobe for speed
            self.index.nprobe = 1
            print(f"[EnhancedSearchIndex.search] Set nprobe to 1 for maximum speed", flush=True)
        
        if hasattr(self.index, 'hnsw'):
            print(f"[EnhancedSearchIndex.search] HNSW index detected", flush=True)
            if hasattr(self.index.hnsw, 'efSearch'):
                print(f"[EnhancedSearchIndex.search] Current efSearch: {self.index.hnsw.efSearch}", flush=True)
                self.index.hnsw.efSearch = 8  # Very aggressive for speed
                print(f"[EnhancedSearchIndex.search] Set efSearch to 8 for maximum speed", flush=True)
        
        sys.stdout.flush()
        print(f"[EnhancedSearchIndex.search] About to call faiss index.search...", flush=True)
        sys.stdout.flush()
        
        # Set OpenMP threads to 1 to avoid thread pool hangs
        try:
            old_threads = faiss.omp_get_max_threads()
            faiss.omp_set_num_threads(1)
            print(f"[EnhancedSearchIndex.search] Set OMP threads from {old_threads} to 1", flush=True)
            sys.stdout.flush()
        except Exception as e:
            print(f"[EnhancedSearchIndex.search] Could not set OMP threads: {e}", flush=True)
        
        t_faiss_start = time.time()
        try:
            print(f"[EnhancedSearchIndex.search] Calling self.index.search(query_vec, {k})...", flush=True)
            sys.stdout.flush()
            D, I = self.index.search(query_vec, k)
            t_faiss_elapsed = time.time() - t_faiss_start
            print(f"[EnhancedSearchIndex.search] FAISS search completed in {t_faiss_elapsed:.3f}s", flush=True)
        except Exception as e:
            print(f"[EnhancedSearchIndex.search] FAISS search failed: {e}", flush=True)
            raise
        
        print(f"[EnhancedSearchIndex.search] Search completed, found {len(I[0])} results", flush=True)
        t_total = time.time() - t_start
        print(f"[EnhancedSearchIndex.search] Total time: {t_total:.3f}s", flush=True)
        sys.stdout.flush()
        return D, I


# Example usage
# def test_search_index():
#     import os

#     data_path = "/data/vision/beery/natural_retrieval/inaturalist"
#     index_name = "siglip-so400m-14-384--290M-lg" 
#     index_path = os.path.join(data_path, "index", index_name, "knn.index")

#     index = EnhancedSearchIndex(index_path, emb_to_inat_index=None)

#     # Example query without geo
#     print("Query without geo")
#     query = np.random.randn(1152).astype(np.float32)
#     k = 10
#     D, I = index.search(query, k)
#     print(D, I)

#     # Example query with geo
#     print("Query with geo")

#     from all_clip import load_clip
#     import torch
#     device = 'cpu'
#     model_name = 'open_clip:ViT-SO400M-14-SigLIP-384/webli' 
#     model, preprocess, tokenizer = load_clip(model_name, device='cpu', use_jit=False)
#     model.eval()

#     from utils import MemoryMappedMetadataProvider
#     metadata_name = "siglip-so400m-14-384"
#     meta = MemoryMappedMetadataProvider(os.path.join(data_path, "embs", metadata_name, "metadata"))

#     def get_text_embeddings(query_text):
#         text_inputs = torch.cat([tokenizer(query_text)]).to(device)
#         with torch.no_grad():
#             text_features = model.encode_text(text_inputs)
#             text_features /= text_features.norm(dim=-1, keepdim=True)
            
#         return text_features
    
#     query = get_text_embeddings('cat').cpu().numpy()

#     # min_lat, max_lat = 30, 35
#     # min_lng, max_lng = -82, -80
#     min_lat, max_lat = 42.31, 42.38
#     min_lng, max_lng = -71.14, -71.01
#     D, I = index.search_with_geo_filter(query, k, min_lat, max_lat, min_lng, max_lng)
#     print("final results:")
#     print('IDs:', I[0])
#     photo_ids = [meta.get(photo_index)[0] if photo_index != -1 else -1 for photo_index in I[0] ]
#     print('Photo IDs:', photo_ids)
#     print('Distances:', D[0])



def run_search_service():

    search_service = Flask(__name__)
    enhanced_index = None

    @search_service.route('/search', methods=['POST'])
    def search_endpoint():
        data = request.get_json()
        query_vec = np.array(data["query_vec"], dtype=np.float32)
        k = data["k"]
        taxon_id = data.get("taxon_id", None)
        geo_filter = data.get("geo_filter", None)
        month_filter = data.get("month_filter", None)
        license_filter = data.get("license_filter", None)
        D, I = enhanced_index.search_with_filters(query_vec, k, geo_filter=geo_filter,
                                                  taxon_id=taxon_id, months=month_filter,
                                                  licenses=license_filter)
        return jsonify({"distances": D.tolist(), "indices": I.tolist()})
    
    # Initialize enhanced_index here:
    data_path = "/data/vision/beery/natural_retrieval/inaturalist"
    index_name = "siglip-so400m-14-384--290M-lg"
    index_path = os.path.join(data_path, "index", index_name, "knn.index")
    enhanced_index = EnhancedSearchIndex(index_path, emb_to_inat_index=None)

    from waitress import serve
    print("Running on port 5003...")
    serve(search_service, host="0.0.0.0", port=5002)
    # search_service.run(host='0.0.0.0', port=5003)
    

if __name__ == "__main__":
    run_search_service()