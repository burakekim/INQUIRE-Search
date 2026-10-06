import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env file

class Config:
    # Flask config
    # The secret key is used to securely store session data (i.e. the user ID)
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY')
    ENV = os.environ.get('FLASK_ENV', 'development')
    LOGIN_REQUIRED = os.environ.get('LOGIN_REQUIRED', 'False').lower() == 'true'
    USE_SEARCH_SERVICE = False  # Set to False to run all core searches locally via FAISS

    MODEL_NAME = 'open_clip:ViT-SO400M-14-SigLIP-384/webli'
    METADATA_NAME = "siglip-so400m-14-384"
    INDEX_NAME = "siglip-so400m-14-384--290M-lg"

    # --- THE RETRIEVAL SCALE SWITCH ---
    RUN_MODE = 'production' 

    if RUN_MODE == 'production':
        DATA_PATH = "/data/vision/beery/scratch/eddie/natural_retrieval/inaturalist"
        INDEX_PATH = os.path.join(DATA_PATH, "index", INDEX_NAME, "knn.index")
        
        # Disable raw embedding loads to protect system RAM
        EMBEDDINGS_PATH = None 
        
        METADATA_PATH = os.path.join(DATA_PATH, "embs", METADATA_NAME, "metadata")
        MAPPINGS_DIR = os.path.join(DATA_PATH, "notebooks")
        PHOTO_ID_TO_EXT_MAP_PATH = os.path.join(MAPPINGS_DIR, "photo_id_to_ext_siglip_so400m.npy")
        # The same map as two sorted arrays (~0.8G in RAM instead of a ~40G Python dict); used when
        # present. Built 2026-10-02 by streaming the pickle above: 289,551,709 ids, the 158.6M
        # non-'jpg' ones stored, every other id is 'jpg'.
        PHOTO_ID_TO_EXT_COMPACT_PATH = "/data/vision/beery/scratch/burake/searchin/data/derived/photo_id_to_ext_compact.npz"
        IMAGE_ID_TO_TAXON_ID_PATH = os.path.join(MAPPINGS_DIR, "image_id_to_taxon_id.npy")
        TAXA_DF_PATH = os.path.join(MAPPINGS_DIR, "final_taxa_with_names.csv")
        GEO_INDEX_PATH = os.path.join(MAPPINGS_DIR, 'image_id_to_latlong_int.npy')
        MONTH_INDEX_PATH = os.path.join(MAPPINGS_DIR, "image_id_to_month.npy")
        TAXA_FILE = os.path.join(MAPPINGS_DIR, "species_names.json")
        FILTERS_DIR = MAPPINGS_DIR

        # Built via backend/build_license_index.py (see that file for why the
        # chunk-concatenation order matters for alignment with FAISS ids).
        LICENSE_INDEX_PATH = "/data/vision/beery/scratch/burake/searchin/data/inputs/inat_data/image_id_to_license.npy"
        LICENSE_CODE_MAP_PATH = "/data/vision/beery/scratch/burake/searchin/data/inputs/inat_data/license_code_to_name.json"
   
    else:
        # Sandbox / Development Paths (Keep your old fallback here)
        DATA_PATH = "static"
        INDEX_PATH = os.path.join(DATA_PATH, "index", "knn.index")
        EMBEDDINGS_PATH = os.path.join(DATA_PATH, "embeddings", "embeddings_0.npy")
        METADATA_PATH = os.path.join(DATA_PATH, "metadata")
        PHOTO_ID_TO_EXT_MAP_PATH = os.path.join(DATA_PATH, "mappings", "siglip_so400m_patch14_384_inat24_rerank_img_mapping.npy")
        IMAGE_ID_TO_TAXON_ID_PATH = os.path.join(DATA_PATH, "mappings", "map_taxon_id_to_image_ids.npy")
        GEO_INDEX_PATH = os.path.join(DATA_PATH, 'mappings', 'image_id_to_latlong_int.npy')
        MONTH_INDEX_PATH = os.path.join(DATA_PATH, "mappings", "image_id_to_month.npy")
        FILTERS_DIR = os.path.join(DATA_PATH, "mappings")
        TAXA_FILE = os.path.join(DATA_PATH, "mappings", "species_name_to_taxon_id.npy")