import os

# Ensure Hugging Face caches are placed in a writable local folder instead of a home directory that may be mounted read-only.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRANSFORMERS_CACHE = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
if not os.environ.get("TRANSFORMERS_CACHE"):
    os.environ["TRANSFORMERS_CACHE"] = DEFAULT_TRANSFORMERS_CACHE
if not os.environ.get("HUGGINGFACE_HUB_CACHE"):
    os.environ["HUGGINGFACE_HUB_CACHE"] = DEFAULT_TRANSFORMERS_CACHE
if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = DEFAULT_TRANSFORMERS_CACHE
os.makedirs(DEFAULT_TRANSFORMERS_CACHE, exist_ok=True)

# macOS workaround: Set KMP_DUPLICATE_LIB_OK to avoid OpenMP duplicate library errors
try:
    if os.uname().sysname == 'Darwin' and os.environ.get('KMP_DUPLICATE_LIB_OK', '').lower() not in ('true', '1', 'yes'):
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        print("Warning: KMP_DUPLICATE_LIB_OK not set — temporarily setting to TRUE to avoid libomp duplicate abort on macOS.")
        print("Permanent fix: uninstall duplicate libomp (e.g. `brew uninstall libomp`) or use a single conda/python environment.")
except Exception:
    pass

from functools import wraps
import ipaddress
import json
import logging
import time
import uuid
import psutil
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from flask import Flask, render_template, request, jsonify, session, redirect

import torch
from all_clip import load_clip

from services.search_service import SearchService
from config import Config


import numpy as np

def make_serializable(obj):
    """Recursively converts numpy data types into native Python types for JSON transfer."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return make_serializable(obj.tolist())
    return obj

# Initialize the Flask application
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Load configuration
app.secret_key = Config.SECRET_KEY
app.env = Config.ENV

db = None


class EmbeddingService:
    def __init__(self, model_name):
        """Initialize the embedding service with the specified model name.
        Example model names:
        - 'open_clip:ViT-L-16-SigLIP-384/webli'
        - 'open_clip:ViT-SO400M-14-SigLIP-384/webli'
        - 'open_clip:ViT-B-32/openai'
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using device:', self.device)
        self.model_name = model_name
        self.model, self.preprocess, self.tokenizer = self._load_model()

    def _load_model(self):
        model, preprocess, tokenizer = load_clip(self.model_name,
                                                 device='cpu',
                                                 use_jit=False)
        model = model.to(self.device)
        model.eval()
        return model, preprocess, tokenizer

    def get_text_embeddings(self, query_text):
        """Generate text embeddings for a given query"""
        text_inputs = torch.cat([self.tokenizer(query_text)]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu()


# Load CLIP Model
print("Downloading Vision-Language model...")
embedding_service = EmbeddingService(Config.MODEL_NAME)
print("Loaded Vision-Language model.")

search_service = SearchService()
print("Loaded Search Service.")

# Executor used to run potentially blocking embedding/search work with timeouts
_executor = ThreadPoolExecutor(max_workers=2)


def session_id_required(func):
    @wraps(func)
    def id_wrapper(*args, **kwargs):
        if 'user_id' not in session:
            session['user_id'] = uuid.uuid4()
        return func(*args, **kwargs)
    return id_wrapper


# This decorated prints the elapsed time of each function
def print_elapsed_time(func):
    @wraps(func)
    def time_wrapper(**kwargs):
        start_time = time.perf_counter()
        result = func(**kwargs)
        time_in_ms = int((time.perf_counter() - start_time) * 1000)
        app.logger.info("\t%s ms for function %s" % (time_in_ms, func.__name__))
        return result
    return time_wrapper


# With LOGIN_REQUIRED=true (set on the public VM, 2026-10-06), every route but the login pages needs
# the key, except from inside CSAIL: the search page's own API calls carry the session cookie, while
# the loop UI and scripts call /process_query from CSAIL without one. A TCP forwarder inside CSAIL
# (like the old bobolink proxy) would therefore let everything through.
_NO_KEY_NEEDED = [ipaddress.ip_network(n) for n in ("127.0.0.0/8", "128.30.0.0/16", "128.52.0.0/16")]


@app.before_request
def require_login():
    if not Config.LOGIN_REQUIRED or session.get('logged_in'):
        return None
    if request.endpoint in ('home', 'login', 'login_request', 'get_login', 'post_login', 'static'):
        return None
    # Behind the VM's Caddy front door every request arrives from 127.0.0.1, so take the visitor's
    # address from the X-Forwarded-For that Caddy sets. Only trusted when the request comes from
    # localhost: a header sent straight to :5002 from outside is ignored.
    ip = request.remote_addr
    if ip == '127.0.0.1' and request.headers.get('X-Forwarded-For'):
        ip = request.headers['X-Forwarded-For'].split(',')[-1].strip()
    if any(ipaddress.ip_address(ip) in n for n in _NO_KEY_NEEDED):
        return None
    return jsonify({'error': 'login required'}), 401


@app.route('/')
@session_id_required
def home():
    print('user id (index):', session['user_id'])
    # If not logged in, redirect to login page
    if 'logged_in' not in session or not session['logged_in']:
        return redirect("/login", code=302)
    return render_template('index.html')


@app.route('/login', methods=['GET'])
@session_id_required
def login():
    # If already logged in, redirect to query page
    if 'logged_in' in session and session['logged_in']:
        return redirect("/", code=302)
    print('[LOGIN] user id:', session['user_id'])
    return render_template('login.html')


@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('dashboard.html')


@app.route('/api/device_info', methods=['GET'])
def device_info():
    return jsonify({
        'mem': psutil.virtual_memory().percent,
        'cpu': psutil.cpu_percent()
    })


@app.route('/api/license_options', methods=['GET'])
def license_options():
    """License codes present in the index, for populating the license filter UI."""
    names = sorted(set(search_service.license_code_to_name.values()))
    return jsonify(licenses=names)


@app.route('/login', methods=['POST'])
@session_id_required
def login_request():
    print(request.form)
    print('[LOGIN] user id:', session['user_id'])
    key = request.form.get('user_input')
    if key == 'bird':
        session['logged_in'] = True
        return redirect("/", code=302)
    else:
        return render_template('login.html', error='Incorrect key.')


def _run_search(embedding, k, filters, label, timeout=120):
    """Submit a search_service.search call to the shared executor with a timeout,
    logging under `label`. Returns (results, None) on success or (None, error_response)."""
    t_search_start = time.time()
    app.logger.info(f"[{label}] Submitting search to executor...")
    search_future = _executor.submit(search_service.search, embedding, k, filters)
    try:
        app.logger.info(f"[{label}] Waiting for search result (timeout={timeout}s)...")
        results = search_future.result(timeout=timeout)
        t_search_elapsed = time.time() - t_search_start
        app.logger.info(f"[{label}] Search completed in {t_search_elapsed:.2f}s, returned {len(results.get('id', []))} results")
        return results, None
    except FuturesTimeoutError:
        app.logger.error(f"[{label}] Search timed out after {time.time()-t_search_start:.2f}s")
        return None, (jsonify({"error": f"Search timed out after {timeout}s. Try reducing k or check index optimization."}), 504)
    except Exception as e:
        app.logger.exception(f"[{label}] Search failed after {time.time()-t_search_start:.2f}s: {e}")
        return None, (jsonify({"error": str(e)}), 500)


@app.route('/process_query', methods=['POST'])
@session_id_required
@print_elapsed_time
def process_query():
    # Get search parameters
    user_query = request.form.get('user_input')
    k = int(request.form.get('image_count'))

    # Parse filters
    try:
        filters = json.loads(request.form.get('filters', '{}'))
    except Exception as e:
        print(f"Error parsing filters: {e}")
        filters = {}
    # Log filters for debugging filter application
    app.logger.info(f"Received filters from frontend: {filters}")

    # Generate embeddings (run in thread with timeout to avoid hanging the request)
    t_start = time.time()
    app.logger.info(f"[process_query] START - Received query='{user_query}' k={k} filters={filters}")
    try:
        t_emb_start = time.time()
        app.logger.info("[process_query] Submitting embedding generation to executor...")
        emb_future = _executor.submit(embedding_service.get_text_embeddings, user_query)
        app.logger.info("[process_query] Waiting for embedding result (timeout=20s)...")
        emb_result = emb_future.result(timeout=20)
        # embedding_service.get_text_embeddings returns a torch tensor; normalize to numpy
        try:
            question_embedding = emb_result.cpu().numpy()
        except Exception:
            # If it's already numpy
            question_embedding = emb_result
        t_emb_elapsed = time.time() - t_emb_start
        app.logger.info(f"[process_query] Embedding generation completed in {t_emb_elapsed:.2f}s, shape={question_embedding.shape}")
    except FuturesTimeoutError:
        app.logger.error(f"[process_query] Embedding generation timed out after {time.time()-t_emb_start:.2f}s")
        return jsonify({"error": "Embedding generation timed out"}), 504
    except Exception as e:
        app.logger.exception(f"[process_query] Embedding generation failed after {time.time()-t_emb_start:.2f}s: {e}")
        return jsonify({"error": str(e)}), 500

    # Perform search (possibly filtered) — also run in thread with timeout
    results, err = _run_search(question_embedding, k, filters, "process_query")
    if err is not None:
        return err

    t_total = time.time() - t_start
    app.logger.info(f"[process_query] COMPLETE - Total time: {t_total:.2f}s")
    
    # Debug: Check species data
    print(f"[process_query] Species data: {len(results.get('species', []))} items", flush=True)
    print(f"[process_query] First 3 species: {results.get('species', [])[:3]}", flush=True)
    print(f"[process_query] Non-empty species: {sum(1 for s in results.get('species', []) if s)}", flush=True)

    safe_results = make_serializable(results)

    return jsonify(
        img_ids=safe_results['id'],
        img_urls=safe_results['img_url'],
        img_names=safe_results['file_name'],
        retrieval_scores=safe_results['scores'],
        species=safe_results['species'],
        # E51: the per-photo identity. `species` above is a dev-only fallback that is empty in
        # production; these three are the real thing, resolved server-side against
        # final_taxa_with_names.csv. The review grid shows them at the moment of judgement.
        taxon_ids=safe_results.get('taxon_id'),
        taxon_names=safe_results.get('taxon_name'),
        taxon_commons=safe_results.get('taxon_common'),
        latitudes=safe_results['latitudes'],
        longitudes=safe_results['longitudes'],
        months=safe_results['months'],
        licenses=safe_results['licenses'],
        # E37 Phase 5: what actually served this request, so the caller can log it per draw.
        filter_mode=safe_results.get('filter_mode'),
        filter_pool_size=safe_results.get('filter_pool_size'),
        nprobe_used=safe_results.get('nprobe'),
        nprobe_requested=safe_results.get('nprobe_requested'),   # E216: differs when the search escalated
    )


@app.route('/search_vec', methods=['POST'])
def search_vec():
    """Image->image (or raw-embedding) search: bypass text encoding and query the 290M index
    directly with a precomputed, L2-normalized SigLIP embedding. Enables query-by-example /
    pseudo-relevance feedback over the index (EXPERIMENTS 3.20 two-stage retrieval).
    Body: {"query_vec": [...] (D,) or (1,D), "k": int, "filters": {...}}."""
    data = request.get_json(force=True, silent=True) or {}
    try:
        vec = np.asarray(data["query_vec"], dtype=np.float32)
        if vec.ndim == 1:
            vec = vec[None, :]
        k = int(data.get("k", 100))
        filters = data.get("filters", {}) or {}
    except Exception as e:
        return jsonify({"error": f"bad payload: {e}"}), 400
    results, err = _run_search(vec, k, filters, "search_vec")
    if err is not None:
        return err
    safe = make_serializable(results)
    return jsonify(img_ids=safe['id'], img_urls=safe['img_url'],
                   retrieval_scores=safe['scores'], species=safe.get('species', []))


@app.route("/api/autocomplete", methods=["GET"])
def autocomplete():
    query = request.args.get("query", "").lower()
    print('>> query:', query)

    if not query:
        return jsonify([])

    matches = search_service.get_autocomplete_suggestions(query,
                                                          max_matches=15)
    return jsonify(matches)


@app.route('/api/add_flagged', methods=['POST'])
@session_id_required
def add_flagged():
    raw_data = request.form.get('data')
    try:
        data = json.loads(raw_data)
        print(data)
        flagged_image_data = data['images']
        filter = data['filter']
        time = data['time']
        query = data['query']

        for im in flagged_image_data:
            assert 'id' in im, 'no id found in image data'
            assert 'file_name' in im, 'no file name found in image data'
            assert 'species' in im, 'no species found in image data'
            assert 'url' in im, 'no url found in image data'
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    flagged_image_list = []
    for im in flagged_image_data:
        flagged_im = dict(id=int(im['id']), url=im['url'],
                          file_name=im['file_name'], species=im['species'],
                          time=time, query=query, filter=filter)
        flagged_image_list.append(flagged_im)

    db.add_flagged_images(flagged_image_list)

    return jsonify({"message": "Data saved to database."})


@app.route('/api/get_captions', methods=['POST'])
@session_id_required
@print_elapsed_time
def get_captions():
    batch = request.json['batch']
    print(batch)
    json_file_path = f'caption_data/generated_captions_{batch}.json'
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    return jsonify(data)


@app.route('/api/get_caption_labels', methods=['GET'])
@session_id_required
@print_elapsed_time
def get_caption_labels():
    json_file_path = 'caption_data/caption_labels.json'
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
    else:
        data = []
    return jsonify(data)


@app.route('/api/get_caption_batches', methods=['GET'])
@session_id_required
@print_elapsed_time
def get_caption_batches():
    data_files = [d for d in os.listdir('caption_data')
                  if d.startswith('generated_captions_')]
    batches = [d.split('_')[-1].split(".")[0] for d in data_files]
    return jsonify(batches)


@app.route('/api/submit_caption_labels', methods=['POST'])
@session_id_required
@print_elapsed_time
def submit_caption_labels():
    print('user id (submit):', session['user_id'])

    data_to_submit = request.json
    print(data_to_submit)

    id = request.json['id']
    images = request.json['images']

    json_file_path = 'caption_data/caption_labels.json'
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    data[id] = images
    with open(json_file_path, 'w') as json_file:
        data = json.dump(data, json_file)

    return jsonify({"message": "saved data"})


@app.route('/submit_data', methods=['POST'])
@session_id_required
@print_elapsed_time
def submit_data():
    print('user id (submit):', session['user_id'])
    data_to_submit = request.form.get('data')
    if data_to_submit:
        try:
            data = json.loads(data_to_submit)
            data['session_id'] = session['user_id'].hex
            # create the directory if it does not exist
            os.makedirs('query_annotation', exist_ok=True)
            # Define the path to the JSON file where you want to save the data
            if 'relevantImages' in data:
                json_file_path = 'query_annotation/retrieved_relevant.json'
            elif 'retrievedImages' in data:
                json_file_path = 'query_annotation/retrieved_no_relevant.json'
            else:
                json_file_path = 'query_annotation/all_queries.json'
            # Load existing data from the JSON file (if any)
            existing_data = []
            try:
                with open(json_file_path, 'r') as json_file:
                    existing_data = json.load(json_file)
            except FileNotFoundError:
                pass

            existing_data.append(data)

            # Write the updated data back to the JSON file
            with open(json_file_path, 'w') as json_file:
                json.dump(existing_data, json_file)
            return jsonify({"message": "Data saved to JSON file."})
        except Exception as e:
            return jsonify({"error": str(e)})
    else:
        return jsonify({"error": "No data received."})


@app.route('/api/get_data', methods=['GET'])
@session_id_required
@print_elapsed_time
def get_data():
    json_file_path = 'query_annotation/retrieved_relevant.json'
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
    else:
        data = []
    return jsonify(data)


@app.route('/api/get_queries', methods=['GET'])
@session_id_required
@print_elapsed_time
def get_queries():
    json_file_path = 'query_annotation/all_queries.json'
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)
    else:
        data = []
    return jsonify(data)


@app.route('/api/get_flagged', methods=['GET'])
@session_id_required
def get_flagged():
    data = db.get_flagged_images()
    return jsonify(data)


@app.route('/api/login', methods=['GET'])
@session_id_required
def get_login():
    user_is_logged_in = ('logged_in' in session and session['logged_in'])
    print('user', session['user_id'], ' is logged in?', user_is_logged_in)
    return jsonify({'logged_in': user_is_logged_in})


@app.route('/api/login', methods=['POST'])
@session_id_required
def post_login():
    print(request.form)
    print('[LOGIN] user id:', session['user_id'])
    key = request.form.get('user_input')
    if key == 'bird':
        session['logged_in'] = True
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Invalid login key'})


if __name__ == '__main__':
    from waitress import serve
    # Port is configurable via BACKEND_PORT (default 5002) so the same app can be
    # served on any unprivileged port for sharing. host=0.0.0.0 -> publicly reachable.
    port = int(os.environ.get("BACKEND_PORT", 5002))
    print(f"Running on port {port}...")
    serve(app, host="0.0.0.0", port=port)
