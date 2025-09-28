import os, requests, logging, subprocess, re, sys, time, math
from ece461.API import llm_api
from huggingface_hub import hf_hub_download, ModelCard, model_info
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub.hf_api import ModelInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union, TypedDict, List
from ece461.url_file_parser import ModelLinks 

# ---- Metric calculation result data type ----
class MetricValue(TypedDict, total=False):
    score: Optional[float]
    latency_ms: Optional[float]
    details: Dict[str, Any]
    ok: bool

# ---- What metric functions may return (to keep adapters trivial) ----
ReturnType = Union[Tuple[float, float], Dict[str, Any], float, int, None]

# ---- Registry via decorator ----
REGISTRY: Dict[str, Callable[..., ReturnType]] = {}

def metric(name: Optional[str] = None) -> Callable[[Callable[..., ReturnType]], Callable[..., ReturnType]]:
    """
    Decorator to register a metric function.
    Args:
        name (Optional[str]): Optional name to register the metric under. If None, use function name.
    """
    def wrap(fn: Callable[..., ReturnType]) -> Callable[..., ReturnType]:
        key = name or fn.__name__
        if key in REGISTRY:
            raise ValueError(f"Metric '{key}' already registered")
        REGISTRY[key] = fn
        return fn
    return wrap

# ---- Normalization: unify return shapes so the dict is predictable ----
def normalize(ret: ReturnType) -> MetricValue:
    """
    Normalize various return types into a standard MetricValue dict.
    Args:
        ret (ReturnType): The raw return value from a metric function.
    """
    out: MetricValue = {"ok": True, "details": {}}
    # Check the type of ret and extract score and latency_ms accordingly
    # Check for tuple of two numbers first
    if isinstance(ret, tuple) and len(ret) == 2 and all(isinstance(x, (int, float)) for x in ret):
        score, latency_ms = float(ret[0]), float(ret[1])
        out.update({"score": score, "latency_ms": latency_ms})
    # Check for dictionary (HW metric)
    elif isinstance(ret, tuple) and isinstance(ret[0], dict):
        details = dict(ret[0])
        latency_ms = details.get("latency_ms", float(ret[1]))
        out.update({
            "score": None,
            "latency_ms": float(latency_ms) if isinstance(latency_ms, (int, float)) else None,
            "details": details
        })
    # Edge cases
    else:
        out.update({"ok": False, "score": None, "latency": None})
    
    return out

# ---- Parallel runner: returns dict[name] -> MetricValue ----
def run_metrics(
    model: ModelLinks,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> Dict[str, MetricValue]:
    """
    Run selected metrics in parallel and return their results.
    Args:
        model_id (str): The model identifier (e.g., "bert-base-uncased").
        include (Optional[Iterable[str]]): Metric names to include. If None, include all.
        exclude (Optional[Iterable[str]]): Metric names to exclude. If None, exclude none.
        max_workers (Optional[int]): Max number of parallel workers. Defaults to min(32, os.cpu_count() * 5).
    """
    selected = {k: v for k, v in REGISTRY.items()}
    if include is not None:
        inc = {n.strip() for n in include}
        selected = {k: v for k, v in selected.items() if k in inc}
    if exclude is not None:
        exc = {n.strip() for n in exclude}
        selected = {k: v for k, v in selected.items() if k not in exc}
    if not selected:
        logging.warning("No metrics selected to run.")
        return {}
    
    # Define number of workers
    cpu = os.cpu_count() or 4
    max_workers = min(32, cpu * 5)

    def call(fn: Callable[..., ReturnType]) -> MetricValue:
        try:
            # Try calling with model_id first
            ret = fn(model=model)  # type: ignore[arg-type]
            return normalize(ret)
        except Exception as e:
            logging.exception("Metric crashed")
            return {"ok": False, "score": None, "latency_ms": None, "details": {}}

    out: Dict[str, MetricValue] = {}
    # Start parallel execution of metrics
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        logging.info("Running %d metrics in parallel with %d workers", len(selected), max_workers)
        result = {pool.submit(call, fn): name for name, fn in selected.items()}
        for fut in as_completed(result):
            out[result[fut]] = fut.result()

    logging.info("Completed running metrics")
    print(out)
    
    return out

# ---- Metric implementations ----
@metric("ramp_up")
def calculate_ramp_up_metric(model: ModelLinks) -> tuple[float, float]:
    """
        Calculate ramp-up time metric.
    """
    # Start latency calculation
    start_time = time.perf_counter()

    # Initiate Ramp-up metric calculation
    readme_data = fetch_readme_content(model.model_id)
    if readme_data == "":
        logging.info("No README content found for model %s", model.model_id)
        score = 0.0
    else:
        prompt = build_ramp_up_prompt(readme_data)
        response = llm_api.query_llm(prompt)
        logging.debug("Ramp-up LLM response: %s", response)
        # Extract the ramp_up_score from the JSON response
        try:
            m = re.search(r'"ramp_up_score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', response)
            extracted = float(m.group(1))
            if extracted < 0.0:
                score = 0.0
            elif extracted > 1.0:
                score = 1.0
            else:
                score = extracted
        except:
            score = 0.0
            logging.error("Unexpected LLM response format for model %s: %s", model.model_id, response)
    
    logging.info("Ramp-up metric LLM score for model %f", score)
    # End latency calculation
    end_time = time.perf_counter()
    latency = (end_time - start_time) * 1000  # Convert to milliseconds
    logging.info("Ramp-up metric latency for model %s: %.2f ms", model.model_id, latency)
    
    return (score, round(latency))

def get_doc_score(model_card_text: str) -> float:
    """Calculates the documentation score based on the README content."""
    if not model_card_text:
        return 0.0
    
    length: int = len(model_card_text)
    # Check for presence of key sections, a good proxy for quality
    key_sections: List[str] = ["how to use", "limitations", "training", "evaluation"]
    num_key_sections: int = sum(section in model_card_text.lower() for section in key_sections)

    if length > 1500 and num_key_sections >= 2:
        return 1.0
    elif length > 500:
        return 0.6
    elif length > 100:
        return 0.2
    return 0.1

def get_author_score(author: str) -> float:
    """Calculates the author authority score."""
    # List of well-known organizations
    known_orgs: List[str] = ["google", "meta", "microsoft", "openai"]
    if author in known_orgs:
        return 1.0
    # Add other checks here if needed, e.g., for GTE-validated users
    return 0.5 # Default for individual users

def get_repro_score(filenames: List[str]) -> float:
    """Calculates the reproducibility score based on repository files."""
    has_weights: bool = any(f.endswith(('.bin', '.safetensors')) for f in filenames)
    has_config: bool = 'config.json' in filenames
    
    if not has_weights:
        return 0.0

    score: float = 0.6 if has_config else 0.2
    
    # Bonus points for files that aid reproducibility
    if 'training_args.bin' in filenames or 'trainer_state.json' in filenames:
        score = min(1.0, score + 0.2)
    if 'eval_results.json' in filenames:
        score = min(1.0, score + 0.2)
        
    return score

def get_community_score(downloads: Optional[int]) -> float:
    """Calculates the community score based on downloads using a log scale."""
    if downloads is None or downloads == 0:
        return 0.0
    return round(min(1.0, math.log10(downloads + 1) / 6), 2)

@metric("bus-factor")
def calculate_bus_factor(model: ModelLinks) -> tuple[float, float]:
    """
    Calculates the Model Resilience Score for a given Hugging Face model repository.
    """
    info: ModelInfo = model_info(model.model_id, files_metadata=True)
    
    start_time = time.perf_counter()
    readme_content: str = info.cardData.get('text', '') if info.cardData else ''
    author: str = info.author
    filenames: List[str] = [f.rfilename for f in info.siblings]
    downloads: Optional[int] = info.downloads

    s_doc: float = get_doc_score(readme_content)
    s_author: float = get_author_score(author)
    s_repro: float = get_repro_score(filenames)
    s_community: float = get_community_score(downloads)
    
    weights: Dict[str, float] = {
        'doc': 0.35,
        'author': 0.30,
        'repro': 0.25,
        'community': 0.10
    }
    
    final_score: float = (weights['doc'] * s_doc +
                          weights['author'] * s_author +
                          weights['repro'] * s_repro +
                          weights['community'] * s_community)
                   
    end_time = time.perf_counter()
    latency = (end_time - start_time) * 1000  # Convert to milliseconds
    logging.info("Ramp-up metric latency for model %s: %.2f ms", model.model_id, latency)
    return (round(final_score, 2), round(latency))

@metric("license")
def calculate_license_metric(model: ModelLinks) -> tuple[float, float]:
    """
        Calculate license compatibility score.
    """
    # Start latency calculation
    start_time = time.perf_counter()

    # Initiate license metric calculation
    model_card_data = fetch_model_card_content(model.model_id)
    if model_card_data == "":
        logging.info("No model card content found for model %s", model.model_id)
        score = 0.0
    else:
        prompt = build_license_prompt(model_card_data)
        response = llm_api.query_llm(prompt)
        logging.debug("License LLM response: %s", response)
        # Extract the license_score from the JSON response
        try:
            m = re.search(r'"license_score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', response)
            extracted = float(m.group(1))
            if extracted < 0.0:
                score = 0.0
            elif extracted > 1.0:
                score = 1.0
            else:
                score = extracted
        except:
            score = 0.0
            logging.error("Unexpected LLM response format for model %s: %s", model.model_id, response)
    
    logging.info("License metric LLM score for model %f", score)
    # End latency calculation
    end_time = time.perf_counter()
    latency = (end_time - start_time) * 1000  # Convert to milliseconds
    logging.info("License metric latency for model %s: %.2f ms", model.model_id, latency)
    
    return (round(score, 2), round(latency))

@metric("performance")
def calculate_performance_metric(model: ModelLinks) -> tuple[float, float]:
    """
        Calculate performance benchmark score.
    """
    # Start latency calculation
    start_time = time.perf_counter()

    # Initiate performance metric calculation
    model_card_data = fetch_model_card_content(model.model_id)
    if model_card_data == "":
        logging.info("No model card content found for model %s", model.model_id)
        score = 0.0
    else:
        prompt = build_performance_prompt(model_card_data)
        response = llm_api.query_llm(prompt)
        logging.debug("Performance LLM response: %s", response)
        # Extract the ramp_up_score from the JSON response
        try:
            m = re.search(r'"performance_score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', response)
            extracted = float(m.group(1))
            if extracted < 0.0:
                score = 0.0
            elif extracted > 1.0:
                score = 1.0
            else:
                score = extracted
        except:
            score = 0.0
            logging.error("Unexpected LLM response format for model %s: %s", model.model_id, response)
    
    logging.info("Performance metric LLM score for model %f", score)
    # End latency calculation
    end_time = time.perf_counter()
    latency = (end_time - start_time) * 1000  # Convert to milliseconds
    logging.info("Performance metric latency for model %s: %.2f ms", model.model_id, latency)

    return (round(score, 2), round(latency))

@metric("size")
def calculate_size_metric(model: ModelLinks) -> tuple[float, float]:
    """
        Calculate size compatibility scores for different hardware types.
    """
    # Start latency calculation
    start_time = time.perf_counter()
    try:
        total_size_mb = get_model_weight_size(model.model_id)
        score = calculate_hardware_compatibility_scores(total_size_mb)
        # End latency calculation
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        return (score, round(latency))
    except Exception as e:
        raise ValueError(f"Failed to calculate size metric for {model.model_id}: {str(e)}")

@metric("code_quality")
def calculate_code_quality(model: ModelLinks) -> tuple[float, float]:
    """
        Run pylint on Python files in 'path' and return normalized score.
    """
    # Start latency calculation
    start_time = time.perf_counter()

    if not model.code:
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000
        return (0.0, latency)

    proc = subprocess.run(
        [sys.executable, "-m", "pylint", "--exit-zero", "--score=y", model.code],
        capture_output=True, text=True
    )
    m = re.search(r"rated at\s+([0-9.]+)/10", proc.stdout)
    score10 = float(m.group(1)) if m else 0.0

    # End latency calculation
    end_time = time.perf_counter()
    latency = (end_time - start_time) * 1000  # Convert to milliseconds
    return (round(max(0.0, min(1.0, score10/10)), 2), round(latency))

@metric("dataset_and_code_quality")
def calculate_dataset_and_code_score(model: ModelLinks) -> tuple[float, float]:
    """
        Calculate the dataset and code quality score
    """
    start_time = time.perf_counter()
    if model.dataset is None and model.code is None:
        score = 0.0
    else:
        prompt = build_dataset_code_prompt(model.dataset, model.code)
        try:
            response = llm_api.query_llm(prompt)
            logging.debug("Dataset and code LLM response: %s", response)
            # Extract the score from the JSON response
            m = re.search(r'"dataset_code_score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', response)
            if m:
                extracted = float(m.group(1))
                score = max(0.0, min(1.0, extracted))
            else:
                logging.error("Could not extract dataset_code_score from LLM response")
                score = 0.0
        except Exception as e:
            logging.error("Error analyzing dataset and code: %s", e)
            score = 0.0
    
    logging.info("Dataset and code metric score: %.3f", score)
    
    # End latency calculation
    end_time = time.perf_counter()
    latency = round((end_time - start_time) * 1000)  # Convert to milliseconds
        
    return round(score, 2), round(latency)

@metric("dataset_quality")
def calculate_dataset_quality(model: ModelLinks) -> tuple[float, float]:
    """
        Calculate dataset quality metric based on dataset URL.
    """
    start_time = time.perf_counter()
    
    if model.dataset is None or not model.dataset.strip():
        score = 0.0
        logging.info("No dataset URL provided, score = 0.0")
    else:
        try:
            prompt = build_dataset_quality_prompt(model.dataset)
            response = llm_api.query_llm(prompt)
            logging.debug("Dataset quality LLM response: %s", response)
            
            # Extract the score from the JSON response
            m = re.search(r'"dataset_quality_score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', response)
            if m:
                extracted = float(m.group(1))
                score = max(0.0, min(1.0, extracted))
            else:
                logging.error("Could not extract dataset_quality_score from LLM response")
                score = 0.0
        except Exception as e:
            logging.error("Error analyzing dataset quality: %s", e)
            score = 0.0
    
    logging.info("Dataset quality metric score: %.3f", score)
    
    end_time = time.perf_counter()
    latency = round((end_time - start_time) * 1000)
    
    return round(score, 2), round(latency)

################################# Supporting Functions #################################
# License metric calculation
def build_license_prompt(model_card_excerpt: str) -> str:
    """
    Single prompt that tells the LLM to analyze the model card and also compute
    the final normalized score for license metric. It explicitly warns about empty 
    headers.
    """
    if not model_card_excerpt.strip():
        model_card_excerpt = "(no model card content provided)"
    return (
        "You evaluate a model's LICENSE based on the model card text.\n"
        "Return ONE JSON object and nothing else (no prose/markdown/fences). "
        "JSON schema (exactly this):\n"
        "{\n"
        '  "license_score": <float 0..1>,\n'
        '  "detected_license": "<SPDX or short name or null>",\n'
        '  "compatible_with_lgpl_2_1": true|false|null,\n'
        '  "confidence_0to1": <float 0..1>,\n'
        '  "rationale": "<short sentence>"\n'
        "}\n\n"
        "How to compute license_score (clamp to [0,1]):\n"
        "1) Compatibility (0..1) relative to LGPL-2.1 needs:\n"
        "   - 1.0: Permissive or weak-copyleft compatible with LGPL-2.1 (MIT, BSD-2/3, Apache-2.0, LGPL-2.1/3.0, MPL-2.0, CC-BY-4.0 for weights, OpenRAIL-M if commercial use allowed).\n"
        "   - 0.0: Non-commercial/research-only (CC-BY-NC, RAIL-NC), strong copyleft over network (AGPL-3.0), or custom terms restricting commercial redistribution.\n"
        "   - 0.3: Unclear/unknown.\n"
        "2) Clarity (0..1):\n"
        "   - 1.0: Explicit SPDX ID or LICENSE link/name present.\n"
        "   - 0.7: Clear license text in card but no SPDX or LICENSE file mentioned.\n"
        "   - 0.3: Vague wording (e.g., 'free for research') without explicit grant.\n"
        "   - 0.0: No license info.\n"
        "3) Final: license_score = clamp01(0.7 * compatibility + 0.3 * clarity).\n"
        "If multiple licenses apply (code vs weights), use the most restrictive when scoring. "
        "Do not invent licenses; be conservative.\n\n"
        "MODEL CARD:\n<<<\n" + model_card_excerpt + "\n>>>\n"
    )

# Dataset and Code metric calculation
def build_dataset_code_prompt(dataset_url: str, code_url: str) -> str:
    """
    Build LLM prompt for analyzing dataset and code availability and quality from URLs.
    """
    return (
        "You evaluate a model's DATASET AND CODE AVAILABILITY based on the provided URLs.\n"
        "Return ONE JSON object and nothing else (no prose/markdown/fences).\n"
        "JSON schema (exactly this):\n"
        "{\n"
        '  "dataset_code_score": <float 0..1>,\n'
        '  "has_dataset": true|false,\n'
        '  "has_code": true|false,\n'
        '  "dataset_quality": <float 0..1>,\n'
        '  "code_quality": <float 0..1>,\n'
        '  "rationale": "<short sentence>"\n'
        "}\n\n"
        "How to compute dataset_code_score (clamp to [0,1]):\n"
        "1) Dataset Availability (0.25): Does the model have a dataset URL provided?\n"
        "   - 0.25 if valid dataset URL present, 0.0 if None or invalid\n"
        "2) Code Availability (0.25): Does the model have a code URL provided?\n"
        "   - 0.25 if valid code URL present, 0.0 if None or invalid\n"
        "3) Dataset Quality (0.25): Based on URL, assess likely quality:\n"
        "   - HuggingFace datasets: 0.2-0.25 (well-structured platform)\n"
        "   - GitHub with clear dataset structure: 0.15-0.2\n"
        "   - Other platforms: 0.1-0.15\n"
        "4) Code Quality (0.25): Based on URL, assess likely quality:\n"
        "   - GitHub repos from known orgs (google, microsoft, etc.): 0.2-0.25\n"
        "   - HuggingFace model repos: 0.15-0.2\n"
        "   - Other GitHub repos: 0.1-0.15\n\n"
        "Final score = dataset_availability + code_availability + dataset_quality + code_quality\n"
        "Be conservative. Only give full points for clearly recognizable, high-quality platforms.\n\n"
        "Dataset URL Provided:\n" + (dataset_url if dataset_url else "None") + "\n"
        "Code URL Provided:\n" + (code_url if code_url else "None") + "\n"
    )

# Dataset quality metric calculation
def build_dataset_quality_prompt(dataset_url: str) -> str:
    """
        Build LLM prompt for analyzing dataset quality from URL.
    """
    return (
        "You evaluate DATASET QUALITY based on the provided dataset URL.\n"
        "Return ONE JSON object and nothing else (no prose/markdown/fences).\n"
        "JSON schema (exactly this):\n"
        "{\n"
        '  "dataset_quality_score": <float 0..1>,\n'
        '  "platform_reputation": <float 0..1>,\n'
        '  "likely_documentation_quality": <float 0..1>,\n'
        '  "likely_data_curation": <float 0..1>,\n'
        '  "accessibility": <float 0..1>,\n'
        '  "rationale": "<short sentence>"\n'
        "}\n\n"
        "How to compute dataset_quality_score (clamp to [0,1]):\n\n"
        "1) Platform Reputation (0.3):\n"
        "   - HuggingFace datasets: 0.25-0.3 (high curation standards)\n"
        "   - Academic institutions (.edu domains): 0.2-0.25\n"
        "   - Known research orgs (Google, Microsoft, etc.): 0.2-0.25\n"
        "   - GitHub from reputable sources: 0.15-0.2\n"
        "   - Other platforms: 0.0-0.15\n\n"
        "2) Likely Documentation Quality (0.3):\n"
        "   - HuggingFace (standardized cards): 0.25-0.3\n"
        "   - Academic papers/repos: 0.2-0.25\n"
        "   - Well-structured GitHub repos: 0.15-0.2\n"
        "   - Basic repos: 0.05-0.15\n"
        "   - Unknown/unclear: 0.0-0.05\n\n"
        "3) Likely Data Curation (0.25):\n"
        "   - Established benchmarks (GLUE, SQuAD, etc.): 0.2-0.25\n"
        "   - Academic datasets: 0.15-0.2\n"
        "   - Community datasets on HF: 0.1-0.15\n"
        "   - Personal/unknown datasets: 0.0-0.1\n\n"
        "4) Accessibility (0.15):\n"
        "   - Free, public access: 0.15\n"
        "   - Registration required: 0.1\n"
        "   - Unclear access: 0.05\n"
        "   - Likely restricted: 0.0\n\n"
        "Final score = platform_reputation + documentation_quality + data_curation + accessibility\n"
        "Be conservative. Only give high scores to clearly recognizable, high-quality datasets.\n\n"
        "Recognize common dataset names and patterns:\n"
        "- GLUE, SQuAD, ImageNet, COCO, LibriSpeech = high quality\n"
        "- HuggingFace URLs with clear dataset names = good quality\n"
        "- Academic paper datasets = moderate to good quality\n"
        "- Personal GitHub repos = lower quality unless clearly well-maintained\n\n"
        f"Dataset URL: {dataset_url}\n"
    )    

# Performance Claims metric calculation
def fetch_model_card_content(model_id: str) -> str:
    """
    Fetch model card content from HF Hub API
    """
    token = os.getenv("HF_TOKEN")
    try:
        model_card = ModelCard.load(model_id, token=token)
        txt = (getattr(model_card, "content", "") or "").strip()
        if txt.strip():
            return txt
    except RepositoryNotFoundError:
        logging.debug("Model repository not found: %s", model_id)
    except HfHubHTTPError as e:
        logging.debug("HfHubHTTPError for %s: %s", model_id, e)
    except Exception as e:
        logging.debug("Unexpected error fetching model card for %s: %s", model_id, e)
    return ""

def build_performance_prompt(model_card_excerpt: str) -> str:
    """
    Single prompt that tells the LLM to analyze the model card and also compute
    the final normalized score. It explicitly warns about empty headers.
    """
    if not model_card_excerpt.strip():
        model_card_excerpt = "(no model card content provided)"
    return (
        "You grade a model's 'Performance Claims' from its model card text.\n"
        "Return ONE JSON object and nothing else (no prose/markdown/fences).\n"
        "JSON schema (exactly this):\n"
        "{\n"
        '  "performance_score": <float 0..1>\n'
        "}\n\n"
        "How to compute performance_score (benchmark-first, conservative; clamp to [0,1]):\n\n"
        "1) Extract quantitative benchmark rows from the text:\n"
        "   - Capture: metric name, model_value, dataset (and split if given), baseline_value (and name) if present.\n"
        "   - Metric direction:\n"
        "     lower-better: WER, CER, PER, perplexity, loss, MAE, MSE, RMSE\n"
        "     higher-better: accuracy/acc, F1, precision, recall, BLEU, ROUGE, mAP, AUC, AP\n"
        "   - Normalize obvious percents (e.g., accuracy 85 → 0.85). Ignore malformed/unclear numbers.\n\n"
        "2) For each row with a valid baseline_value, compute relative improvement (clip to [-1,1]):\n"
        "   - higher-better:   rel = (model_value - baseline_value) / max(1e-9, abs(baseline_value))\n"
        "   - lower-better:    rel = (baseline_value - model_value) / max(1e-9, abs(baseline_value))\n"
        "   - If it's unclear the same dataset/split/protocol was used, halve rel.\n\n"
        "3) Aggregate signals:\n"
        "   - mean_rel = average rel over all valid rows (if none, use 0).\n"
        "   - evidence ∈ [0,1] (sum then cap at 1.0):\n"
        "       +0.30 if code/scripts are referenced\n"
        "       +0.30 if hyperparams/seeds/hardware are listed\n"
        "       +0.20 if external validation is peer-reviewed (or +0.10 if third-party)\n"
        "   - coverage ∈ [0,1]: let D = #unique datasets with numbers; coverage = min(1.0, sqrt(D)/3.0)\n\n"
        "4) Final score (clamp to [0,1]):\n"
        "   - If no numeric benchmarks found: performance_score ≤ 0.10.\n"
        "   - If numbers but no baselines:   performance_score ≤ 0.25.\n"
        "   - Else: performance_score = clamp01( 0.70*mean_rel + 0.20*evidence + 0.10*coverage )\n\n"
        "Rules:\n"
        "- Do not invent numbers. Be conservative when uncertain.\n"
        "- Penalize overstated claims that aren't supported by the numbers.\n\n"
        "MODEL CARD:\n<<<\n" + model_card_excerpt + "\n>>>\n"
    )

# Ramp-up metric calculation
def fetch_readme_content(model_id: str) -> str:
    """
    Fetch README content from HF Hub API
    """
    token = os.getenv("HF_TOKEN")
    # For a model repository
    try:
        path = hf_hub_download(repo_id=model_id, filename="README.md", token=token)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
            if txt.strip():
                return txt
    except EntryNotFoundError:
        logging.debug("No README.md found in model repo %s", model_id)
    except (RepositoryNotFoundError, HfHubHTTPError) as e:
        logging.debug("hf_hub_download failed for %s (%s): %s", model_id, "README", e)
    except Exception as e:
        logging.debug("hf_hub_download unexpected error: %s", e)
    return ""
    
def build_ramp_up_prompt(readme_excerpt: str) -> str:
    """
    Single prompt that tells the LLM to analyze the README and also compute
    the final normalized score. It explicitly warns about empty headers.
    """
    if not readme_excerpt.strip():
        readme_excerpt = "(no README content provided)"
    return (
        "You are a grader of developer 'Ramp-Up' time (time to first successful inference/use of model) based on README text.\n"
        "Return a single JSON object and nothing else (no prose nor markdown) with exactly these fields:\n"
        "{\n"
        '  "ramp_up_score": <float 0..1>,\n'
        '  "estimated_steps_to_run_once": <integer 0..20>,\n'
        '  "has_install_section": true|false,\n'
        '  "has_quickstart_section": true|false,\n'
        '  "has_minimal_inference_example": "yes"|"no"|"unclear",\n'
        '  "prerequisites_clearly_stated": true|false,\n'
        '  "missing_critical_info": ["none" OR any of "cuda","model_weights","env_vars","dataset_link"],\n'
        '  "clarity_0to1": <float 0..1>,\n'
        '  "completeness_0to1": <float 0..1>,\n'
        '  "confidence_0to1": <float 0..1>,\n'
        '  "rationale": "<one short sentence>"\n'
        "}\n"
        "Use the following scoring rubric (use this to compute ramp_up_score):\n"
        "- 0.20  Install section WITH actionable commands (pip/conda/git/docker/python). A header with no real content gets 0.\n"
        "- 0.20  Quickstart that runs a single inference. If only a header or vague text, give 0.\n"
        "- 0.15  Minimal inference example present (code or command). Map yes=1, unclear=0.4, no=0.\n"
        "- 0.15  Fewer steps is better: 3 steps → 1.0, 10+ steps → 0.0, linear in between.\n"
        "- 0.10  Prerequisites clearly stated (e.g., Python, CUDA version, weights, env vars).\n"
        "- 0.10  Clarity (0..1) and 0.10 Completeness (0..1).\n"
        "If unsure, choose conservative values. Penalize empty or placeholder sections.\n\n"
        "README:\n<<<\n" + readme_excerpt + "\n>>>\n")

# Size metric calculation
def get_model_weight_size(model_id: str) -> float:
    """Get total size of all files in MB via HF API."""
    # Use the tree API endpoint that includes file sizes
    url = f"https://huggingface.co/api/models/{model_id}/tree/main"
    
    # Get API key from environment
    api_key = os.getenv('HF_Key')
    headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        files_data = response.json()
        
        # The /tree/main endpoint returns an array of file objects
        if not isinstance(files_data, list):
            raise ValueError(f"Unexpected API response format: expected list, got {type(files_data)}")
        
        total_size_bytes = 0
        files_found = []
        
        for file_info in files_data:
            filename = file_info.get('path', '')
            file_size = file_info.get('size', 0)
            file_type = file_info.get('type', '')
            
            # Only process files and only if they have a size
            if file_type != 'file' or file_size == 0:
                continue
                
            # Add ALL files to the total
            total_size_bytes += file_size
            files_found.append({
                'filename': filename,
                'size_mb': file_size / (1000 * 1000) 
            })
        
        if not files_found:
            raise ValueError(f"No files found for model {model_id}")
        
        total_size_mb = total_size_bytes / (1000 * 1000)
        
        return round(total_size_mb, 2)
        
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to fetch model data from HF API: {str(e)}")
    except (KeyError, ValueError) as e:
        raise ValueError(f"Error parsing API response: {str(e)}")


def calculate_hardware_compatibility_scores(size_mb: float) -> dict:
    """Get scores for the different hardware types"""
    scores = {}
    
    # Raspberry Pi
    if size_mb <= 50:
        scores['raspberry_pi'] = 1.0
    elif size_mb <= 100:
        scores['raspberry_pi'] = 1.0 - ((size_mb - 50) / 50)
    else:
        scores['raspberry_pi'] = 0.0
    
    # Jetson Nano
    if size_mb <= 200:
        scores['jetson_nano'] = 1.0
    elif size_mb <= 500:
        scores['jetson_nano'] = 1.0 - ((size_mb - 200) / 300)
    else:
        scores['jetson_nano'] = 0.0
    
    # Desktop PC
    if size_mb <= 1000:
        scores['desktop_pc'] = 1.0
    elif size_mb <= 2000:
        scores['desktop_pc'] = 1.0 - ((size_mb - 1000) / 1000)
    else:
        scores['desktop_pc'] = 0.0
    
    # AWS Server
    if size_mb <= 5000:
        scores['aws_server'] = 1.0
    elif size_mb <= 10000:
        scores['aws_server'] = 1.0 - ((size_mb - 5000) / 5000)
    else:
        scores['aws_server'] = 0.0
    
    return scores
