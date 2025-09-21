import os, requests, logging, subprocess, re, sys, json
from ece461.API import llm_api
import time
from huggingface_hub import hf_hub_download, ModelCard
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError

class Metrics():
    @staticmethod
    def calculate_all_metrics():
        """
            Run all metric calculations and return their results.
        """
        pass
    
    @staticmethod
    def calculate_ramp_up_metric(model_id: str) -> tuple[float, float]:
        """
            Calculate ramp-up time metric.
        """
        # Start latency calculation
        start_time = time.perf_counter()

        # Initiate Ramp-up metric calculation
        readme_data = fetch_readme_content(model_id)
        if readme_data == "":
            logging.info("No README content found for model %s", model_id)
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
                logging.error("Unexpected LLM response format for model %s: %s", model_id, response)
        
        logging.info("Ramp-up metric LLM score for model %f", score)
        # End latency calculation
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        logging.info("Ramp-up metric latency for model %s: %.2f ms", model_id, latency)
        
        return score, latency
    
    @staticmethod
    def calculate_license_metric(model_id: str) -> tuple[float, float]:
        """
            Calculate license compatibility score.
        """
        # Start latency calculation
        start_time = time.perf_counter()

        # Initiate license metric calculation
        model_card_data = fetch_model_card_content(model_id)
        if model_card_data == "":
            logging.info("No model card content found for model %s", model_id)
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
                logging.error("Unexpected LLM response format for model %s: %s", model_id, response)
        
        logging.info("License metric LLM score for model %f", score)
        # End latency calculation
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        logging.info("License metric latency for model %s: %.2f ms", model_id, latency)
        
        return score, latency

    @staticmethod
    def calculate_performance_metric(model_id: str) -> tuple[float, float]:
        """
            Calculate performance benchmark score.
        """
        # Start latency calculation
        start_time = time.perf_counter()

        # Initiate performance metric calculation
        model_card_data = fetch_model_card_content(model_id)
        if model_card_data == "":
            logging.info("No model card content found for model %s", model_id)
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
                logging.error("Unexpected LLM response format for model %s: %s", model_id, response)
        
        logging.info("Performance metric LLM score for model %f", score)
        # End latency calculation
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        logging.info("Performance metric latency for model %s: %.2f ms", model_id, latency)

        return score, latency

    @staticmethod
    def calculate_size_metric(model_id: str) -> dict:
        """
            Calculate size compatibility scores for different hardware types.
        """
        try:
            total_size_mb = get_model_weight_size(model_id)
            return calculate_hardware_compatibility_scores(total_size_mb)
        except Exception as e:
            raise ValueError(f"Failed to calculate size metric for {model_id}: {str(e)}")
    
    @staticmethod
    def score_code_quality(path: str) -> dict[str, object]:
        """
            Run pylint on Python files in 'path' and return normalized score.
        """
        proc = subprocess.run(
            [sys.executable, "-m", "pylint", "--exit-zero", "--score=y", path],
            capture_output=True, text=True
        )
        m = re.search(r"rated at\s+([0-9.]+)/10", proc.stdout)
        score10 = float(m.group(1)) if m else 0.0
        return {
            "name": "code_quality",
            "score": max(0.0, min(1.0, score10/10)),
            "raw": {"pylint_score": score10}
        } 

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
            
            # Only process files (not directories), and only if they have a size
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
        
        return total_size_mb
        
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