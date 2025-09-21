# from huggingface_hub import ModelCard
#
# card = ModelCard.load('openai/whisper-tiny')
#
# model_data = card.data.to_dict()
#
# model_text = card.text
#
# print(card.content)
import math
from typing import List, Dict, Optional
from huggingface_hub import model_info
from huggingface_hub.hf_api import ModelInfo

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
    known_orgs: List[str] = ["google", "meta", "microsoft", "openai", "bigscience", "stabilityai", "runwayml"]
    if author in known_orgs:
        return 1.0
    # Add other checks here if needed, e.g., for GTE-validated users
    return 0.2 # Default for individual users

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
    return min(1.0, math.log10(downloads + 1) / 6)

def calculate_bus_factor(repo_id: str) -> float:
    """
    Calculates the Model Resilience Score for a given Hugging Face model repository.
    """
    info: ModelInfo = model_info(repo_id, files_metadata=True)
    
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
                   
    return final_score

print(calculate_bus_factor("openai/whisper-tiny"))

