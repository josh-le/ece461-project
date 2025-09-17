import os
import requests
from typing import Optional, List, Dict, Any


def calculate_size_metric(model_id: str) -> dict:
    """Calculate size compatibility scores for different hardware types."""
    try:
        total_size_mb = get_model_weight_size(model_id)
        return calculate_hardware_compatibility_scores(total_size_mb)
    except Exception as e:
        raise ValueError(f"Failed to calculate size metric for {model_id}: {str(e)}")


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
                'size_bytes': file_size,
                'size_mb_binary': file_size / (1024 * 1024),  # Binary MB
                'size_mb_decimal': file_size / (1000 * 1000)  # Decimal MB
            })
        
        if not files_found:
            raise ValueError(f"No files found for model {model_id}")
        
        # Calculate totals in both formats
        total_mb_binary = total_size_bytes / (1024 * 1024)
        total_mb_decimal = total_size_bytes / (1000 * 1000)
        
        # Debug output - show both calculations
        print(f"Model {model_id}: Found {len(files_found)} total files")
        print(f"  Total size: {total_mb_binary:.2f} MB (binary) | {total_mb_decimal:.2f} MB (decimal)")
        
        # Show largest files in both formats for comparison
        largest_files = sorted(files_found, key=lambda x: x['size_bytes'], reverse=True)[:3]
        print("  Largest files:")
        for f in largest_files:
            print(f"    - {f['filename']}: {f['size_mb_binary']:.2f} MB (binary) | {f['size_mb_decimal']:.2f} MB (decimal)")
        
        # Return decimal for now - we can switch based on which matches better
        return total_mb_decimal
        
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to fetch model data from HF API: {str(e)}")
    except (KeyError, ValueError) as e:
        raise ValueError(f"Error parsing API response: {str(e)}")


def calculate_hardware_compatibility_scores(size_mb: float) -> dict:
    """Get scores for the different hardware types"""
    scores = {}
    
    # Raspberry Pi - very constrained
    if size_mb <= 50:
        scores['raspberry_pi'] = 1.0
    elif size_mb <= 100:
        scores['raspberry_pi'] = 1.0 - ((size_mb - 50) / 50)
    else:
        scores['raspberry_pi'] = 0.0
    
    # Jetson Nano - moderately constrained
    if size_mb <= 200:
        scores['jetson_nano'] = 1.0
    elif size_mb <= 500:
        scores['jetson_nano'] = 1.0 - ((size_mb - 200) / 300)
    else:
        scores['jetson_nano'] = 0.0
    
    # Desktop PC - less constrained
    if size_mb <= 1000:
        scores['desktop_pc'] = 1.0
    elif size_mb <= 2000:
        scores['desktop_pc'] = 1.0 - ((size_mb - 1000) / 1000)
    else:
        scores['desktop_pc'] = 0.0
    
    # AWS Server - minimal constraints
    if size_mb <= 5000:
        scores['aws_server'] = 1.0
    elif size_mb <= 10000:
        scores['aws_server'] = 1.0 - ((size_mb - 5000) / 5000)
    else:
        scores['aws_server'] = 0.0
    
    return scores


# Testing
def test_size_metric() -> None:
    """Test with real HF models."""
    test_models = [
        "distilbert-base-uncased",  # Medium model
        "bert-base-uncased",        # Large model 
        "microsoft/DialoGPT-small"  # Medium model
    ]
    
    for model_id in test_models:
        try:
            scores = calculate_size_metric(model_id)
            print(f"{model_id}:")
            for hardware, score in scores.items():
                print(f"  {hardware}: {score:.3f}")
            print()
        except Exception as e:
            print(f"{model_id}: Error = {str(e)}")
            print()


if __name__ == "__main__":
    test_size_metric()
