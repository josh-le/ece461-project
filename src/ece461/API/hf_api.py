import requests
import os
import logging

def extract_hf_model_id(url):
    """Extract model ID from HuggingFace model URL"""
    if not url or 'huggingface.co' not in url or 'datasets' in url:
        return None
    
    clean_url = url.strip()
    # Remove /tree/main or other suffixes
    if '/tree/' in clean_url:
        clean_url = clean_url.split('/tree/')[0]
    
    parts = clean_url.split('/')
    if len(parts) >= 5:
        return f"{parts[-2]}/{parts[-1]}"
    
    return None

def extract_hf_dataset_id(url):
    """Extract dataset ID from HuggingFace dataset URL"""
    if not url or 'huggingface.co/datasets' not in url:
        return None
    
    clean_url = url.strip()
    # Remove /tree/main or other suffixes
    if '/tree/' in clean_url:
        clean_url = clean_url.split('/tree/')[0]
    
    parts = clean_url.split('/')
    
    # For datasets: https://huggingface.co/datasets/name OR https://huggingface.co/datasets/owner/name
    if len(parts) >= 6:  # owner/name format
        return f"{parts[-2]}/{parts[-1]}"
    elif len(parts) >= 5:  # just name format
        return parts[-1]
    
    return None
