import pytest
import sys
import os
import requests
sys.path.append('src')

from ece461.url_file_parser import parse_url_file
from ece461.API.hf_api import process_hf_links

def test_hf_with_parsed_urls():
    """Test hf_api.py with actual parsed URLs from sample file"""
    token = os.getenv("HF_TOKEN")
    if not token:
        pytest.skip("HF_TOKEN not set")

    # Test API connection first
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get("https://huggingface.co/api/models/bert-base-uncased", headers=headers)
    assert response.status_code == 200
    
    # Use your actual URL parser
    links = parse_url_file("samples/sample-input.txt")
    print(f"Parsed {len(links)} links from sample file")
    
    # Get some basic metrics for HF models
    headers = {"Authorization": f"Bearer {token}"}
    
    for i, link in enumerate(links, 1):
        if link.model and 'huggingface.co' in link.model:
            model_path = link.model.strip().split('/')[-2:]
            model_name = '/'.join(model_path) if len(model_path) == 2 else None
            
            if model_name:
                try:
                    response = requests.get(f"https://huggingface.co/api/models/{model_name}", headers=headers)
                    if response.status_code == 200:
                        model = response.json()
                        print(f"HF Model {i}: {model_name}")
                        print(f"  Downloads: {model.get('downloads', 0):,}")
                        print(f"  Likes: {model.get('likes', 0)}")
                        print(f"  License: {model.get('cardData', {}).get('license', 'unknown')}")
                except:
                    pass
    
    # Test with real parsed data
    process_hf_links(links)
    print("Completed HF processing with real URLs")
