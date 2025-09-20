import requests
import os
import logging

def process_hf_links(links):
    """Process Hugging Face API with parsed URLs"""
    token = os.getenv("HF_TOKEN")
    if not token:
        logging.error("HF_TOKEN environment variable not set")
        return
    
    headers = {"Authorization": f"Bearer {token}"}
    
    for link in links:
        # Process model URLs
        if link.model and 'huggingface.co' in link.model:
            model_url = link.model.strip()
            # Remove /tree/main or other suffixes
            if '/tree/' in model_url:
                model_url = model_url.split('/tree/')[0]
            model_parts = model_url.split('/')
            if len(model_parts) >= 5:
                model_name = f"{model_parts[-2]}/{model_parts[-1]}"
                try:
                    response = requests.get(f"https://huggingface.co/api/models/{model_name}", headers=headers)
                    if response.status_code == 200:
                        model_data = response.json()
                        logging.info(f"HF API processed model: {model_name}")
                        logging.debug(f"Model {model_name}: {model_data.get('downloads', 0):,} downloads")
                    else:
                        logging.error(f"HF API failed for {model_name}: status {response.status_code}")
                except Exception as e:
                    logging.error(f"HF API request failed for {model_name}: {e}")
        
        # Process dataset URLs
        if link.dataset and 'huggingface.co/datasets' in link.dataset:
            dataset_url = link.dataset.strip()
            # Remove /tree/main or other suffixes
            if '/tree/' in dataset_url:
                dataset_url = dataset_url.split('/tree/')[0]
            dataset_parts = dataset_url.split('/')
            if len(dataset_parts) >= 6:
                dataset_name = f"{dataset_parts[-2]}/{dataset_parts[-1]}"
                try:
                    response = requests.get(f"https://huggingface.co/api/datasets/{dataset_name}", headers=headers)
                    if response.status_code == 200:
                        dataset_data = response.json()
                        logging.info(f"HF API processed dataset: {dataset_name}")
                        logging.debug(f"Dataset {dataset_name}: {dataset_data.get('downloads', 0):,} downloads")
                    else:
                        logging.error(f"HF dataset API failed for {dataset_name}: status {response.status_code}")
                except Exception as e:
                    logging.error(f"HF dataset API request failed for {dataset_name}: {e}")
