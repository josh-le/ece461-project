import requests
import os
import logging

def extract_github_id(url):
    """Extract repository ID from GitHub URL"""
    if not url or 'github.com' not in url:
        return None
    
    clean_url = url.strip()
    if clean_url.endswith('.git'):
        clean_url = clean_url[:-4]
    
    parts = clean_url.split('/')
    if len(parts) >= 5:
        # Always use parts 3 and 4 (owner and repo) regardless of extra path
        return f"{parts[3]}/{parts[4]}"
    
    return None
