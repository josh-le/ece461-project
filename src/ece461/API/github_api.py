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
<<<<<<< HEAD

def extract_gitlab_id(url):
    """Extract repository ID from GitLab URL"""
    if not url or 'gitlab.com' not in url:
        return None
    
    clean_url = url.strip()
    if clean_url.endswith('.git'):
        clean_url = clean_url[:-4]
    
    parts = clean_url.split('/')
    if len(parts) >= 5:
        return f"{parts[3]}/{parts[4]}"
    
    # if parts[3] is a user, then we need to use parts[4] and parts[5]
    if parts[3] == 'users':
        return f"{parts[4]}/{parts[5]}"
    
    return None
=======
>>>>>>> 67f3a39c8de136351be9fa4cd156d2079c5ab961
