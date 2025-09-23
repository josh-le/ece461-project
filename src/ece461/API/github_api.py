import requests
import os
import logging

def process_github_links(links):
    """Process GitHub API with parsed URLs"""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        logging.error("GITHUB_TOKEN environment variable not set")
        return
    
    headers = {"Authorization": f"token {token}"}
    
    for link in links:
        if link.code and 'github.com' in link.code:
            # Extract repo name from GitHub URL
            repo_url = link.code.strip()
            # Remove .git suffix if present
            if repo_url.endswith('.git'):
                repo_url = repo_url[:-4]
            repo_parts = repo_url.split('/')
            if len(repo_parts) >= 5:
                repo_name = f"{repo_parts[-2]}/{repo_parts[-1]}"
                try:
                    response = requests.get(f"https://api.github.com/repos/{repo_name}", headers=headers)
                    if response.status_code == 200:
                        repo_data = response.json()
                        logging.info(f"GitHub API processed repo: {repo_name}")
                        logging.debug(f"Repo {repo_name}: {repo_data.get('stargazers_count', 0)} stars")
                    else:
                        logging.error(f"GitHub API failed for {repo_name}: status {response.status_code}")
                except Exception as e:
                    logging.error(f"GitHub API request failed for {repo_name}: {e}")
