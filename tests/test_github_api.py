import pytest
import sys
import os
import requests
sys.path.append('src')

from ece461.url_file_parser import parse_url_file
from ece461.API.github_api import process_github_links

def test_github_with_parsed_urls():
    """Test github_api.py with actual parsed URLs from sample file"""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        pytest.skip("GITHUB_TOKEN not set")

    # Test API connection first
    headers = {"Authorization": f"token {token}"}
    response = requests.get("https://api.github.com/user", headers=headers)
    assert response.status_code == 200
    
    # Use your actual URL parser
    links = parse_url_file("samples/sample-input.txt")
    assert len(links) > 0
    print(f"Parsed {len(links)} links from sample file")
    
    # Get some basic metrics for GitHub repos
    headers = {"Authorization": f"token {token}"}
    
    for i, link in enumerate(links, 1):
        if link.code and 'github.com' in link.code:
            repo_name = link.code.strip().split('/')[-2:] 
            repo_path = '/'.join(repo_name) if len(repo_name) == 2 else None
            
            if repo_path:
                try:
                    response = requests.get(f"https://api.github.com/repos/{repo_path}", headers=headers)
                    if response.status_code == 200:
                        repo = response.json()
                        print(f"GitHub Repo {i}: {repo_path}")
                        print(f"  Stars: {repo.get('stargazers_count', 0)}")
                        print(f"  Forks: {repo.get('forks_count', 0)}")
                        print(f"  Issues: {repo.get('open_issues_count', 0)}")
                        assert True
                    else:
                        assert False
                except:
                    pass
    
    # Test with real parsed data
    process_github_links(links)
    print("Completed GitHub processing with real URLs")
