import pytest
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ece461.API.hf_api import extract_hf_model_id, extract_hf_dataset_id
from ece461.API.github_api import extract_github_id, extract_gitlab_id


class TestHuggingFaceIDExtraction:
    """Test HuggingFace ID extraction from various URL formats"""
    
    def test_extract_hf_model_basic(self):
        """Test basic model URL extraction"""
        url = "https://huggingface.co/microsoft/DialoGPT-medium"
        result = extract_hf_model_id(url)
        assert result == "microsoft/DialoGPT-medium"
    
    def test_extract_hf_model_with_tree(self):
        """Test model URL with /tree/main suffix"""
        url = "https://huggingface.co/microsoft/DialoGPT-medium/tree/main"
        result = extract_hf_model_id(url)
        assert result == "microsoft/DialoGPT-medium"
    
    def test_extract_hf_model_with_tree_branch(self):
        """Test model URL with different branch"""
        url = "https://huggingface.co/microsoft/DialoGPT-medium/tree/development"
        result = extract_hf_model_id(url)
        assert result == "microsoft/DialoGPT-medium"
    
    def test_extract_hf_dataset_basic(self):
        """Test basic dataset URL extraction"""
        url = "https://huggingface.co/datasets/squad"
        result = extract_hf_dataset_id(url)
        assert result == "squad"
    
    def test_extract_hf_dataset_with_owner(self):
        """Test dataset URL with owner/name format"""
        url = "https://huggingface.co/datasets/microsoft/ms_marco"
        result = extract_hf_dataset_id(url)
        assert result == "microsoft/ms_marco"
    
    def test_extract_hf_dataset_with_tree(self):
        """Test dataset URL with /tree/main suffix"""
        url = "https://huggingface.co/datasets/squad/tree/main"
        result = extract_hf_dataset_id(url)
        assert result == "squad"
    
    def test_extract_hf_url_with_whitespace(self):
        """Test URL with leading/trailing whitespace"""
        url = "  https://huggingface.co/microsoft/DialoGPT-medium  "
        result = extract_hf_model_id(url)
        assert result == "microsoft/DialoGPT-medium"
    
    def test_extract_hf_invalid_urls(self):
        """Test invalid URLs return None"""
        invalid_urls = [
            "",
            None,
            "https://github.com/microsoft/vscode",
            "https://google.com",
            "https://huggingface.co/",
            "https://huggingface.co/microsoft",
            "not_a_url"
        ]
        
        for url in invalid_urls:
            model_result = extract_hf_model_id(url)
            dataset_result = extract_hf_dataset_id(url)
            assert model_result is None, f"Expected None for model URL: {url}"
            assert dataset_result is None, f"Expected None for dataset URL: {url}"


class TestGitHubIDExtraction:
    """Test GitHub ID extraction from various URL formats"""
    
    def test_extract_github_repo_basic(self):
        """Test basic GitHub repo URL extraction"""
        url = "https://github.com/microsoft/vscode"
        result = extract_github_id(url)
        assert result == "microsoft/vscode"
    
    def test_extract_github_repo_with_git_suffix(self):
        """Test GitHub repo URL with .git suffix"""
        url = "https://github.com/microsoft/vscode.git"
        result = extract_github_id(url)
        assert result == "microsoft/vscode"
    
    def test_extract_github_repo_with_path(self):
        """Test GitHub repo URL with additional path"""
        url = "https://github.com/microsoft/vscode/blob/main/README.md"
        result = extract_github_id(url)
        assert result == "microsoft/vscode"
    
    def test_extract_github_url_with_whitespace(self):
        """Test URL with leading/trailing whitespace"""
        url = "  https://github.com/tensorflow/tensorflow  "
        result = extract_github_id(url)
        assert result == "tensorflow/tensorflow"

    def test_extract_gitlab_repo_basic(self):
        """Test basic GitLab repo URL extraction"""
        url = "https://gitlab.com/gitlab-org/gitlab"
        result = extract_gitlab_id(url)
        assert result == "gitlab-org/gitlab"
    
    def test_extract_gitlab_repo_with_git_suffix(self):
        """Test GitLab repo URL with .git suffix"""
        url = "https://gitlab.com/gitlab-org/gitlab.git"
        result = extract_gitlab_id(url)
        assert result == "gitlab-org/gitlab"
    
    def test_extract_github_invalid_urls(self):
        """Test invalid URLs return None"""
        invalid_urls = [
            "",
            None,
            "https://huggingface.co/microsoft/model",
            "https://google.com",
            "https://github.com/",
            "https://github.com/microsoft",
            "not_a_url"
        ]
        
        for url in invalid_urls:
            result = extract_github_id(url)
            assert result is None, f"Expected None for URL: {url}"


if __name__ == "__main__":
    pytest.main([__file__])
