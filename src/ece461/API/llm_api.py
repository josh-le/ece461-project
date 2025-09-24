import requests
import json
import logging
import os

GENAI_API_URL = "https://genai.rcac.purdue.edu/api/chat/completions"

def query_llm(prompt: str) -> str:
    """
    Queries the LLM API with the given message and returns the response.
    
    Args:
        prompt (str): The message to send to the LLM API.
    
    Returns:
        str: The content message from the LLM API response.
    """

    # Retrieve the API key from environment variables
    jwt_token_or_api_key = os.getenv("GEN_AI_STUDIO_API_KEY")
    if not jwt_token_or_api_key:
        logging.error("GENAI_API_KEY environment variable not set.")
        raise ValueError("GENAI_API_KEY environment variable not set.")

    headers = {
        "Authorization": f"Bearer {jwt_token_or_api_key}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "gpt-oss:120b",
        "temperature": 0.0,
        "messages": [
        {
            "role": "user",
            "content": prompt
        }
        ],
        "stream": False
    }
    response = requests.post(GENAI_API_URL, headers=headers, json=body)
    if response.status_code == 200:
        logging.info("LLM API response received successfully")
        data = response.json()
        logging.debug(f"LLM API response data: {json.dumps(data, indent=2)}")
        return data["choices"][0]["message"]["content"]
    else:
        logging.error(f"LLM API request failed with status code {response.status_code}: {response.text}")
        raise Exception(f"Error: {response.status_code}, {response.text}")