import json
import os
from datetime import datetime
import requests

def web_search(query, date, top_k=3):
    """
    Fetches search results using the Serper API and extracts URLs and snippets.

    Parameters:
    query (str): The search query.
    date (datetime.date): The date to filter results up to.
    top_k (int): The number of search results to fetch.

    Returns:
    tuple: A tuple containing two lists:
           - List of URLs from the organic results.
           - List of snippets from the organic results (empty string if snippet is missing).
    """
    # Format the date to the required format
    end_date = datetime.strptime(date, "%d-%m-%Y").strftime('%d/%m/%Y')

    # Serper API endpoint
    url = "https://google.serper.dev/search"

    # Prepare the payload
    payload = json.dumps({
        "q": query,
        "num": top_k,
        "tbs": f"cdr:1,cd_min:1/1/1900,cd_max:{end_date}"
    })

    # Read API key from environment (recommended: put it in a .env file as SERPER_API_KEY)
    api_key = os.getenv('SERPER_API_KEY')
    if not api_key:
        raise RuntimeError(
            "Missing SERPER_API_KEY. Set it as an environment variable or in a .env file."
        )

    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    # Make the POST request to the Serper API
    response = requests.request("POST", url, headers=headers, data=payload)

    # Check if the request was successful
    if response.status_code == 200:
        results = response.json()
        
        # Extract URLs and snippets from the organic results
        urls = []
        snippets = []
        
        for item in results.get("organic", []):
            if len(urls) >= top_k:
                break
            url = item.get("link", "")
            if url.endswith("pdf"):
                continue
            urls.append(url)  # Get URL, default to empty string if missing
            snippets.append(item.get("snippet", ""))  # Get snippet, default to empty string if missing
        return urls, snippets
    else:
        raise Exception(f"Failed to fetch search results. Status code: {response.status_code}, Response: {response.text}")