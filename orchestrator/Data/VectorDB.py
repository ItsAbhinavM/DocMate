import json
import requests
from Secrets.keys import vector_store_url

url = vector_store_url


def retrieve_contact(query):
    data = {
        "query": query,
    }

    # Send the POST request
    response = requests.post(url, json=data)

    # Check the response
    if response.status_code == 200:
        print(response.text)
    else:
        print(f"POST request failed with status code {response.status_code}")
        print("Error message:")
        print(response.text)

    return json.loads(response.text)
