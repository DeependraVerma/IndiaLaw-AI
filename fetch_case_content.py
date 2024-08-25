from bs4 import BeautifulSoup
import requests

def fetch_case_content(url):
    """Fetches and scrapes all <p> tag content from the <div class="judgments"> section at the provided URL."""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract content from the <div class="judgments"> section
        judgments_div = soup.find('div', {'class': 'judgments'})
        if judgments_div:
            # Extract all <p> tags within the judgments_div
            paragraphs = judgments_div.find_all('p')
            # Fetch all text from the paragraphs
            paragraphs_text = [p.get_text(separator="\n") for p in paragraphs]
            return paragraphs_text
        else:
            print("Judgments div not found!")
            return None
    else:
        print(f"Failed to retrieve page with status code: {response.status_code}")
        return None



# Example usage
#url = 'https://indiankanoon.org/doc/150051/'
#case_content = fetch_case_content(url)
#print(case_content)
