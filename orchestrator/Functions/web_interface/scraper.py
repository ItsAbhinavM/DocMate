import json
from collections import defaultdict

from playwright.sync_api import sync_playwright


def scrape_website(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)

        data = defaultdict(list)

        role_mappings = {
            "paragraph": "paragraphs",
            "listitem": "list_items",
            "cell": "table_elements",
            "heading": "headings",
        }

        for role, key in role_mappings.items():
            for element in page.get_by_role(role).all():
                content = element.inner_text()
                if content:
                    data[key].append(content)

        # Used by code blocks in medium
        for pre in page.query_selector_all("pre"):
            data["preformatted"].append(pre.inner_text())

        for link in page.get_by_role("link").all():
            url = link.get_attribute("href")
            if url and "http" in url:
                data["links"].append(url)
        print(json.dumps(data))

        browser.close()


if __name__ == "__main__":
    scrape_website(
        "https://medium.com/the-ai-forum/implementing-advanced-rag-in-langchain-using-raptor-258a51c503c6"
    )
