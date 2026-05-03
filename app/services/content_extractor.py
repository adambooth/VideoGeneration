from __future__ import annotations

from dataclasses import dataclass

import requests
import trafilatura
from bs4 import BeautifulSoup


@dataclass
class ExtractedContent:
    combined_text: str
    summary: str
    sources: list[dict]


class ContentExtractor:
    def extract(self, urls: list[str]) -> ExtractedContent:
        sources: list[dict] = []
        chunks: list[str] = []

        for url in urls:
            text = self._extract_single(url)
            sources.append({"url": url, "text": text})
            chunks.append(f"Source: {url}\n{text}")

        combined = "\n\n".join(chunks).strip()
        summary_lines = [f"- {item['url']}" for item in sources]
        summary = "Imported sources:\n" + "\n".join(summary_lines)
        return ExtractedContent(combined_text=combined, summary=summary, sources=sources)

    def _extract_single(self, url: str) -> str:
        if "reddit.com" in url.rstrip("/"):
            reddit_url = url.rstrip("/") + ".json"
            response = requests.get(
                reddit_url,
                timeout=20,
                headers={"User-Agent": "AutomatedVideoCreator/1.0"},
            )
            response.raise_for_status()
            payload = response.json()
            post = payload[0]["data"]["children"][0]["data"]
            text_bits = [post.get("title", ""), post.get("selftext", "")]
            return "\n".join(bit for bit in text_bits if bit).strip()

        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            if extracted:
                return extracted.strip()

        response = requests.get(url, timeout=20, headers={"User-Agent": "AutomatedVideoCreator/1.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join(part.strip() for part in soup.stripped_strings)
        if not text:
            raise ValueError(f"Could not extract readable text from {url}")
        return text
