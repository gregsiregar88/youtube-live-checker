import os
import re
import json
import asyncio
import logging
import contextlib
from functools import lru_cache
from typing import Dict, List, Optional

import aiohttp
from aiohttp import TCPConnector
from bs4 import BeautifulSoup
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

# ---------------- Config ---------------- #

class Config:
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv("PORT", "8000"))
    CACHE_TTL: int = 10
    ENABLE_CACHING: bool = True
    USER_AGENT: str = (
        "Mozilla/5.0 (X11; Linux x86_64; rv:142.0) Gecko/20100101 Firefox/142.0"
    )

config = Config()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHANNELS_FILE: str = os.path.join(BASE_DIR, "channels_with_id.json")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("yt-live")

# ---------------- HTML Helpers ---------------- #

class HTMLParser:
    @staticmethod
    def extract_canonical_url(html: str) -> Optional[str]:
        soup = BeautifulSoup(html, "html.parser")
        link = soup.find("link", {"rel": "canonical"})
        return link["href"] if link else None

    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|&|$)", url)
        return match.group(1) if match else None

    @staticmethod
    def is_live_stream(html: str) -> bool:
        return '"isLive":true' in html

# ---------------- Cache ---------------- #

class CacheManager:
    def __init__(self, ttl: int):
        self.ttl = ttl
        self.cache: Dict[str, tuple] = {}

    def get(self, key: str):
        if key in self.cache:
            data, expiry = self.cache[key]
            if asyncio.get_event_loop().time() < expiry:
                return data
            self.cache.pop(key, None)
        return None

    def set(self, key: str, value):
        expiry = asyncio.get_event_loop().time() + self.ttl
        self.cache[key] = (value, expiry)

cache_manager = CacheManager(config.CACHE_TTL)

# ---------------- YouTube API Client ---------------- #

class YouTubeAPIClient:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.api_key = os.getenv("YOUTUBE_API_KEY")

    def _extract_video_id(self, url: str) -> Optional[str]:
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|&|$)", url)
        return match.group(1) if match else None

    async def fetch_status(self, urls: List[str]) -> Dict[str, Dict]:
        video_ids = [self._extract_video_id(url) for url in urls]
        video_ids = [vid for vid in video_ids if vid]

        if not self.api_key or not video_ids:
            return {}

        results = {}
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i : i + 50]
            data = await self._fetch_batch(batch)
            results.update(data)
        return results

    async def _fetch_batch(self, video_ids: List[str]) -> Dict[str, Dict]:
        url = "https://www.googleapis.com/youtube/v3/videos"
        params = {
            "part": "liveStreamingDetails,snippet",
            "id": ",".join(video_ids),
            "key": self.api_key,
        }
        try:
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                return self._parse_response(data)
        except Exception as e:
            logger.error(f"API fetch error: {e}")
            return {}

    def _parse_response(self, data: Dict) -> Dict[str, Dict]:
        result = {}
        for item in data.get("items", []):
            video_id = item["id"]
            live_details = item.get("liveStreamingDetails", {})
            snippet = item.get("snippet", {})
            if live_details.get("actualStartTime") and not live_details.get("actualEndTime"):
                status = "LIVE"
            elif live_details.get("scheduledStartTime") and not live_details.get("actualStartTime"):
                status = "UPCOMING"
            else:
                status = "OFFLINE"
            result[video_id] = {
                "status": status,
                "title": snippet.get("title", ""),
                "channel": snippet.get("channelTitle", ""),
            }
        return result

# ---------------- FastAPI ---------------- #

app = FastAPI()
connector = TCPConnector(limit=100)
session: Optional[aiohttp.ClientSession] = None
yt_client: Optional[YouTubeAPIClient] = None
_warm_task: Optional[asyncio.Task] = None

async def fetch_html(url: str) -> Optional[str]:
    headers = {"User-Agent": config.USER_AGENT, "Accept": "text/html"}
    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                return await response.text()
            logger.warning(f"Fetch failed for {url}: {response.status}")
    except Exception as e:
        logger.error(f"Fetch error {url}: {e}")
    return None

async def _analyze_html_content(html: str, url: str, api_status: Optional[Dict]) -> Dict:
    canonical = HTMLParser.extract_canonical_url(html)
    if canonical:
        video_id = HTMLParser.extract_video_id(canonical)
        if video_id and api_status and video_id in api_status:
            return {"url": url, "videoId": video_id, **api_status[video_id]}
        elif video_id:
            if HTMLParser.is_live_stream(html):
                return {"url": url, "videoId": video_id, "status": "LIVE"}
            else:
                return {"url": url, "videoId": video_id, "status": "OFFLINE"}
    return {"url": url, "status": "OFFLINE"}

@app.get("/check")
async def check_status(urls: List[str] = Query(..., description="YouTube URLs to check")):
    cache_key = ",".join(sorted(urls))
    cached = cache_manager.get(cache_key)
    if cached:
        return JSONResponse(content=cached)

    api_results = await yt_client.fetch_status(urls) if yt_client else {}
    results = []
    for url in urls:
        html = await fetch_html(url)
        if html:
            analysis = await _analyze_html_content(html, url, api_results)
        else:
            analysis = {"url": url, "status": "ERROR"}
        results.append(analysis)

    cache_manager.set(cache_key, results)
    return JSONResponse(content=results)

# ---------------- Lifecycle ---------------- #

@app.on_event("startup")
async def startup_event():
    global session, yt_client, _warm_task
    session = aiohttp.ClientSession(connector=connector)
    yt_client = YouTubeAPIClient(session)
    if config.ENABLE_CACHING:
        _warm_task = asyncio.create_task(warm_cache_periodically())

@app.on_event("shutdown")
async def shutdown_event():
    global _warm_task
    if _warm_task:
        _warm_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _warm_task
    if session:
        await session.close()

# ---------------- Cache warmer ---------------- #

async def warm_cache_periodically():
    while True:
        try:
            if os.path.exists(CHANNELS_FILE):
                with open(CHANNELS_FILE) as f:
                    data = json.load(f)
                urls = [ch["url"] for ch in data.values() if "url" in ch]
                if urls:
                    await check_status(urls=urls)
        except Exception as e:
            logger.error(f"Cache warm error: {e}")
        await asyncio.sleep(300)

# ---------------- Main ---------------- #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT, workers=1, log_level="warning")
