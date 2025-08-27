# MIT License
# Copyright (c) 2025 Greg Siregar

import re
import json
import os
import asyncio
import time
import sys
from dotenv import load_dotenv
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from typing import Dict, List, Set, Optional
from urllib.parse import urlparse, parse_qs, urlunparse
from playwright.async_api import async_playwright

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:142.0) Gecko/20100101 Firefox/142.0"
}

api_key = os.getenv("API_KEY")
if not api_key:
    logger.error("API_KEY environment variable is not set")
    raise ValueError("API_KEY environment variable is not set")

app = FastAPI(title="YouTube Live Checker API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add a simple cache mechanism
class SimpleCache:
    def __init__(self, ttl_seconds: int = 30):
        self.cache: Dict[str, dict] = {}
        self.ttl = ttl_seconds
        self.timestamps: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[dict]:
        if key in self.cache and time.time() - self.timestamps[key] < self.ttl:
            return self.cache[key]
        return None
    
    def set(self, key: str, value: dict):
        self.cache[key] = value
        self.timestamps[key] = time.time()

# Global cache instance
cache = SimpleCache(ttl_seconds=30)

class YTLiveChecker:
    def __init__(self, channels_file=None):
        default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'channels_with_id.json')
    
        self.channels_file = channels_file or os.getenv("CHANNELS_FILE", default_path)
        logger.info(f"Using channels file: {self.channels_file}")
        self.channels = self.load_channels()
        self.scheduled_pattern = re.compile(r'(Scheduled for|Live in|Premieres|Waiting for)', re.IGNORECASE)
        self.WAITING_ROOM_URLS = {
            "w4dgql_5Rzk", "O9V_EFbgpKQ", "rKMhl43RHo0", "MDwkJVqui_M",
            "INFI9FahPY0", "TLw3Taw5jxI", "-tMd6H-IxcA", "wUEN1KE2ZcU",
            "xbJ8tbA_Phw", "VKzTYEBsImc", "XeKiT4cLT6U", "hsAr4h_Mljw",
            "Fl1vM3scybw", "lGr_kZmjskI", "JAUSrqX0hW8", "38fJIy2FoDg",
            "1WhsM61BUfk", "8deE3F_WgBA", "c7K6RInG3Dw", "6GZ5XGzRY-g",
            "UkqwIcO3YN8", "hlDFczhR2mo", "Criw5zhE0bI", "u8aMX32hlgQ",
            "2JciZo2afXg", "GZHhb_zHVno", "SCVbMM71viE", "M9H0gTvKmGU",
            "Pd9VYQtr2c4", "wnjd9XuuXg0", "IlYErJ1ry_8", "VoWHIX4tp5k",
            "9vaxfw1qFcY", "sDZFWow3IKI", "XfTmGxpgN0g", "-35UzQOnzRE"
        }

    def load_channels(self):
        if not os.path.exists(self.channels_file):
            logger.error(f"Channels file not found at {self.channels_file}")
            return []
        try:
            with open(self.channels_file, 'r') as f:
                content = f.read()
                if not content.strip():
                    logger.error(f"Channels file {self.channels_file} is empty")
                    return []
                channels = json.loads(content)
                if not isinstance(channels, list):
                    logger.error(f"channels_with_id.json must contain a list, got: {type(channels)}, content: {content}")
                    return []
                for channel in channels:
                    if not isinstance(channel, dict) or 'handle' not in channel or 'id' not in channel:
                        logger.error(f"Invalid channel format in {self.channels_file}: {channel}, full content: {content}")
                        return []
                # Remove duplicate channels
                unique_channels = {tuple(channel.items()): channel for channel in channels}.values()
                logger.info(f"Loaded {len(unique_channels)} channels")
                return list(unique_channels)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in channels file {self.channels_file}: {str(e)}, content: {content}")
            return []
        except Exception as e:
            logger.error(f"Error loading channels file {self.channels_file}: {str(e)}, content: {content}")
            return []

    async def check_all_channels(self):
        if not self.channels:
            logger.warning("No channels to check")
            return []
        
        async with async_playwright() as p:
            # Launch browser with specific options to avoid detection
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-accelerated-2d-canvas',
                    '--no-first-run',
                    '--no-zygote',
                    '--disable-gpu',
                    '--window-size=1920,1080'
                ]
            )
            
            # Create a new context with specific settings
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                viewport={'width': 1920, 'height': 1080},
                java_script_enabled=True
            )
            
            # Block images and videos to improve performance
            await context.route("**/*", lambda route: route.abort() 
                if route.request.resource_type in ["image", "media", "font"] 
                else route.continue_()
            )
            
            try:
                tasks = [
                    self.check_channel(context, f"https://www.youtube.com/@{channel['handle']}/live", 
                                      channel['handle'], channel['id'])
                    for channel in self.channels
                ]
                return await asyncio.gather(*tasks)
            finally:
                await context.close()
                await browser.close()

    def normalize_youtube_url(self, url: str) -> str:
        """Normalize YouTube URL by removing tracking parameters and standardizing format"""
        if not url or '/watch?v=' not in url:
            return url
            
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            
            # Keep only the essential parameters
            essential_params = {}
            if 'v' in query_params:
                essential_params['v'] = query_params['v'][0]
            
            # Rebuild the URL without tracking parameters
            normalized_query = '&'.join([f"{k}={v}" for k, v in essential_params.items()])
            normalized_url = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                normalized_query,
                ''  # Remove fragment
            ))
            
            return normalized_url
        except Exception as e:
            logger.error(f"Error normalizing URL {url}: {str(e)}")
            return url

    async def check_channel(self, context, url, handle, channel_id):
        try:
            page = await context.new_page()
            
            # Set a reasonable timeout
            page.set_default_timeout(10000)
            
            # Navigate to the URL
            await page.goto(url, wait_until='domcontentloaded')
            
            # Wait for the page to load completely
            await page.wait_for_load_state('networkidle', timeout=10000)
            
            # Get the page content
            content = await page.content()
            
            if '"isLiveNow":true' in content:
                canonical_url = self.extract_canonical_url(content)
                normalized_url = self.normalize_youtube_url(canonical_url)
                await page.close()
                return self.make_result(handle, channel_id, live=True, video_url=normalized_url)
            
            canonical_url = self.extract_canonical_url(content)
            if canonical_url and '/watch?v=' in canonical_url:
                video_id = canonical_url.split('watch?v=')[1].split('&')[0]
                if video_id not in self.WAITING_ROOM_URLS:
                    normalized_url = self.normalize_youtube_url(canonical_url)
                    if self.is_live(content):
                        await page.close()
                        return self.make_result(handle, channel_id, live=True, video_url=normalized_url)
                    else:
                        await page.close()
                        return self.make_result(handle, channel_id, live=False, video_url=normalized_url, scheduled=True)
            
            alt_url = self.find_alt_video(content)
            if alt_url:
                normalized_alt_url = self.normalize_youtube_url(alt_url)
                await page.close()
                return await self.check_alt_url(context, normalized_alt_url, handle, channel_id)
            
            await page.close()
            return self.make_result(handle, channel_id)
        except Exception as e:
            logger.error(f"Error checking channel {handle}: {str(e)}")
            if 'page' in locals():
                await page.close()
            return self.make_result(handle, channel_id, error=str(e))

    async def check_alt_url(self, context, alt_url, handle, channel_id):
        try:
            page = await context.new_page()
            await page.goto(alt_url, wait_until='domcontentloaded')
            await page.wait_for_load_state('networkidle', timeout=10000)
            content = await page.content()
            
            if self.is_live(content):
                await page.close()
                return self.make_result(handle, channel_id, live=True, video_url=alt_url)
            else:
                await page.close()
                return self.make_result(handle, channel_id, live=False, video_url=alt_url, scheduled=True)
        except Exception as e:
            logger.error(f"Error checking alt URL {alt_url}: {str(e)}")
            if 'page' in locals():
                await page.close()
            return self.make_result(handle, channel_id, error=f"Alt URL check failed: {str(e)}")

    def make_result(self, handle, channel_id, live=False, video_url=None, scheduled=False, error=None):
        return {
            "handle": handle,
            "channel_id": channel_id,
            "live": live,
            "video_url": video_url,
            "scheduled": scheduled,
            "error": error
        }

    def extract_canonical_url(self, html):
        start = html.find('<link rel="canonical" href="')
        if start == -1:
            return None
        start += len('<link rel="canonical" href="')
        end = html.find('"', start)
        return html[start:end] if end != -1 else None

    def is_live(self, html):
        if self.scheduled_pattern.search(html):
            return False
        if '"isLiveNow":true' in html or 'hqdefault_live.jpg' in html:
            return True
        if 'watching now' in html.lower():
            if re.search(r'(\d+[,.]?\d*\s*watching now)', html, re.IGNORECASE):
                return True
        return False

    def find_alt_video(self, html):
        start = 0
        while True:
            start = html.find('id="video-title"', start)
            if start == -1:
                break
            href_start = html.find('href="', start)
            if href_start == -1:
                start += 1
                continue
            href_start += len('href="')
            href_end = html.find('"', href_start)
            if href_end == -1:
                start += 1
                continue
            href = html[href_start:href_end]
            if '/watch?v=' in href:
                video_id = href.split('/watch?v=')[1].split('&')[0]
                if video_id not in self.WAITING_ROOM_URLS:
                    return f"https://www.youtube.com{href}"
            start = href_end + 1
        return None

class YouTubeAPI:
    def __init__(self, api_key, headers):
        self.api_key = api_key
        self.headers = headers

    async def fetch_video_details(self, session, video_ids):
        if not video_ids:
            logger.info("No video IDs to fetch")
            return None
        id_string = ",".join(video_ids)
        url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,contentDetails,liveStreamingDetails&id={id_string}&key={self.api_key}"
        logger.debug(f"Fetching video details from: {url}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    logger.debug(f"API response status: {response.status}")
                    if response.status != 200:
                        logger.error(f"API request failed with status {response.status}")
                        return None
                    try:
                        data = await response.json()
                        if not isinstance(data, dict):
                            logger.error(f"API response is not a dictionary: {data}")
                            return None
                        if 'items' not in data:
                            logger.error("API response missing 'items' key")
                            return None
                        return data
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON response: {e}")
                        return None
        except Exception as e:
            logger.error(f"Error fetching video details: {str(e)}")
            return None

    def process_video_details(self, data, video_urls):
        combined_data = []
        if not data or not isinstance(data, dict) or 'items' not in data:
            logger.error(f"Invalid data for processing: {data}")
            return combined_data
        logger.debug(f"Processing {len(data['items'])} items with {len(video_urls)} video URLs")
        try:
            # Create a mapping of video_id to video_url to avoid index mismatches
            video_url_map = {}
            for url in video_urls:
                if url and 'v=' in url:
                    video_id = url.split('v=')[1].split('&')[0]
                    video_url_map[video_id] = url
            
            for i, item in enumerate(data['items']):
                if not isinstance(item, dict):
                    logger.warning(f"Skipping non-dict item at index {i}: {item}")
                    continue
                video_id = item.get('id', 'unknown')
                snippet = item.get('snippet')
                if not isinstance(snippet, dict):
                    logger.warning(f"Skipping item with invalid snippet for video {video_id} at index {i}: {snippet}")
                    continue
                title = snippet.get('title', 'No title')
                channel_title = snippet.get('channelTitle', 'Unknown channel')
                video_url = video_url_map.get(video_id)
                thumbnail = snippet.get("thumbnails", {}).get("high", {}).get("url", "No thumbnail")
                streaming_details = item.get('liveStreamingDetails', {})
                if not isinstance(streaming_details, dict):
                    logger.warning(f"Invalid streaming_details for video {video_id} at index {i}: {streaming_details}")
                    continue
                status = "live" if streaming_details.get('actualStartTime') else "upcoming"
                # Return a dictionary instead of a tuple for better readability
                combined_data.append({
                    "channel_title": channel_title,
                    "title": title,
                    "video_url": video_url,
                    "thumbnail": thumbnail,
                    "streaming_details": streaming_details,
                    "status": status
                })
        except Exception as e:
            logger.error(f"Error processing video details: {str(e)}, item: {item}")
        return combined_data

def extract_video_ids(video_urls):
    ids = []
    for url in video_urls:
        if url and "v=" in url:
            video_id = url.split("v=")[1].split('&')[0]
            ids.append(video_id)
    return list(set(ids))  # Remove duplicates

async def check_channels():
    # Check cache first
    cache_key = "check_channels_result"
    cached_result = cache.get(cache_key)
    if cached_result:
        logger.info("Returning cached result")
        return cached_result
        
    start_time = time.time()
    checker = YTLiveChecker()
    results = await checker.check_all_channels()
    # Deduplicate streams to avoid duplicate video URLs
    seen_urls = set()
    live_handles = []
    scheduled_streams = []
    for result in results:
        video_url = result.get('video_url')
        if video_url and video_url not in seen_urls:
            if result.get('live'):
                live_handles.append((result['handle'], video_url))
            elif result.get('scheduled'):
                scheduled_streams.append((result['handle'], video_url))
            seen_urls.add(video_url)
    all_streams = [*live_handles, *scheduled_streams]
    video_urls = [t[1] for t in all_streams if t[1] is not None]
    video_ids = extract_video_ids(video_urls)
    youtube_api = YouTubeAPI(api_key, headers)
    combined_data = []

    if video_ids:
        async with aiohttp.ClientSession() as session:
            data = await youtube_api.fetch_video_details(session, video_ids)
            if data:
                combined_data = youtube_api.process_video_details(data, video_urls)
            else:
                logger.warning("No valid data returned from YouTube API")
    else:
        logger.info("No video IDs to process")

    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000
    avg_time = duration_ms / len(results) if results else 0

    result = {
        "data": combined_data,
        "metrics": {
            "total_execution_time_ms": duration_ms,
            "channels_checked": len(results),
            "average_time_per_channel_ms": avg_time,
            "live_streams_count": len(live_handles),
            "scheduled_streams_count": len(scheduled_streams)
        }
    }
    
    # Cache the result
    cache.set(cache_key, result)
    return result

@app.get("/")
async def root():
    return {
        "message": "YouTube Live Checker API",
        "version": "1.0.0",
        "endpoints": {
            "/check": "Check all channels for live streams",
            "/live": "Get only live streams",
            "/upcoming": "Get only upcoming streams",
            "/metrics": "Get performance metrics from last check",
            "/debug/results": "Get raw results for debugging"
        }
    }

@app.get("/check")
async def check_all_channels():
    try:
        results = await check_channels()
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"Error in check_all_channels: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking channels: {str(e)}")

@app.get("/live")
async def get_live_streams():
    try:
        results = await check_channels()
        if not results or "data" not in results or not isinstance(results["data"], list):
            logger.warning(f"Invalid results data: {results}")
            return {"live_streams": [], "count": 0}

        live_streams = []
        for stream in results["data"]:
            try:
                if not isinstance(stream, dict):
                    logger.warning(f"Skipping non-dict stream: {stream}")
                    continue
                if stream.get("status") == "live":
                    live_streams.append(stream)
                    logger.debug(f"Added live stream: {stream}")
            except Exception as e:
                logger.error(f"Error processing stream: {str(e)}, Stream: {stream}")
                continue

        return {
            "live_streams": live_streams,
            "count": len(live_streams)
        }
    except Exception as e:
        logger.error(f"Error in get_live_streams: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting live streams: {str(e)}")

@app.get("/upcoming")
async def get_upcoming_streams():
    try:
        results = await check_channels()
        if not results or "data" not in results or not isinstance(results["data"], list):
            logger.warning(f"Invalid results data: {results}")
            return {"upcoming_streams": [], "count": 0}

        upcoming_streams = []
        for stream in results["data"]:
            try:
                if not isinstance(stream, dict):
                    logger.warning(f"Skipping non-dict stream: {stream}")
                    continue
                if stream.get("status") == "upcoming":
                    upcoming_streams.append(stream)
            except Exception as e:
                logger.error(f"Error processing stream: {str(e)}, Stream: {stream}")
                continue

        return {
            "upcoming_streams": upcoming_streams,
            "count": len(upcoming_streams)
        }
    except Exception as e:
        logger.error(f"Error in get_upcoming_streams: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting upcoming streams: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    try:
        results = await check_channels()
        return results.get("metrics", {})
    except Exception as e:
        logger.error(f"Error in get_metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")

@app.get("/debug/results")
async def debug_results():
    try:
        results = await check_channels()
        return JSONResponse(content=results)
    except Exception as e:
        logger.error(f"Error in debug_results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching results: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)