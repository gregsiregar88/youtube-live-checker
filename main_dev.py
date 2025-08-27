# MIT License
# Copyright (c) 2025 Greg Siregar

import re
import json
import os
import asyncio
import aiohttp
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
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
            "9vaxfw1qFcY", "sDZFWow3IKI"
        }

    def load_channels(self):
        if not os.path.exists(self.channels_file):
            logger.warning(f"Channels file not found at {self.channels_file}")
            return []
        try:
            with open(self.channels_file, 'r') as f:
                content = f.read()
                if not content.strip():
                    logger.warning(f"Channels file {self.channels_file} is empty")
                    return []
                channels = json.loads(content)
                if not isinstance(channels, list):
                    logger.warning(f"channels_with_id.json must contain a list")
                    return []
                for channel in channels:
                    if not isinstance(channel, dict) or 'handle' not in channel or 'id' not in channel:
                        logger.warning(f"Invalid channel format in {self.channels_file}: {channel}")
                        return []
                unique_channels = {tuple(channel.items()): channel for channel in channels}.values()
                return list(unique_channels)
        except Exception:
            return []

    async def check_all_channels(self):
        if not self.channels:
            return []
        async with aiohttp.ClientSession(
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:143.0) Gecko/20100101 Firefox/143.0",
                "Accept-Language": "en-US,en;q=0.5",
            },
            timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            tasks = [
                self.check_channel(session, f"https://www.youtube.com/@{channel['handle']}/live", 
                                  channel['handle'], channel['id'])
                for channel in self.channels
            ]
            return await asyncio.gather(*tasks)

    async def check_channel(self, session, url, handle, channel_id):
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return self.make_result(handle, channel_id, error=f"HTTP {resp.status}")
                text = await resp.text()
            canonical_url = self.extract_canonical_url(text)
            if '"isLiveNow":true' in text:
                return self.make_result(handle, channel_id, live=True, video_url=canonical_url)
            if canonical_url and '/watch?v=' in canonical_url:
                video_id = canonical_url.split('watch?v=')[1].split('&')[0]
                if video_id not in self.WAITING_ROOM_URLS:
                    if self.is_live(text):
                        return self.make_result(handle, channel_id, live=True, video_url=canonical_url)
                    else:
                        return self.make_result(handle, channel_id, live=False, video_url=canonical_url, scheduled=True)
            alt_url = self.find_alt_video(text)
            if alt_url:
                return await self.check_alt_url(session, alt_url, handle, channel_id)
            return self.make_result(handle, channel_id)
        except Exception as e:
            return self.make_result(handle, channel_id, error=str(e))

    async def check_alt_url(self, session, alt_url, handle, channel_id):
        try:
            async with session.get(alt_url) as alt_resp:
                if alt_resp.status == 200:
                    alt_text = await alt_resp.text()
                    if self.is_live(alt_text):
                        return self.make_result(handle, channel_id, live=True, video_url=alt_url)
                    else:
                        return self.make_result(handle, channel_id, live=False, video_url=alt_url, scheduled=True)
        except Exception as e:
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
        if 'watching now' in html.lower() and re.search(r'(\d+[,.]?\d*\s*watching now)', html, re.IGNORECASE):
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
            return None
        id_string = ",".join(video_ids)
        url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,contentDetails,liveStreamingDetails&id={id_string}&key={self.api_key}"
        try:
            async with session.get(url, headers=self.headers) as response:
                if response.status != 200:
                    return None
                try:
                    data = await response.json()
                    if not isinstance(data, dict) or 'items' not in data:
                        return None
                    return data
                except json.JSONDecodeError:
                    return None
        except Exception:
            return None

    def process_video_details(self, data, video_urls):
        combined_data = []
        if not data or not isinstance(data, dict) or 'items' not in data:
            return combined_data
        try:
            video_url_map = {url.split('v=')[1].split('&')[0]: url for url in video_urls if url and 'v=' in url}
            for i, item in enumerate(data['items']):
                if not isinstance(item, dict):
                    continue
                video_id = item.get('id', 'unknown')
                snippet = item.get('snippet')
                if not isinstance(snippet, dict):
                    continue
                title = snippet.get('title', 'No title')
                channel_title = snippet.get('channelTitle', 'Unknown channel')
                video_url = video_url_map.get(video_id)
                thumbnail = snippet.get("thumbnails", "No thumbnails")
                streaming_details = item.get('liveStreamingDetails', {})
                if not isinstance(streaming_details, dict):
                    continue
                status = "live" if streaming_details.get('actualStartTime') else "upcoming"
                streaming_details['status'] = status
                combined_data.append((channel_title, title, video_url, thumbnail, streaming_details))
        except Exception:
            pass
        return combined_data

def extract_video_ids(video_urls):
    ids = []
    for url in video_urls:
        if url and "v=" in url:
            video_id = url.split("v=")[1].split('&')[0]
            ids.append(video_id)
    return list(set(ids))

async def check_channels():
    start_time = time.time()
    checker = YTLiveChecker()
    results = await checker.check_all_channels()
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

    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000
    avg_time = duration_ms / len(results) if results else 0

    return {
        "data": combined_data,
        "metrics": {
            "total_execution_time_ms": duration_ms,
            "channels_checked": len(results),
            "average_time_per_channel_ms": avg_time,
            "live_streams_count": len(live_handles),
            "scheduled_streams_count": len(scheduled_streams)
        }
    }

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
        raise HTTPException(status_code=500, detail=f"Error checking channels: {str(e)}")

@app.get("/live")
async def get_live_streams():
    try:
        results = await check_channels()
        if not results or "data" not in results or not isinstance(results["data"], list):
            return {"live_streams": [], "count": 0}

        live_streams = []
        for stream in results["data"]:
            if isinstance(stream, tuple) and len(stream) >= 4 and isinstance(stream[4], dict):
                if stream[4].get('status') == 'live':
                    live_streams.append(stream)

        return {
            "live_streams": live_streams,
            "count": len(live_streams)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting live streams: {str(e)}")

@app.get("/upcoming")
async def get_upcoming_streams():
    try:
        results = await check_channels()
        if not results or "data" not in results or not isinstance(results["data"], list):
            return {"upcoming_streams": [], "count": 0}

        upcoming_streams = []
        for stream in results["data"]:
            if isinstance(stream, tuple) and len(stream) >= 4 and isinstance(stream[4], dict):
                if stream[4].get('status') == 'upcoming':
                    upcoming_streams.append(stream)

        return {
            "upcoming_streams": upcoming_streams,
            "count": len(upcoming_streams)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting upcoming streams: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    try:
        results = await check_channels()
        return results.get("metrics", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")

@app.get("/debug/results")
async def debug_results():
    try:
        results = await check_channels()
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching results: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
