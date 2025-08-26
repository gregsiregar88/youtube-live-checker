import re
import json
import os
import asyncio
import aiohttp
import time
import sys
from dotenv import load_dotenv
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

load_dotenv()

headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:142.0) Gecko/20100101 Firefox/142.0"
}

livenow = []
upcoming = []
api_key = os.getenv("API_KEY")

# Initialize FastAPI app
app = FastAPI(title="YouTube Live Checker API", version="1.0.0")

# Add CORS middleware to allow requests from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class YTLiveChecker:
    def __init__(self, channels_file=None):
        # Use absolute path for Railway
        self.channels_file = channels_file or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            'channels_with_id.json'
        )
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
        """Load channels from JSON file."""
        try:
            with open(self.channels_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: Channels file not found at {self.channels_file}")
            return []
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in channels file")
            return []

    async def check_all_channels(self):
        """Check all channels for live status asynchronously."""
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
        """Check if a specific channel is live."""
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return self.make_result(handle, channel_id, error=f"HTTP {resp.status}")
                text = await resp.text()

            # Direct check for live stream
            if '"isLiveNow":true' in text:
                return self.make_result(handle, channel_id, live=True, 
                                       video_url=self.extract_canonical_url(text))

            # Check for scheduled streams
            canonical_url = self.extract_canonical_url(text)
            if canonical_url and '/watch?v=' in canonical_url:
                video_id = canonical_url.split('watch?v=')[1].split('&')[0]
                if video_id not in self.WAITING_ROOM_URLS:
                    if self.is_live(text):
                        return self.make_result(handle, channel_id, live=True, video_url=canonical_url)
                    else:
                        return self.make_result(handle, channel_id, live=False, 
                                               video_url=canonical_url, scheduled=True)

            # Check for alternative video URLs
            alt_url = self.find_alt_video(text)
            if alt_url:
                return await self.check_alt_url(session, alt_url, handle, channel_id)

            return self.make_result(handle, channel_id)

        except Exception as e:
            return self.make_result(handle, channel_id, error=str(e))

    async def check_alt_url(self, session, alt_url, handle, channel_id):
        """Check alternative URL for live status."""
        try:
            async with session.get(alt_url) as alt_resp:
                if alt_resp.status == 200:
                    alt_text = await alt_resp.text()
                    if self.is_live(alt_text):
                        return self.make_result(handle, channel_id, live=True, video_url=alt_url)
                    else:
                        return self.make_result(handle, channel_id, live=False, 
                                               video_url=alt_url, scheduled=True)
        except Exception as e:
            return self.make_result(handle, channel_id, error=f"Alt URL check failed: {str(e)}")

    def make_result(self, handle, channel_id, live=False, video_url=None, scheduled=False, error=None):
        """Create a standardized result dictionary."""
        return {
            "handle": handle,
            "channel_id": channel_id,
            "live": live,
            "video_url": video_url,
            "scheduled": scheduled,
            "error": error
        }

    def extract_canonical_url(self, html):
        """Extract canonical URL from HTML."""
        start = html.find('<link rel="canonical" href="')
        if start == -1:
            return None
        start += len('<link rel="canonical" href="')
        end = html.find('"', start)
        return html[start:end] if end != -1 else None

    def is_live(self, html):
        """Determine if stream is live based on HTML content."""
        if self.scheduled_pattern.search(html):
            return False
        if '"isLiveNow":true' in html:
            return True
        if 'hqdefault_live.jpg' in html:
            return True
        if 'watching now' in html.lower():
            if re.search(r'(\d+[,.]?\d*\s*watching now)', html, re.IGNORECASE):
                return True
        return False

    def find_alt_video(self, html):
        """Find alternative video URLs in HTML."""
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
    """Handle YouTube API requests and data processing."""
    
    def __init__(self, api_key, headers):
        self.api_key = api_key
        self.headers = headers

    async def fetch_video_details(self, session, video_ids):
        """Fetch video details from YouTube API."""
        if not video_ids:
            return None
            
        id_string = ",".join(video_ids)
        url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,contentDetails,liveStreamingDetails&id={id_string}&key={self.api_key}"
        
        try:
            async with session.get(url, headers=self.headers) as response:
                return response
        except Exception as e:
            print(f"Error fetching video details: {e}")
            return None

    def process_video_details(self, data, video_urls):
        """Process YouTube API response data."""
        combined_data = []
        
        if not data or 'items' not in data:
            return combined_data
            
        try:
            for i, item in enumerate(data['items']):
                if 'snippet' not in item:
                    continue
                    
                snippet = item['snippet']
                title = snippet.get('title', 'No title')
                channel_title = snippet.get('channelTitle', 'Unknown channel')
                
                # Get the corresponding video URL
                video_url = video_urls[i] if i < len(video_urls) else None
                
                # Determine stream status
                streaming_details = item.get('liveStreamingDetails', {})
                status = "live" if streaming_details and 'actualStartTime' in streaming_details else "upcoming"
                streaming_details['status'] = status
                
                combined_data.append((channel_title, title, video_url, streaming_details))
                
        except Exception as e:
            print(f"Error processing video details: {e}")
            
        return combined_data


def extract_video_ids(video_urls):
    """Extract video IDs from YouTube URLs."""
    ids = []
    for url in video_urls:
        if url and "v=" in url:
            video_id = url.split("v=")[1].split('&')[0]
            ids.append(video_id)
    return ids


async def check_channels():
    """Check all channels and return processed results."""
    start_time = time.time()
    
    # Check all channels
    checker = YTLiveChecker()
    results = await checker.check_all_channels()
    
    # Process results
    live_handles = [(result['handle'], result['video_url']) 
                   for result in results if result.get('live')]
    
    scheduled_streams = [(result['handle'], result['video_url']) 
                        for result in results if result.get('scheduled')]
    
    all_streams = [*live_handles, *scheduled_streams]
    video_urls = [t[1] for t in all_streams if t[1] is not None]
    video_ids = extract_video_ids(video_urls)
    
    # Fetch and process YouTube API data
    youtube_api = YouTubeAPI(api_key, headers)
    combined_data = []
    
    if video_ids:
        async with aiohttp.ClientSession() as session:
            response = await youtube_api.fetch_video_details(session, video_ids)
            
            if response and response.status == 200:
                try:
                    data = await response.json()
                    combined_data = youtube_api.process_video_details(data, video_urls)
                except Exception as e:
                    print(f"Error parsing YouTube API response: {e}")
            else:
                status = response.status if response else "No response"
                print(f"YouTube API error: {status}")
    
    # Calculate performance metrics
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


# FastAPI Routes
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "YouTube Live Checker API",
        "version": "1.0.0",
        "endpoints": {
            "/check": "Check all channels for live streams",
            "/live": "Get only live streams",
            "/upcoming": "Get only upcoming streams",
            "/metrics": "Get performance metrics from last check"
        }
    }


@app.get("/check")
async def check_all_channels():
    """Check all channels and return results."""
    try:
        results = await check_channels()
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking channels: {str(e)}")


@app.get("/live")
async def get_live_streams():
    """Get only live streams from the last check."""
    try:
        results = await check_channels()
        # Safely access the data with proper error handling
        if not results or "data" not in results:
            return {"live_streams": [], "count": 0}
            
        live_streams = []
        for stream in results["data"]:
            try:
                if len(stream) >= 4 and isinstance(stream[3], dict) and stream[3].get('status') == 'live':
                    live_streams.append(stream)
            except (IndexError, TypeError, AttributeError) as e:
                print(f"Error processing stream data: {e}")
                continue
                
        return {
            "live_streams": live_streams,
            "count": len(live_streams)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting live streams: {str(e)}")


@app.get("/upcoming")
async def get_upcoming_streams():
    """Get only upcoming streams from the last check."""
    try:
        results = await check_channels()
        # Safely access the data with proper error handling
        if not results or "data" not in results:
            return {"upcoming_streams": [], "count": 0}
            
        upcoming_streams = []
        for stream in results["data"]:
            try:
                if len(stream) >= 4 and isinstance(stream[3], dict) and stream[3].get('status') == 'upcoming':
                    upcoming_streams.append(stream)
            except (IndexError, TypeError, AttributeError) as e:
                print(f"Error processing stream data: {e}")
                continue
                
        return {
            "upcoming_streams": upcoming_streams,
            "count": len(upcoming_streams)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting upcoming streams: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Get performance metrics from the last check."""
    try:
        results = await check_channels()
        return results.get("metrics", {})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")


if __name__ == "__main__":
    # Get port from Railway environment variable or default to 8000
    port = int(os.getenv("PORT", 8000))
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)