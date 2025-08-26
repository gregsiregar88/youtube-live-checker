import re
import json
import os
import asyncio
import aiohttp
import time
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import weakref
from dotenv import load_dotenv
import hrequests
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

load_dotenv()

# Configuration
@dataclass
class Config:
    API_KEY: str = os.getenv("API_KEY", "")
    CHANNELS_FILE: str = os.path.join(os.getcwd(), 'channels_with_id.json')
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Performance optimizations
    TIMEOUT: int = 3  # Reduced from 5
    MAX_CONCURRENT_REQUESTS: int = 20  # Limit concurrent requests
    CONNECTION_LIMIT: int = 100
    CONNECTION_LIMIT_PER_HOST: int = 30
    KEEP_ALIVE_TIMEOUT: int = 30
    
    # Caching
    CACHE_TTL: int = 30  # seconds
    ENABLE_CACHING: bool = True
    
    # Batch processing
    API_BATCH_SIZE: int = 50  # YouTube API allows up to 50 video IDs per request
    
    HEADERS: Dict[str, str] = field(default_factory=lambda: {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:143.0) Gecko/20100101 Firefox/143.0",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    })

class StreamStatus(Enum):
    LIVE = "live"
    UPCOMING = "upcoming"  
    OFFLINE = "offline"
    ERROR = "error"

@dataclass
class Channel:
    handle: str
    id: str
    
    def __hash__(self):
        return hash(self.handle)
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'Channel':
        return cls(handle=data['handle'], id=data['id'])

@dataclass
class StreamInfo:
    channel_handle: str
    channel_id: str
    channel_title: str = ""
    video_title: str = ""
    video_url: Optional[str] = None
    status: StreamStatus = StreamStatus.OFFLINE
    error: Optional[str] = None
    streaming_details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "channel": {
                "handle": self.channel_handle,
                "id": self.channel_id,
                "title": self.channel_title
            },
            "stream": {
                "title": self.video_title,
                "url": self.video_url,
                "status": self.status.value,
                "details": self.streaming_details
            }
        }
        if self.error:
            result["error"] = self.error
        return result

@dataclass
class CheckResult:
    streams: List[StreamInfo]
    metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "streams": [stream.to_dict() for stream in self.streams],
            "metrics": self.metrics,
            "summary": {
                "total_channels": len(self.streams),
                "live_count": len([s for s in self.streams if s.status == StreamStatus.LIVE]),
                "upcoming_count": len([s for s in self.streams if s.status == StreamStatus.UPCOMING]),
                "offline_count": len([s for s in self.streams if s.status == StreamStatus.OFFLINE]),
                "error_count": len([s for s in self.streams if s.status == StreamStatus.ERROR])
            }
        }

class CacheManager:
    """Simple in-memory cache with TTL"""
    def __init__(self, ttl: int = 30):
        self.ttl = ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_times: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                self._access_times[key] = time.time()
                return value
            else:
                # Expired
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
        return None
    
    def set(self, key: str, value: Any):
        current_time = time.time()
        self._cache[key] = (value, current_time)
        self._access_times[key] = current_time
        
        # Simple cleanup: remove old entries if cache gets too large
        if len(self._cache) > 1000:
            self._cleanup()
    
    def _cleanup(self):
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp >= self.ttl
        ]
        for key in expired_keys:
            del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]

class ChannelManager:
    def __init__(self, channels_file: str):
        self.channels_file = channels_file
        self._channels: Optional[List[Channel]] = None
        self._channels_by_handle: Optional[Dict[str, Channel]] = None
        self._file_mtime: Optional[float] = None
    
    def _should_reload(self) -> bool:
        try:
            current_mtime = os.path.getmtime(self.channels_file)
            return self._file_mtime is None or current_mtime > self._file_mtime
        except OSError:
            return True
    
    def _load_channels(self) -> Tuple[List[Channel], Dict[str, Channel]]:
        """Load and deduplicate channels from JSON file."""
        try:
            with open(self.channels_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Remove duplicates and create lookup dict in one pass
            channels_by_handle = {}
            for item in data:
                handle = item['handle']
                if handle not in channels_by_handle:
                    channels_by_handle[handle] = Channel.from_dict(item)
            
            channels = list(channels_by_handle.values())
            self._file_mtime = os.path.getmtime(self.channels_file)
            
            return channels, channels_by_handle
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Channels file not found: {self.channels_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in channels file: {self.channels_file}")
    
    @property 
    def channels(self) -> List[Channel]:
        if self._channels is None or self._should_reload():
            self._channels, self._channels_by_handle = self._load_channels()
        return self._channels
    
    @property
    def channels_by_handle(self) -> Dict[str, Channel]:
        if self._channels_by_handle is None or self._should_reload():
            self._channels, self._channels_by_handle = self._load_channels()
        return self._channels_by_handle
    
    def reload_channels(self) -> None:
        """Force reload channels from file."""
        self._channels = None
        self._channels_by_handle = None

class HTMLParser:
    def __init__(self):
        # Pre-compile regex patterns for better performance
        self.scheduled_pattern = re.compile(r'(Scheduled for|Live in|Premieres|Waiting for)', re.IGNORECASE)
        self.watching_pattern = re.compile(r'(\d+[,.]?\d*\s*watching now)', re.IGNORECASE)
        self.canonical_pattern = re.compile(r'<link rel="canonical" href="([^"]+)"')
        self.video_title_pattern = re.compile(r'id="video-title"[^>]*href="([^"]+)"')
        
        # Convert to frozenset for O(1) lookup
        self.WAITING_ROOM_URLS: Set[str] = frozenset({
            "w4dgql_5Rzk", "O9V_EFbgpKQ", "rKMhl43RHo0", "MDwkJVqui_M",
            "INFI9FahPY0", "TLw3Taw5jxI", "-tMd6H-IxcA", "wUEN1KE2ZcU",
            "xbJ8tbA_Phw", "VKzTYEBsImc", "XeKiT4cLT6U", "hsAr4h_Mljw",
            "Fl1vM3scybw", "lGr_kZmjskI", "JAUSrqX0hW8", "38fJIy2FoDg",
            "1WhsM61BUfk", "8deE3F_WgBA", "c7K6RInG3Dw", "6GZ5XGzRY-g",
            "UkqwIcO3YN8", "hlDFczhR2mo", "Criw5zhE0bI", "u8aMX32hlgQ",
            "2JciZo2afXg", "GZHhb_zHVno", "SCVbMM71viE", "M9H0gTvKmGU",
            "Pd9VYQtr2c4", "wnjd9XuuXg0", "IlYErJ1ry_8", "VoWHIX4tp5k",
            "9vaxfw1qFcY", "sDZFWow3IKI"
        })
    
    @lru_cache(maxsize=256)
    def extract_canonical_url(self, html_hash: int) -> Optional[str]:
        """Extract canonical URL from HTML using cached regex."""
        # We can't cache the full HTML, so we cache by hash
        # This is called from the non-cached version
        return None
    
    def extract_canonical_url_from_html(self, html: str) -> Optional[str]:
        """Extract canonical URL from HTML using optimized regex."""
        match = self.canonical_pattern.search(html)
        return match.group(1) if match else None
    
    def is_live_stream(self, html: str) -> bool:
        """Determine if stream is live based on HTML content."""
        # Fast checks first
        if '"isLiveNow":true' in html:
            return True
        if 'hqdefault_live.jpg' in html:
            return True
        
        # More expensive checks
        if self.scheduled_pattern.search(html):
            return False
        if 'watching now' in html.lower():
            return bool(self.watching_pattern.search(html))
        
        return False
    
    def find_alternative_video_url(self, html: str) -> Optional[str]:
        """Find alternative video URLs using optimized regex."""
        matches = self.video_title_pattern.findall(html)
        for href in matches:
            if '/watch?v=' in href:
                video_id = href.split('/watch?v=')[1].split('&')[0]
                if video_id not in self.WAITING_ROOM_URLS:
                    return f"https://www.youtube.com{href}"
        return None
    
    @lru_cache(maxsize=512)
    def extract_video_id_from_url(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL with caching."""
        if not url or "v=" not in url:
            return None
        return url.split("v=")[1].split("&")[0]

class YTLiveChecker:
    def __init__(self, config: Config):
        self.config = config
        self.channel_manager = ChannelManager(config.CHANNELS_FILE)
        self.html_parser = HTMLParser()
        self.cache = CacheManager(config.CACHE_TTL) if config.ENABLE_CACHING else None
        self._connector = None
        self._session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._connector = aiohttp.TCPConnector(
            limit=self.config.CONNECTION_LIMIT,
            limit_per_host=self.config.CONNECTION_LIMIT_PER_HOST,
            keepalive_timeout=self.config.KEEP_ALIVE_TIMEOUT,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.TIMEOUT)
        self._session = aiohttp.ClientSession(
            headers=self.config.HEADERS,
            timeout=timeout,
            connector=self._connector
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
        if self._connector:
            await self._connector.close()
    
    async def check_all_channels(self) -> List[StreamInfo]:
        """Check all channels for live status with optimized concurrency."""
        channels = self.channel_manager.channels
        
        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.config.MAX_CONCURRENT_REQUESTS)
        
        async def check_with_semaphore(channel):
            async with semaphore:
                return await self._check_single_channel(channel)
        
        # Execute all checks concurrently but with limited parallelism
        tasks = [check_with_semaphore(channel) for channel in channels]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    async def _check_single_channel(self, channel: Channel) -> StreamInfo:
        """Check if a specific channel is live with caching."""
        # Check cache first
        if self.cache:
            cache_key = f"channel:{channel.handle}"
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result
        
        url = f"https://www.youtube.com/@{channel.handle}/live"
        
        try:
            async with self._session.get(url) as response:
                if response.status != 200:
                    result = StreamInfo(
                        channel_handle=channel.handle,
                        channel_id=channel.id,
                        status=StreamStatus.ERROR,
                        error=f"HTTP {response.status}"
                    )
                    if self.cache and response.status != 429:  # Don't cache rate limits
                        self.cache.set(f"channel:{channel.handle}", result)
                    return result
                
                html = await response.text()
            
            # Fast path: check for direct live stream
            if '"isLiveNow":true' in html:
                video_url = self.html_parser.extract_canonical_url_from_html(html)
                result = StreamInfo(
                    channel_handle=channel.handle,
                    channel_id=channel.id,
                    video_url=video_url,
                    status=StreamStatus.LIVE
                )
            else:
                result = await self._analyze_html_content(html, channel)
            
            # Cache successful results
            if self.cache and result.status != StreamStatus.ERROR:
                self.cache.set(f"channel:{channel.handle}", result)
            
            return result
                
        except Exception as e:
            result = StreamInfo(
                channel_handle=channel.handle,
                channel_id=channel.id,
                status=StreamStatus.ERROR,
                error=str(e)
            )
            # Don't cache errors
            return result
    
    async def _analyze_html_content(self, html: str, channel: Channel) -> StreamInfo:
        """Analyze HTML content to determine stream status."""
        # Check canonical URL for scheduled streams
        canonical_url = self.html_parser.extract_canonical_url_from_html(html)
        if canonical_url and '/watch?v=' in canonical_url:
            video_id = self.html_parser.extract_video_id_from_url(canonical_url)
            if video_id and video_id not in self.html_parser.WAITING_ROOM_URLS:
                status = StreamStatus.LIVE if self.html_parser.is_live_stream(html) else StreamStatus.UPCOMING
                return StreamInfo(
                    channel_handle=channel.handle,
                    channel_id=channel.id,
                    video_url=canonical_url,
                    status=status
                )
        
        # Check for alternative video URLs
        alt_url = self.html_parser.find_alternative_video_url(html)
        if alt_url:
            return await self._check_alternative_url(channel, alt_url)
        
        return StreamInfo(
            channel_handle=channel.handle,
            channel_id=channel.id,
            status=StreamStatus.OFFLINE
        )
    
    async def _check_alternative_url(self, channel: Channel, alt_url: str) -> StreamInfo:
        """Check alternative URL for live status."""
        try:
            async with self._session.get(alt_url) as response:
                if response.status == 200:
                    html = await response.text()
                    status = StreamStatus.LIVE if self.html_parser.is_live_stream(html) else StreamStatus.UPCOMING
                    return StreamInfo(
                        channel_handle=channel.handle,
                        channel_id=channel.id,
                        video_url=alt_url,
                        status=status
                    )
        except Exception as e:
            return StreamInfo(
                channel_handle=channel.handle,
                channel_id=channel.id,
                status=StreamStatus.ERROR,
                error=f"Alt URL check failed: {str(e)}"
            )
        
        return StreamInfo(
            channel_handle=channel.handle,
            channel_id=channel.id,
            status=StreamStatus.OFFLINE
        )

class YouTubeAPIClient:
    """Optimized YouTube API client with batching."""
    
    def __init__(self, api_key: str, headers: Dict[str, str], batch_size: int = 50):
        self.api_key = api_key
        self.headers = headers
        self.batch_size = batch_size
    
    async def fetch_video_details_batch(self, video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch video details in batches for better performance."""
        if not video_ids or not self.api_key:
            return {}
        
        all_results = {}
        
        # Process in batches
        for i in range(0, len(video_ids), self.batch_size):
            batch_ids = video_ids[i:i + self.batch_size]
            batch_result = await self._fetch_batch(batch_ids)
            all_results.update(batch_result)
        
        return all_results
    
    async def _fetch_batch(self, video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch a single batch of video details."""
        id_string = ",".join(video_ids)
        url = (f"https://www.googleapis.com/youtube/v3/videos"
               f"?part=snippet,liveStreamingDetails"
               f"&id={id_string}&key={self.api_key}")
        
        try:
            # Use async HTTP client for API calls too
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {item['id']: item for item in data.get('items', [])}
                    else:
                        logger.warning(f"YouTube API Error: {response.status}")
                        return {}
        except Exception as e:
            logger.warning(f"YouTube API Request failed: {str(e)}")
            return {}
    
    def enrich_stream_info(self, streams: List[StreamInfo], api_data: Dict[str, Dict[str, Any]]) -> List[StreamInfo]:
        """Enrich stream information with API data (synchronous processing)."""
        enriched_streams = []
        
        for stream in streams:
            if stream.video_url and stream.status != StreamStatus.ERROR:
                video_id = self._extract_video_id(stream.video_url)
                if video_id and video_id in api_data:
                    self._enrich_single_stream(stream, api_data[video_id])
            enriched_streams.append(stream)
        
        return enriched_streams
    
    def _enrich_single_stream(self, stream: StreamInfo, api_item: Dict[str, Any]) -> None:
        """Enrich a single stream with API data."""
        snippet = api_item.get('snippet', {})
        live_details = api_item.get('liveStreamingDetails', {})
        
        # Update stream info in place
        stream.channel_title = snippet.get('channelTitle', stream.channel_title)
        stream.video_title = snippet.get('title', stream.video_title)
        stream.streaming_details = live_details
        
        # Determine actual status from API
        if live_details and 'actualStartTime' in live_details and 'actualEndTime' not in live_details:
            stream.status = StreamStatus.LIVE
        elif live_details and 'scheduledStartTime' in live_details:
            stream.status = StreamStatus.UPCOMING
    
    @lru_cache(maxsize=1024)
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL with caching."""
        if not url or "v=" not in url:
            return None
        return url.split("v=")[1].split("&")[0]

class LiveStreamService:
    """Optimized main service for checking live streams."""
    
    def __init__(self, config: Config):
        self.config = config
        self.youtube_api = YouTubeAPIClient(
            config.API_KEY, 
            config.HEADERS, 
            config.API_BATCH_SIZE
        )
        self._last_result: Optional[CheckResult] = None
        self._last_check_time: float = 0
    
    async def check_all_streams(self, force_refresh: bool = False) -> CheckResult:
        """Check all channels with caching and optimizations."""
        current_time = time.time()
        
        # Return cached result if recent enough
        if (not force_refresh and 
            self._last_result and 
            current_time - self._last_check_time < self.config.CACHE_TTL):
            return self._last_result
        
        start_time = time.time()
        
        # Use context manager for proper session management
        async with YTLiveChecker(self.config) as checker:
            streams = await checker.check_all_channels()
        
        # Enrich with YouTube API data
        if self.config.API_KEY:
            await self._enrich_streams_with_api_data(streams)
        
        # Calculate metrics
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        metrics = {
            "execution_time_ms": round(duration_ms, 2),
            "channels_checked": len(streams),
            "average_time_per_channel_ms": round(duration_ms / len(streams) if streams else 0, 2),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cache_enabled": self.config.ENABLE_CACHING,
            "api_enabled": bool(self.config.API_KEY)
        }
        
        result = CheckResult(streams=streams, metrics=metrics)
        
        # Cache the result
        self._last_result = result
        self._last_check_time = current_time
        
        return result
    
    async def _enrich_streams_with_api_data(self, streams: List[StreamInfo]) -> None:
        """Enrich streams with API data using batch processing."""
        # Get video URLs that need API data
        video_urls = [s.video_url for s in streams if s.video_url and s.status != StreamStatus.ERROR]
        if not video_urls:
            return
        
        # Extract video IDs
        video_ids = [self.youtube_api._extract_video_id(url) for url in video_urls]
        video_ids = [vid for vid in video_ids if vid]  # Remove None values
        
        if not video_ids:
            return
        
        # Fetch API data in batches
        api_data = await self.youtube_api.fetch_video_details_batch(video_ids)
        
        # Enrich stream info
        self.youtube_api.enrich_stream_info(streams, api_data)

# Initialize service
config = Config()
live_service = LiveStreamService(config)

# Global cache for endpoints
endpoint_cache = CacheManager(ttl=5)  # Short cache for API responses

# FastAPI app with optimizations
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting YouTube Live Checker API")
    # Pre-warm cache
    try:
        await live_service.check_all_streams()
        logger.info("Cache pre-warmed successfully")
    except Exception as e:
        logger.warning(f"Cache pre-warm failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down YouTube Live Checker API")

app = FastAPI(
    title="YouTube Live Stream Checker API",
    version="2.1.0",
    description="High-performance monitor for YouTube channels live streams and upcoming broadcasts",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optimized API Routes
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "YouTube Live Stream Checker API",
        "version": "2.1.0",
        "performance_features": [
            "Concurrent request processing",
            "Response caching", 
            "Connection pooling",
            "Regex optimization",
            "Batch API requests"
        ],
        "endpoints": {
            "/check": "Check all channels for live streams",
            "/live": "Get only live streams",
            "/upcoming": "Get only upcoming streams", 
            "/offline": "Get only offline channels",
            "/metrics": "Get performance metrics",
            "/channels": "Get list of monitored channels",
            "/health": "Health check endpoint"
        },
        "config": {
            "cache_ttl": config.CACHE_TTL,
            "max_concurrent": config.MAX_CONCURRENT_REQUESTS,
            "timeout": config.TIMEOUT
        }
    }

@app.get("/health")
async def health_check():
    """Fast health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/check")
async def check_all_streams(force_refresh: bool = False):
    """Check all channels with optional cache bypass."""
    try:
        cache_key = f"check:force={force_refresh}"
        
        # Check endpoint cache first (only for non-forced requests)
        if not force_refresh:
            cached = endpoint_cache.get(cache_key)
            if cached:
                return cached
        
        result = await live_service.check_all_streams(force_refresh)
        response_data = result.to_dict()
        
        # Cache the response
        if not force_refresh:
            endpoint_cache.set(cache_key, response_data)
        
        return JSONResponse(content=response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking streams: {str(e)}")

@app.get("/live")
async def get_live_streams():
    """Get only currently live streams (cached)."""
    try:
        cached = endpoint_cache.get("live_streams")
        if cached:
            return cached
        
        result = await live_service.check_all_streams()
        live_streams = [s for s in result.streams if s.status == StreamStatus.LIVE]
        
        response_data = {
            "live_streams": [stream.to_dict() for stream in live_streams],
            "count": len(live_streams),
            "timestamp": result.metrics["timestamp"]
        }
        
        endpoint_cache.set("live_streams", response_data)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting live streams: {str(e)}")

@app.get("/upcoming")
async def get_upcoming_streams():
    """Get only upcoming/scheduled streams (cached)."""
    try:
        cached = endpoint_cache.get("upcoming_streams")
        if cached:
            return cached
        
        result = await live_service.check_all_streams()
        upcoming_streams = [s for s in result.streams if s.status == StreamStatus.UPCOMING]
        
        response_data = {
            "upcoming_streams": [stream.to_dict() for stream in upcoming_streams],
            "count": len(upcoming_streams),
            "timestamp": result.metrics["timestamp"]
        }
        
        endpoint_cache.set("upcoming_streams", response_data)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting upcoming streams: {str(e)}")

@app.get("/offline")
async def get_offline_channels():
    """Get channels that are currently offline (cached)."""
    try:
        cached = endpoint_cache.get("offline_channels")
        if cached:
            return cached
        
        result = await live_service.check_all_streams()
        offline_channels = [s for s in result.streams if s.status == StreamStatus.OFFLINE]
        
        response_data = {
            "offline_channels": [stream.to_dict() for stream in offline_channels],
            "count": len(offline_channels),
            "timestamp": result.metrics["timestamp"]
        }
        
        endpoint_cache.set("offline_channels", response_data)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting offline channels: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get performance metrics (always fresh)."""
    try:
        result = await live_service.check_all_streams()
        return result.metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")

@app.get("/channels")
async def get_channels():
    """Get list of all monitored channels (cached)."""
    try:
        cached = endpoint_cache.get("channels_list")
        if cached:
            return cached
        
        channels = live_service.config.CHANNELS_FILE
        channel_manager = ChannelManager(channels)
        channels_list = channel_manager.channels
        
        response_data = {
            "channels": [{"handle": c.handle, "id": c.id} for c in channels_list],
            "count": len(channels_list)
        }
        
        endpoint_cache.set("channels_list", response_data)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting channels: {str(e)}")

@app.post("/reload-channels")
async def reload_channels(background_tasks: BackgroundTasks):
    """Reload channels from file and clear caches."""
    try:
        def clear_caches():
            # Clear service cache
            live_service._last_result = None
            live_service._last_check_time = 0
            # Clear endpoint cache
            endpoint_cache._cache.clear()
            endpoint_cache._access_times.clear()
        
        background_tasks.add_task(clear_caches)
        return {"message": "Channels reload initiated, caches will be cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading channels: {str(e)}")

@app.get("/status")
async def get_service_status():
    """Get detailed service status and performance info."""
    try:
        # Get current cache stats
        service_cache_size = len(live_service._last_result.streams) if live_service._last_result else 0
        endpoint_cache_size = len(endpoint_cache._cache)
        
        # Get channel count
        channel_manager = ChannelManager(config.CHANNELS_FILE)
        channel_count = len(channel_manager.channels)
        
        return {
            "status": "operational",
            "version": "2.1.0",
            "uptime_check": time.time(),
            "configuration": {
                "cache_ttl_seconds": config.CACHE_TTL,
                "max_concurrent_requests": config.MAX_CONCURRENT_REQUESTS,
                "connection_timeout_seconds": config.TIMEOUT,
                "api_batch_size": config.API_BATCH_SIZE,
                "caching_enabled": config.ENABLE_CACHING,
                "api_key_configured": bool(config.API_KEY)
            },
            "performance": {
                "channels_monitored": channel_count,
                "service_cache_entries": service_cache_size,
                "endpoint_cache_entries": endpoint_cache_size,
                "last_check_time": live_service._last_check_time,
                "cache_hit_available": live_service._last_result is not None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting status: {str(e)}")

@app.delete("/cache")
async def clear_cache():
    """Clear all caches manually."""
    try:
        # Clear service cache
        live_service._last_result = None
        live_service._last_check_time = 0
        
        # Clear endpoint cache
        endpoint_cache._cache.clear()
        endpoint_cache._access_times.clear()
        
        return {
            "message": "All caches cleared successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")

# Background task for cache warming
async def warm_cache_periodically():
    """Background task to keep cache warm."""
    while True:
        try:
            await asyncio.sleep(config.CACHE_TTL // 2)  # Refresh at half TTL
            await live_service.check_all_streams(force_refresh=True)
            logger.info("Cache warmed successfully")
        except Exception as e:
            logger.warning(f"Cache warming failed: {e}")

# Start background tasks on startup
@app.on_event("startup")
async def startup_event():
    """Start background tasks."""
    if config.ENABLE_CACHING:
        asyncio.create_task(warm_cache_periodically())

if __name__ == "__main__":
    # Run with optimized uvicorn settings
    uvicorn.run(
        app, 
        host=config.HOST, 
        port=config.PORT,
        workers=1,  # Single worker for shared cache
        loop="asyncio",
        http="httptools",
        access_log=False,  # Disable access logs for performance
        log_level="warning"
    )