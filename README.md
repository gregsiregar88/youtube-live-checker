# 🎥 YouTube Live Checker API
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)


A **FastAPI** service to monitor multiple YouTube channels for **live** and **upcoming streams**, with metrics and debugging support.  

**Base URL:**  
```
https://web-production-e592e.up.railway.app/
```

---

## ⚙️ Features

- Check if channels are **currently live** or have **scheduled streams**.
- Fetch detailed video information from YouTube Data API v3 using your **API_KEY**.
- Return performance metrics such as execution time and number of streams found.
- Debug endpoint with raw results for troubleshooting.
- Handles multiple channels and removes duplicates automatically.

---

## 📂 Setup

1. **Environment Variables**: Create a `.env` file with:
```
API_KEY=YOUR_YOUTUBE_API_KEY
CHANNELS_FILE=path/to/channels_with_id.json  # optional, defaults to channels_with_id.json
PORT=8000  # optional
```

2. **Channels file**: `channels_with_id.json` must contain a **list of dictionaries**:
```json
[
  {"handle": "tokinosora", "id": "UCp6993wxpyDPHUpavwDFqgg"},
  {"handle": "robocosan", "id": "UCDqI2jOz0weumE8s7paEk6g"}
]
```

3. **Run the API**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 📌 Endpoints

### `GET /`
Root endpoint, shows API info and available endpoints.

**Response**
```json
{
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
```

---

### `GET /check`
Check all channels for live or upcoming streams and fetch video details from YouTube API.

**Response**
```json
{
  "data": [
    [
      "Streamer Name",
      "Stream Title",
      "https://youtube.com/watch?v=abcd1234",
      {
        "actualStartTime": "2025-08-27T11:45:00Z",
        "scheduledStartTime": "2025-08-27T12:00:00Z",
        "status": "live"
      }
    ]
  ],
  "metrics": {
    "total_execution_time_ms": 3250,
    "channels_checked": 50,
    "average_time_per_channel_ms": 65,
    "live_streams_count": 2,
    "scheduled_streams_count": 1
  }
}
```

---

### `GET /live`
Return only currently live streams.

**Response**
```json
{
  "live_streams": [
    [
      "Streamer Name",
      "Stream Title",
      "https://youtube.com/watch?v=abcd1234",
      {"status": "live"}
    ]
  ],
  "count": 2
}
```

---

### `GET /upcoming`
Return only upcoming streams.

**Response**
```json
{
  "upcoming_streams": [
    [
      "Streamer Name",
      "Scheduled Stream Title",
      "https://youtube.com/watch?v=efgh5678",
      {"status": "upcoming"}
    ]
  ],
  "count": 1
}
```

---

### `GET /metrics`
Return performance metrics from the last check.

**Response**
```json
{
  "total_execution_time_ms": 3250,
  "channels_checked": 50,
  "average_time_per_channel_ms": 65,
  "live_streams_count": 2,
  "scheduled_streams_count": 1
}
```

---

### `GET /debug/results`
Return the **raw results** including all channels checked and any errors.

**Response**
```json
{
  "data": [
    {
      "handle": "tokinosora",
      "channel_id": "UCp6993wxpyDPHUpavwDFqgg",
      "live": true,
      "video_url": "https://youtube.com/watch?v=abcd1234",
      "scheduled": false,
      "error": null
    }
  ],
  "metrics": {
    "total_execution_time_ms": 3250,
    "channels_checked": 50,
    "average_time_per_channel_ms": 65,
    "live_streams_count": 2,
    "scheduled_streams_count": 1
  }
}
```

---

## ⚡ Usage Examples

### Curl
```bash
curl https://web-production-e592e.up.railway.app/live
```

### Python (requests)
```python
import requests

url = "https://web-production-e592e.up.railway.app/upcoming"
response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    print(data)
```

---

## ✅ Notes

- All endpoints use **GET** requests.
- **Timestamps** follow **UTC ISO 8601** format.
- Channels with duplicate video URLs are deduplicated automatically.
- Channels currently in YouTube "waiting rooms" are ignored.
- Handles both HoloEN, HoloJP, and other VTuber channels included in `channels_with_id.json`.
