# 🎥 YouTube Live Checker API

A simple API to check multiple YouTube channels for **live** and **upcoming** streams.  
Deployed on [Railway](https://railway.app).

**Base URL:**  
```
https://web-production-e592e.up.railway.app/
```

---

## 📌 Endpoints

### `GET /check`
Check all channels for both live and upcoming streams.

**Example Response**
```json
{
  "checked": 15,
  "live": 2,
  "upcoming": 1,
  "timestamp": "2025-08-27T12:00:00Z"
}
```

---

### `GET /live`
Get only currently live streams.

**Example Response**
```json
[
  {
    "channel_id": "UC123456789",
    "channel_name": "Streamer One",
    "title": "🔴 Live Now!",
    "url": "https://youtube.com/watch?v=abcd1234",
    "start_time": "2025-08-27T11:45:00Z"
  }
]
```

---

### `GET /upcoming`
Get only upcoming scheduled streams.

**Example Response**
```json
[
  {
    "channel_id": "UC987654321",
    "channel_name": "Streamer Two",
    "title": "Big Stream Coming Soon",
    "url": "https://youtube.com/watch?v=efgh5678",
    "scheduled_time": "2025-08-27T14:00:00Z"
  }
]
```

---

### `GET /metrics`
Get performance metrics from the last check.

**Example Response**
```json
{
  "last_check_duration": "3.24s",
  "channels_checked": 15,
  "live_found": 2,
  "upcoming_found": 1,
  "errors": 0
}
```

---

### `GET /debug/results`
Get raw results from the last API call (for debugging).

**Example Response**
```json
{
  "raw": [
    {
      "channel_id": "UC123456789",
      "status": "live",
      "video_id": "abcd1234"
    },
    {
      "channel_id": "UC987654321",
      "status": "upcoming",
      "video_id": "efgh5678"
    }
  ]
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

url = "https://web-production-e592e.up.railway.app/live"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    print(data)
```

---

## ✅ Notes
- All endpoints are `GET` requests.
- Timestamps follow **UTC ISO 8601** format.
- Useful for monitoring multiple YouTube channels automatically.
