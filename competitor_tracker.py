"""
COMPETITOR TRACKER
==================
Śledzi ostatnie uploady kanałów konkurencji.

Tryby:
- YouTube Data API (jeśli google-api-python-client dostępne i masz credentials)
- fallback: youtube-search-python (search-based, mniej dokładne)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional

try:
    from googleapiclient.discovery import build
    GOOGLE_API_AVAILABLE = True
except Exception:
    GOOGLE_API_AVAILABLE = False

try:
    from youtubesearchpython import VideosSearch
    YT_SEARCH_AVAILABLE = True
except Exception:
    YT_SEARCH_AVAILABLE = False


class CompetitorTracker:
    def __init__(self, youtube_api_client=None):
        self.youtube = youtube_api_client

    @staticmethod
    def _iso(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def fetch_recent_uploads(
        self,
        competitors: List[Dict[str, Any]],
        days: int = 14,
        max_per_channel: int = 8,
        region: str = "PL",
        language: str = "pl"
    ) -> List[Dict[str, Any]]:
        """
        Zwraca listę ostatnich filmów konkurencji.

        competitors: [{"channel_id": "...", "name": "..."}]
        """
        since = datetime.now(timezone.utc) - timedelta(days=days)
        out: List[Dict[str, Any]] = []

        # 1) YouTube Data API, jeśli dostępny
        if self.youtube is not None:
            for c in competitors:
                ch_id = (c.get("channel_id") or "").strip()
                if not ch_id:
                    continue
                try:
                    # find uploads playlist
                    ch = self.youtube.channels().list(
                        part="contentDetails,snippet",
                        id=ch_id,
                        maxResults=1
                    ).execute()
                    items = ch.get("items", [])
                    if not items:
                        continue
                    uploads_pl = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
                    ch_name = items[0]["snippet"].get("title") or c.get("name") or ch_id

                    pl_resp = self.youtube.playlistItems().list(
                        part="snippet,contentDetails",
                        playlistId=uploads_pl,
                        maxResults=max_per_channel
                    ).execute()
                    for it in pl_resp.get("items", []):
                        sn = it.get("snippet", {})
                        cd = it.get("contentDetails", {})
                        pub = sn.get("publishedAt") or cd.get("videoPublishedAt")
                        if not pub:
                            continue
                        try:
                            pub_dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                        except Exception:
                            pub_dt = None
                        if pub_dt and pub_dt < since:
                            continue
                        out.append({
                            "source": "youtube_api",
                            "competitor_id": c.get("id", ""),
                            "channel_id": ch_id,
                            "channel_name": ch_name,
                            "video_id": cd.get("videoId", ""),
                            "title": sn.get("title", ""),
                            "publishedAt": pub,
                            "url": f"https://www.youtube.com/watch?v={cd.get('videoId','')}",
                            "description": sn.get("description", "")[:4000]
                        })
                except Exception as e:
                    out.append({
                        "source": "youtube_api",
                        "competitor_id": c.get("id", ""),
                        "channel_id": ch_id,
                        "channel_name": c.get("name", ch_id),
                        "error": str(e)
                    })
            return out

        # 2) Fallback: youtube-search-python
        if not YT_SEARCH_AVAILABLE:
            return [{"error": "Brak YouTube API i brak youtube-search-python. Zainstaluj youtube-search-python lub włącz YouTube API."}]

        for c in competitors:
            name = (c.get("name") or "").strip()
            ch_id = (c.get("channel_id") or "").strip()
            query = name or ch_id
            if not query:
                continue
            try:
                vs = VideosSearch(query, limit=max_per_channel, region=region, language=language)
                res = vs.result()
                for v in res.get("result", []):
                    out.append({
                        "source": "search",
                        "competitor_id": c.get("id", ""),
                        "channel_id": ch_id,
                        "channel_name": v.get("channel", {}).get("name", name or ch_id),
                        "video_id": v.get("id", ""),
                        "title": v.get("title", ""),
                        "publishedTime": v.get("publishedTime", ""),
                        "duration": v.get("duration", ""),
                        "views": v.get("viewCount", {}).get("text", ""),
                        "url": v.get("link", "")
                    })
            except TypeError as e:
                if "proxies" in str(e):
                    out.append({
                        "error": (
                            "Błąd kompatybilności youtube-search-python z httpx. "
                            "Zainstaluj: pip install httpx==0.24.1"
                        ),
                        "competitor_id": c.get("id", ""),
                        "query": query
                    })
                else:
                    raise
            except Exception as e:
                out.append({"error": str(e), "competitor_id": c.get("id",""), "query": query})

        return out


def get_competitor_tracker(youtube_api_client=None) -> CompetitorTracker:
    return CompetitorTracker(youtube_api_client=youtube_api_client)
