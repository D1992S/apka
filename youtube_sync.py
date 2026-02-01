"""
YOUTUBE SYNC MODULE
====================
Synchronizacja danych z YouTube API.
- Pobiera listę filmów z kanału
- Pobiera statystyki (views, likes)
- Pobiera retencję (wymaga Analytics API + OAuth)
- Pobiera transkrypty
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Opcjonalne importy
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    TRANSCRIPT_API_AVAILABLE = True
except ImportError:
    TRANSCRIPT_API_AVAILABLE = False


# Ścieżki
CONFIG_DIR = Path("./app_data")
CREDENTIALS_FILE = CONFIG_DIR / "youtube_credentials.json"
TOKEN_FILE = CONFIG_DIR / "youtube_token.pickle"
CHANNEL_DATA_DIR = Path("./channel_data")
SYNCED_DATA_FILE = CHANNEL_DATA_DIR / "synced_channel_data.csv"


# Scopes dla API
SCOPES = [
    'https://www.googleapis.com/auth/youtube.readonly',
    'https://www.googleapis.com/auth/yt-analytics.readonly'
]


class YouTubeSync:
    """Synchronizacja danych z YouTube"""
    
    def __init__(self):
        self.youtube = None
        self.analytics = None
        self.credentials = None
        self._channel_id = None
        self.api_key = None
        self.channel_id = None
        
    def is_available(self) -> bool:
        """Sprawdza czy Google API jest dostępne"""
        return GOOGLE_API_AVAILABLE
    
    def has_credentials(self) -> bool:
        """Sprawdza czy plik credentials istnieje"""
        return CREDENTIALS_FILE.exists()
    
    def is_authenticated(self) -> bool:
        """Sprawdza czy jesteśmy zalogowani"""
        return self.credentials is not None and getattr(self.credentials, 'valid', False)

    def set_api_key(self, api_key: str):
        """Ustawia API key dla publicznych zapytań"""
        self.api_key = api_key or None

    def set_channel_id(self, channel_id: str):
        """Ustawia channel_id dla zapytań publicznych"""
        self.channel_id = (channel_id or "").strip() or None

    def ensure_public_client(self) -> bool:
        """Buduje klienta YouTube Data API z API key (bez OAuth)."""
        if not GOOGLE_API_AVAILABLE:
            return False
        if self.youtube is not None:
            return True
        if not self.api_key:
            return False
        try:
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)
            return True
        except Exception:
            return False
    
    def setup_instructions(self) -> str:
        """Instrukcje konfiguracji"""
        return """
## 🔧 Jak skonfigurować YouTube API:

### Krok 1: Utwórz projekt w Google Cloud
1. Wejdź na: https://console.cloud.google.com/
2. Utwórz nowy projekt (np. "YT-Evaluator")
3. Włącz APIs:
   - YouTube Data API v3
   - YouTube Analytics API

### Krok 2: Utwórz credentials
1. Idź do "APIs & Services" → "Credentials"
2. Kliknij "Create Credentials" → "OAuth client ID"
3. Typ: "Desktop application"
4. Pobierz JSON i zapisz jako `youtube_credentials.json`

### Krok 3: Wgraj plik
1. Skopiuj `youtube_credentials.json` do folderu `app_data/`
2. Kliknij "Zaloguj do YouTube" w aplikacji
3. Zaloguj się przez przeglądarkę

### ⚠️ Uwaga:
- Pierwszy raz będzie prośba o zgodę w przeglądarce
- Zaznacz wszystkie uprawnienia
- Token jest zapisywany lokalnie - nie musisz logować się za każdym razem
"""
    
    def authenticate(self) -> Tuple[bool, str]:
        """
        Autoryzacja OAuth2.
        
        Returns:
            (success, message)
        """
        if not GOOGLE_API_AVAILABLE:
            return False, "Zainstaluj biblioteki: pip install google-api-python-client google-auth-oauthlib"
        
        if not CREDENTIALS_FILE.exists():
            return False, f"Brak pliku credentials. Umieść youtube_credentials.json w folderze {CONFIG_DIR}"
        
        creds = None
        
        # Sprawdź zapisany token
        if TOKEN_FILE.exists():
            try:
                with open(TOKEN_FILE, 'rb') as token:
                    creds = pickle.load(token)
            except (IOError, pickle.PickleError) as e:
                print(f"⚠ Nie udało się wczytać tokenu YouTube: {e}")
        
        # Odśwież lub uzyskaj nowy token
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    creds = None
            
            if not creds:
                try:
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(CREDENTIALS_FILE), SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                except Exception as e:
                    return False, f"Błąd autoryzacji: {e}"
            
            # Zapisz token
            CONFIG_DIR.mkdir(exist_ok=True)
            with open(TOKEN_FILE, 'wb') as token:
                pickle.dump(creds, token)
        
        self.credentials = creds
        
        # Zbuduj serwisy
        try:
            self.youtube = build('youtube', 'v3', credentials=creds)
            self.analytics = build('youtubeAnalytics', 'v2', credentials=creds)
            return True, "✅ Zalogowano pomyślnie!"
        except Exception as e:
            return False, f"Błąd budowania API: {e}"
    
    def get_channel_id(self) -> Optional[str]:
        """Pobiera ID kanału zalogowanego użytkownika"""
        if self.channel_id:
            return self.channel_id
        if not self.youtube:
            return None
        
        if self._channel_id:
            return self._channel_id
        
        try:
            response = self.youtube.channels().list(
                part='id,snippet',
                mine=True
            ).execute()
            
            if response.get('items'):
                self._channel_id = response['items'][0]['id']
                return self._channel_id
        except Exception as e:
            print(f"Błąd pobierania channel ID: {e}")
        
        return None
    
    def get_channel_info(self) -> Optional[Dict]:
        """Pobiera informacje o kanale"""
        if not self.youtube:
            return None
        
        channel_id = self.get_channel_id()
        if not channel_id:
            return None
        
        try:
            response = self.youtube.channels().list(
                part='snippet,statistics,contentDetails',
                id=channel_id
            ).execute()
            
            if response.get('items'):
                item = response['items'][0]
                return {
                    'id': item['id'],
                    'title': item['snippet']['title'],
                    'description': item['snippet'].get('description', ''),
                    'subscribers': int(item['statistics'].get('subscriberCount', 0)),
                    'total_views': int(item['statistics'].get('viewCount', 0)),
                    'video_count': int(item['statistics'].get('videoCount', 0)),
                    'uploads_playlist': item['contentDetails']['relatedPlaylists']['uploads']
                }
        except Exception as e:
            print(f"Błąd pobierania info o kanale: {e}")
        
        return None
    
    def get_all_videos(self, max_results: int = 500) -> List[Dict]:
        """
        Pobiera listę wszystkich filmów z kanału.
        
        Args:
            max_results: Maksymalna liczba filmów
            
        Returns:
            Lista słowników z danymi filmów
        """
        if not self.youtube:
            return []
        
        channel_info = self.get_channel_info()
        if not channel_info:
            return []
        
        uploads_playlist = channel_info.get('uploads_playlist')
        if not uploads_playlist:
            return []
        
        videos = []
        next_page_token = None
        
        while len(videos) < max_results:
            try:
                response = self.youtube.playlistItems().list(
                    part='snippet,contentDetails',
                    playlistId=uploads_playlist,
                    maxResults=min(50, max_results - len(videos)),
                    pageToken=next_page_token
                ).execute()
                
                for item in response.get('items', []):
                    video_id = item['contentDetails']['videoId']
                    snippet = item['snippet']
                    
                    videos.append({
                        'video_id': video_id,
                        'title': snippet.get('title', ''),
                        'description': snippet.get('description', '')[:500],
                        'published_at': snippet.get('publishedAt', ''),
                        'thumbnail': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
                    })
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            except Exception as e:
                print(f"Błąd pobierania filmów: {e}")
                break
        
        return videos
    
    def get_video_statistics(self, video_ids: List[str]) -> Dict[str, Dict]:
        """
        Pobiera statystyki dla listy filmów.
        
        Args:
            video_ids: Lista ID filmów
            
        Returns:
            Dict: {video_id: {views, likes, comments}}
        """
        if not self.youtube:
            return {}
        
        stats = {}
        
        # API przyjmuje max 50 ID na raz
        for i in range(0, len(video_ids), 50):
            batch = video_ids[i:i+50]
            
            try:
                response = self.youtube.videos().list(
                    part='statistics,contentDetails',
                    id=','.join(batch)
                ).execute()
                
                for item in response.get('items', []):
                    vid = item['id']
                    s = item.get('statistics', {})
                    
                    # Parsuj duration
                    duration = item.get('contentDetails', {}).get('duration', 'PT0S')
                    duration_seconds = self._parse_duration(duration)
                    
                    stats[vid] = {
                        'views': int(s.get('viewCount', 0)),
                        'likes': int(s.get('likeCount', 0)),
                        'comments': int(s.get('commentCount', 0)),
                        'duration_seconds': duration_seconds,
                    }
                    
            except Exception as e:
                print(f"Błąd pobierania statystyk: {e}")
        
        return stats
    
    def _parse_duration(self, duration: str) -> int:
        """Parsuje ISO 8601 duration na sekundy"""
        import re
        
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration)
        
        if not match:
            return 0
        
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        
        return hours * 3600 + minutes * 60 + seconds
    
    def get_video_analytics(self, video_id: str, start_date: str = None, end_date: str = None) -> Optional[Dict]:
        """
        Pobiera analytics dla filmu (retencja, AVD).
        
        ⚠️ Wymaga YouTube Analytics API
        """
        if not self.analytics:
            return None
        
        channel_id = self.get_channel_id()
        if not channel_id:
            return None
        
        if not start_date:
            start_date = "2020-01-01"
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            response = self.analytics.reports().query(
                ids=f'channel=={channel_id}',
                startDate=start_date,
                endDate=end_date,
                metrics='averageViewDuration,averageViewPercentage,views',
                dimensions='video',
                filters=f'video=={video_id}'
            ).execute()
            
            rows = response.get('rows', [])
            if rows:
                row = rows[0]
                return {
                    'video_id': row[0],
                    'avg_view_duration': row[1],
                    'avg_view_percentage': row[2],  # To jest retencja!
                    'views': row[3],
                }
                
        except Exception as e:
            print(f"Błąd pobierania analytics dla {video_id}: {e}")
        
        return None
    
    def get_bulk_analytics(self, video_ids: List[str] = None, days: int = 7) -> Dict[str, Dict]:
        """
        Pobiera analytics dla wielu filmów.
        
        Args:
            video_ids: Lista ID (None = wszystkie)
            days: Ile dni wstecz (0 = lifetime)
        """
        if not self.analytics:
            return {}
        
        channel_id = self.get_channel_id()
        if not channel_id:
            return {}
        
        if days > 0:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        else:
            start_date = "2020-01-01"
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            response = self.analytics.reports().query(
                ids=f'channel=={channel_id}',
                startDate=start_date,
                endDate=end_date,
                metrics='averageViewDuration,averageViewPercentage,views',
                dimensions='video',
                maxResults=500,
                sort='-views'
            ).execute()
            
            analytics = {}
            for row in response.get('rows', []):
                vid = row[0]
                
                # Filtruj jeśli podano listę
                if video_ids and vid not in video_ids:
                    continue
                
                analytics[vid] = {
                    'avg_view_duration': row[1],
                    'retention': row[2],  # avg_view_percentage
                    'views_period': row[3],
                }
            
            return analytics
            
        except Exception as e:
            print(f"Błąd pobierania bulk analytics: {e}")
            return {}
    
    def get_transcript(self, video_id: str, language: str = 'pl') -> Optional[str]:
        """Pobiera transkrypt filmu"""
        if not TRANSCRIPT_API_AVAILABLE:
            return None
        
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=[language, 'en']
            )
            
            # Złącz w tekst
            full_text = ' '.join([t['text'] for t in transcript_list])
            return full_text
            
        except Exception as e:
            print(f"Błąd pobierania transkryptu dla {video_id}: {e}")
            return None
    
    def get_hook(self, video_id: str, seconds: int = 120) -> Optional[str]:
        """Pobiera hook (pierwsze X sekund) transkryptu"""
        if not TRANSCRIPT_API_AVAILABLE:
            return None
        
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id, 
                languages=['pl', 'en']
            )
            
            # Zbierz tekst do X sekund
            hook_parts = []
            for t in transcript_list:
                if t['start'] <= seconds:
                    hook_parts.append(t['text'])
                else:
                    break
            
            return ' '.join(hook_parts)
            
        except Exception as e:
            return None
    
    def sync_all(self, include_analytics: bool = True, include_transcripts: bool = False,
                 progress_callback=None) -> Tuple[pd.DataFrame, str]:
        """
        Pełna synchronizacja danych kanału.
        
        Args:
            include_analytics: Czy pobierać retencję (wolniejsze)
            include_transcripts: Czy pobierać transkrypty (jeszcze wolniejsze)
            progress_callback: Funkcja do raportowania postępu
            
        Returns:
            (DataFrame z danymi, status message)
        """
        if not self.youtube:
            return pd.DataFrame(), "❌ Nie zalogowano do YouTube API"
        
        def report(msg):
            if progress_callback:
                progress_callback(msg)
            print(msg)
        
        # 1. Pobierz listę filmów
        report("📥 Pobieram listę filmów...")
        videos = self.get_all_videos()
        
        if not videos:
            return pd.DataFrame(), "❌ Nie znaleziono filmów na kanale"
        
        report(f"✅ Znaleziono {len(videos)} filmów")
        
        # 2. Pobierz statystyki
        report("📊 Pobieram statystyki...")
        video_ids = [v['video_id'] for v in videos]
        stats = self.get_video_statistics(video_ids)
        
        # 3. Pobierz analytics (retencja)
        analytics = {}
        if include_analytics:
            report("📈 Pobieram analytics (retencja)...")
            analytics = self.get_bulk_analytics()
            report(f"✅ Pobrano analytics dla {len(analytics)} filmów")
        
        # 4. Złącz dane
        report("🔄 Łączę dane...")
        data = []
        
        for video in videos:
            vid = video['video_id']
            s = stats.get(vid, {})
            a = analytics.get(vid, {})
            
            row = {
                'video_id': vid,
                'title': video['title'],
                'description': video['description'],
                'published_at': video['published_at'],
                'views': s.get('views', 0),
                'likes': s.get('likes', 0),
                'comments': s.get('comments', 0),
                'duration_seconds': s.get('duration_seconds', 0),
                'retention': a.get('retention', None),
                'avg_view_duration': a.get('avg_view_duration', None),
            }
            
            # 5. Transkrypty (opcjonalnie)
            if include_transcripts:
                report(f"📝 Pobieram transkrypt: {video['title'][:30]}...")
                row['transcript'] = self.get_transcript(vid)
                row['hook_120s'] = self.get_hook(vid, 120)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # 6. Dodaj auto-labels
        if 'views' in df.columns and 'retention' in df.columns:
            report("🏷️ Generuję auto-labels...")
            df['label'] = df.apply(self._auto_label, axis=1)
        
        # 7. Zapisz
        CHANNEL_DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(SYNCED_DATA_FILE, index=False)
        report(f"💾 Zapisano do {SYNCED_DATA_FILE}")
        
        return df, f"✅ Zsynchronizowano {len(df)} filmów!"
    
    def _auto_label(self, row) -> str:
        """Auto-labelowanie na podstawie views i retention"""
        views = row.get('views', 0) or 0
        retention = row.get('retention', 0) or 0
        
        # PASS: views >= 50k LUB retention >= 45%
        if views >= 50000 or retention >= 45:
            return "PASS"
        
        # FAIL: views < 15k AND retention < 25%
        if views < 15000 and retention < 25:
            return "FAIL"
        
        # BORDER: reszta
        return "BORDER"
    
    def get_last_sync_time(self) -> Optional[str]:
        """Zwraca czas ostatniej synchronizacji"""
        if SYNCED_DATA_FILE.exists():
            mtime = SYNCED_DATA_FILE.stat().st_mtime
            return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        return None
    
    def load_synced_data(self) -> Optional[pd.DataFrame]:
        """Ładuje zsynchronizowane dane"""
        if SYNCED_DATA_FILE.exists():
            return pd.read_csv(SYNCED_DATA_FILE)
        return None


# =============================================================================
# COMPETITOR ANALYSIS
# =============================================================================

class CompetitorAnalyzer:
    """Analiza kanałów konkurencji"""
    
    DARK_DOC_CHANNELS = [
        # Polskie
        "UCX6OQ3DkcsbYNE6H8uQQuVA",  # MrWhoseTheBoss (przykład)
        # Dodaj ID kanałów dark doc
    ]
    
    def __init__(self, youtube_service):
        self.youtube = youtube_service
    
    def find_similar_channels(self, keywords: List[str] = None) -> List[Dict]:
        """
        Znajduje podobne kanały w niszy.
        """
        if not self.youtube:
            return []
        
        if not keywords:
            keywords = ["dark documentary", "mroczny dokument", "tajemnice", "zbrodnie dokumentalne"]
        
        channels = []
        
        for keyword in keywords[:3]:  # Limit queries
            try:
                response = self.youtube.search().list(
                    part='snippet',
                    q=keyword,
                    type='channel',
                    maxResults=10,
                    relevanceLanguage='pl'
                ).execute()
                
                for item in response.get('items', []):
                    channel_id = item['snippet']['channelId']
                    
                    # Pobierz statystyki kanału
                    ch_response = self.youtube.channels().list(
                        part='statistics,snippet',
                        id=channel_id
                    ).execute()
                    
                    if ch_response.get('items'):
                        ch = ch_response['items'][0]
                        channels.append({
                            'id': channel_id,
                            'title': ch['snippet']['title'],
                            'subscribers': int(ch['statistics'].get('subscriberCount', 0)),
                            'views': int(ch['statistics'].get('viewCount', 0)),
                            'videos': int(ch['statistics'].get('videoCount', 0)),
                        })
                        
            except Exception as e:
                print(f"Błąd wyszukiwania kanałów: {e}")
        
        # Usuń duplikaty
        seen = set()
        unique = []
        for ch in channels:
            if ch['id'] not in seen:
                seen.add(ch['id'])
                unique.append(ch)
        
        return sorted(unique, key=lambda x: x['subscribers'], reverse=True)
    
    def analyze_competitor(self, channel_id: str, max_videos: int = 50) -> Dict:
        """
        Analizuje kanał konkurencji.
        """
        if not self.youtube:
            return {}
        
        try:
            # Info o kanale
            ch_response = self.youtube.channels().list(
                part='statistics,snippet,contentDetails',
                id=channel_id
            ).execute()
            
            if not ch_response.get('items'):
                return {'error': 'Kanał nie znaleziony'}
            
            channel = ch_response['items'][0]
            uploads_playlist = channel['contentDetails']['relatedPlaylists']['uploads']
            
            # Pobierz filmy
            videos = []
            next_page = None
            
            while len(videos) < max_videos:
                pl_response = self.youtube.playlistItems().list(
                    part='snippet',
                    playlistId=uploads_playlist,
                    maxResults=min(50, max_videos - len(videos)),
                    pageToken=next_page
                ).execute()
                
                video_ids = [item['snippet']['resourceId']['videoId'] 
                            for item in pl_response.get('items', [])]
                
                # Pobierz statystyki
                if video_ids:
                    stats_response = self.youtube.videos().list(
                        part='statistics,snippet',
                        id=','.join(video_ids)
                    ).execute()
                    
                    for item in stats_response.get('items', []):
                        videos.append({
                            'title': item['snippet']['title'],
                            'views': int(item['statistics'].get('viewCount', 0)),
                            'published': item['snippet']['publishedAt'],
                        })
                
                next_page = pl_response.get('nextPageToken')
                if not next_page:
                    break
            
            # Analiza
            if videos:
                views = [v['views'] for v in videos]
                
                return {
                    'channel_name': channel['snippet']['title'],
                    'subscribers': int(channel['statistics'].get('subscriberCount', 0)),
                    'total_videos': len(videos),
                    'avg_views': sum(views) // len(views),
                    'median_views': sorted(views)[len(views)//2],
                    'max_views': max(views),
                    'top_videos': sorted(videos, key=lambda x: x['views'], reverse=True)[:5],
                    'recent_videos': videos[:10],
                }
            
        except Exception as e:
            return {'error': str(e)}
        
        return {}


# ==========================================================================
# ANALYTICS: TIMESERIES (opcjonalne)
# ==========================================================================
def get_video_daily_timeseries(
    self,
    video_id: str,
    days: int = 30,
    metrics: str = "views,estimatedMinutesWatched,averageViewDuration",
    start_date: str = None,
    end_date: str = None
) -> Dict:
    """
    Pobiera dzienne metryki z YouTube Analytics API dla pojedynczego filmu.

    Wymaga:
    - OAuth credentials
    - YouTube Analytics API włączone w projekcie
    """
    if not self.analytics:
        return {"error": "Brak klienta YouTube Analytics. Zaloguj się w zakładce Dane."}
    if not video_id:
        return {"error": "Brak video_id"}

    try:
        if end_date is None:
            end_date = datetime.now().date().isoformat()
        if start_date is None:
            start_date = (datetime.now().date() - timedelta(days=days)).isoformat()

        resp = self.analytics.reports().query(
            ids="channel==MINE",
            startDate=start_date,
            endDate=end_date,
            metrics=metrics,
            dimensions="day",
            filters=f"video=={video_id}",
            sort="day"
        ).execute()

        cols = [c.get("name") for c in resp.get("columnHeaders", [])]
        rows = resp.get("rows", []) or []
        data = [dict(zip(cols, r)) for r in rows]

        return {
            "status": "OK",
            "video_id": video_id,
            "startDate": start_date,
            "endDate": end_date,
            "metrics": metrics,
            "rows": data
        }
    except Exception as e:
        return {"error": str(e), "video_id": video_id}

def get_channel_daily_timeseries(
    self,
    days: int = 60,
    metrics: str = "views,estimatedMinutesWatched,averageViewDuration",
    start_date: str = None,
    end_date: str = None
) -> Dict:
    """Dzienne metryki kanału (YouTube Analytics API)."""
    if not self.analytics:
        return {"error": "Brak klienta YouTube Analytics. Zaloguj się w zakładce Dane."}
    try:
        if end_date is None:
            end_date = datetime.now().date().isoformat()
        if start_date is None:
            start_date = (datetime.now().date() - timedelta(days=days)).isoformat()

        resp = self.analytics.reports().query(
            ids="channel==MINE",
            startDate=start_date,
            endDate=end_date,
            metrics=metrics,
            dimensions="day",
            sort="day"
        ).execute()
        cols = [c.get("name") for c in resp.get("columnHeaders", [])]
        rows = resp.get("rows", []) or []
        data = [dict(zip(cols, r)) for r in rows]

        return {
            "status": "OK",
            "startDate": start_date,
            "endDate": end_date,
            "metrics": metrics,
            "rows": data
        }
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# SINGLETON
# =============================================================================

_youtube_sync = None

def get_youtube_sync() -> YouTubeSync:
    global _youtube_sync
    if _youtube_sync is None:
        _youtube_sync = YouTubeSync()
    return _youtube_sync
