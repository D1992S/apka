"""
EXTERNAL SOURCES MODULE (NOWY dla v4)
======================================
Zewnętrzne źródła danych:
- Wikipedia API (popularność tematu)
- Google News RSS (świeżość tematu)
- Sezonowość (kiedy publikować)
- Trend Discovery (co trenduje teraz)
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List
import requests


class WikipediaAPI:
    """Wikipedia pageviews i info o temacie"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'YT-Evaluator/1.0 (Educational Project)'
    
    def get_topic_popularity(self, topic: str, days: int = 30) -> Dict:
        """
        Pobiera popularność tematu na Wikipedii.
        
        Returns:
            Dict z pageviews, trend, wikipedia_score
        """
        title = topic.replace(' ', '_')
        
        end = datetime.now()
        start = end - timedelta(days=days)
        
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/pl.wikipedia/all-access/all-agents/{title}/daily/{start.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}"
        
        result = {
            'topic': topic,
            'total_pageviews_30d': 0,
            'avg_daily_views': 0,
            'trend': 'UNKNOWN',
            'wikipedia_score': 0,
            'peak_date': None,
            'article_exists': False,
        }
        
        try:
            resp = self.session.get(url, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                views = [item['views'] for item in data.get('items', [])]
                
                if views and len(views) > 0:
                    result['article_exists'] = True
                    result['total_pageviews_30d'] = sum(views)
                    result['avg_daily_views'] = sum(views) // len(views)

                    # Peak - bezpieczne znajdowanie maksimum
                    max_views = max(views)
                    try:
                        max_idx = views.index(max_views)
                        if data.get('items') and max_idx < len(data['items']):
                            result['peak_date'] = data['items'][max_idx].get('timestamp', '')[:8]
                    except (ValueError, IndexError):
                        pass  # Nie udało się znaleźć peak date
                    
                    # Trend
                    if len(views) >= 7:
                        first_week = sum(views[:7])
                        last_week = sum(views[-7:])
                        
                        if last_week > first_week * 1.3:
                            result['trend'] = 'RISING'
                        elif last_week < first_week * 0.7:
                            result['trend'] = 'FALLING'
                        else:
                            result['trend'] = 'STABLE'
                    
                    # Score based on pageviews
                    total = result['total_pageviews_30d']
                    if total >= 100000:
                        result['wikipedia_score'] = 100
                    elif total >= 50000:
                        result['wikipedia_score'] = 80
                    elif total >= 20000:
                        result['wikipedia_score'] = 60
                    elif total >= 5000:
                        result['wikipedia_score'] = 40
                    elif total >= 1000:
                        result['wikipedia_score'] = 20
                    else:
                        result['wikipedia_score'] = 10
                        
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def search_articles(self, query: str, limit: int = 5) -> List[Dict]:
        """Wyszukuje artykuły na Wikipedii"""
        url = "https://pl.wikipedia.org/w/api.php"
        params = {
            'action': 'opensearch',
            'search': query,
            'limit': limit,
            'format': 'json'
        }
        
        results = []
        try:
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                titles = data[1] if len(data) > 1 else []
                descriptions = data[2] if len(data) > 2 else []
                urls = data[3] if len(data) > 3 else []
                
                for i, title in enumerate(titles):
                    results.append({
                        'title': title,
                        'description': descriptions[i] if i < len(descriptions) else '',
                        'url': urls[i] if i < len(urls) else '',
                    })
        except Exception as e:
            print(f"⚠ Błąd wyszukiwania Wikipedia: {e}")

        return results


class NewsChecker:
    """Sprawdza obecność tematu w newsach"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    
    def get_news_score(self, topic: str) -> Dict:
        """
        Sprawdza Google News RSS dla tematu.
        
        Returns:
            Dict z news_score, recent_headlines, recommendation
        """
        result = {
            'topic': topic,
            'news_score': 0,
            'has_recent_news': False,
            'estimated_articles': 0,
            'recent_headlines': [],
            'recommendation': '',
        }
        
        try:
            # Google News RSS
            url = f"https://news.google.com/rss/search?q={topic}&hl=pl&gl=PL&ceid=PL:pl"
            resp = self.session.get(url, timeout=10)
            
            if resp.status_code == 200:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(resp.content)
                items = root.findall('.//item')
                
                result['estimated_articles'] = len(items)
                result['has_recent_news'] = len(items) > 0
                
                # Get headlines
                for item in items[:5]:
                    title = item.find('title')
                    pub_date = item.find('pubDate')
                    if title is not None:
                        result['recent_headlines'].append({
                            'title': title.text,
                            'date': pub_date.text if pub_date is not None else '',
                        })
                
                # Score
                count = len(items)
                if count >= 10:
                    result['news_score'] = 100
                    result['recommendation'] = "🔥 HOT NEWS - temat jest bardzo świeży! Idealne okno czasowe."
                elif count >= 5:
                    result['news_score'] = 70
                    result['recommendation'] = "📰 Temat obecny w mediach - umiarkowane zainteresowanie."
                elif count >= 2:
                    result['news_score'] = 40
                    result['recommendation'] = "📅 Sporadyczne newsy - temat może być evergreen."
                elif count >= 1:
                    result['news_score'] = 20
                    result['recommendation'] = "🗄️ Pojedyncze wzmianki - temat historyczny."
                else:
                    result['news_score'] = 5
                    result['recommendation'] = "📦 Brak newsów - temat archiwalny, mniejsza konkurencja."
                    
        except Exception as e:
            result['error'] = str(e)
            result['recommendation'] = "Nie udało się sprawdzić newsów."
        
        return result


class SeasonalityAnalyzer:
    """Analizuje sezonowość tematów"""
    
    # Znane wzorce sezonowe dla dark doc
    PATTERNS = {
        # Katastrofy i rocznice
        'titanic': (4, 'Rocznica zatonięcia (15 kwietnia 1912)'),
        'smoleńsk': (4, 'Rocznica katastrofy (10 kwietnia 2010)'),
        'smolensk': (4, 'Rocznica katastrofy (10 kwietnia 2010)'),
        'czarnobyl': (4, 'Rocznica katastrofy (26 kwietnia 1986)'),
        'chernobyl': (4, 'Rocznica katastrofy (26 kwietnia 1986)'),
        'fukushima': (3, 'Rocznica katastrofy (11 marca 2011)'),
        
        # WWII
        'ww2': (9, 'Rocznica wybuchu II wojny światowej (1 września)'),
        'wwii': (9, 'Rocznica wybuchu II wojny światowej'),
        'ii wojna': (9, 'Rocznica wybuchu wojny'),
        'powstanie warszawskie': (8, 'Rocznica wybuchu (1 sierpnia 1944)'),
        
        # Holocaust
        'holocaust': (1, 'Dzień Pamięci Ofiar Holokaustu (27 stycznia)'),
        'holokaust': (1, 'Dzień Pamięci Ofiar Holokaustu (27 stycznia)'),
        'auschwitz': (1, 'Rocznica wyzwolenia (27 stycznia 1945)'),
        'oświęcim': (1, 'Rocznica wyzwolenia'),
        
        # 9/11
        '11 września': (9, 'Rocznica ataków (11 września 2001)'),
        '9/11': (9, 'Rocznica ataków'),
        'world trade center': (9, 'Rocznica ataków'),
        'wtc': (9, 'Rocznica ataków'),
        
        # Inne
        'jonestown': (11, 'Rocznica masakry (18 listopada 1978)'),
        'pearl harbor': (12, 'Rocznica ataku (7 grudnia 1941)'),
        'diana': (8, 'Rocznica śmierci (31 sierpnia 1997)'),
    }
    
    MONTHS_PL = {
        1: 'Styczeń', 2: 'Luty', 3: 'Marzec', 4: 'Kwiecień',
        5: 'Maj', 6: 'Czerwiec', 7: 'Lipiec', 8: 'Sierpień',
        9: 'Wrzesień', 10: 'Październik', 11: 'Listopad', 12: 'Grudzień'
    }
    
    def analyze_topic_seasonality(self, topic: str) -> Dict:
        """
        Analizuje sezonowość tematu.
        
        Returns:
            Dict z has_seasonality, peak_month, current_relevance, recommendation
        """
        result = {
            'topic': topic,
            'has_seasonality': False,
            'peak_month': None,
            'peak_month_name': None,
            'reason': None,
            'current_relevance': 'NORMAL',
            'months_until_peak': None,
            'recommendation': '🌐 Temat evergreen - brak wyraźnej sezonowości. Publikuj kiedy chcesz.',
        }
        
        topic_lower = topic.lower()
        
        for pattern, (month, reason) in self.PATTERNS.items():
            if pattern in topic_lower:
                result['has_seasonality'] = True
                result['peak_month'] = month
                result['peak_month_name'] = self.MONTHS_PL[month]
                result['reason'] = reason
                
                current_month = datetime.now().month
                
                # Calculate months until peak
                if current_month <= month:
                    months_until = month - current_month
                else:
                    months_until = (12 - current_month) + month
                
                result['months_until_peak'] = months_until
                
                # Determine relevance
                if current_month == month:
                    result['current_relevance'] = 'PEAK'
                    result['recommendation'] = f"🔥 IDEALNY MOMENT! To jest miesiąc peak dla tego tematu ({result['peak_month_name']}). Publikuj JAK NAJSZYBCIEJ!"
                elif months_until == 1 or (month == 1 and current_month == 12):
                    result['current_relevance'] = 'PRE_PEAK'
                    result['recommendation'] = f"⏰ Miesiąc przed peak! Przygotuj film na {result['peak_month_name']}."
                elif months_until == 11 or current_month == month + 1:
                    result['current_relevance'] = 'POST_PEAK'
                    result['recommendation'] = f"⚠️ Zaraz po peak - zainteresowanie spada. Rozważ poczekanie do następnego roku."
                else:
                    result['current_relevance'] = 'OFF_SEASON'
                    result['recommendation'] = f"📅 Peak za {months_until} miesięcy ({result['peak_month_name']}). Zaplanuj produkcję z wyprzedzeniem!"
                
                break
        
        return result
    
    def get_seasonal_topics_for_month(self, month: int = None) -> List[Dict]:
        """Zwraca tematy które mają peak w danym miesiącu"""
        if month is None:
            month = datetime.now().month
        
        topics = []
        for pattern, (peak_month, reason) in self.PATTERNS.items():
            if peak_month == month:
                topics.append({
                    'topic': pattern.title(),
                    'reason': reason,
                    'pattern': pattern,
                })
        
        return topics
    
    def get_upcoming_peaks(self, months_ahead: int = 3) -> List[Dict]:
        """Zwraca tematy z peak w najbliższych miesiącach"""
        current_month = datetime.now().month
        upcoming = []
        
        for i in range(months_ahead):
            check_month = ((current_month - 1 + i) % 12) + 1
            topics = self.get_seasonal_topics_for_month(check_month)
            
            for t in topics:
                t['month'] = check_month
                t['month_name'] = self.MONTHS_PL[check_month]
                t['months_away'] = i
                upcoming.append(t)
        
        return upcoming


class TrendDiscovery:
    """Aktywnie szuka trendujących tematów"""
    
    DARK_DOC_KEYWORDS = [
        "zbrodnia", "morderstwo", "zaginięcie", "katastrofa", "skandal",
        "afera", "śledztwo", "tajemnica", "sekta", "spisek", "tragedia",
        "wypadek", "oszustwo", "mafia", "korupcja", "śmierć", "polska"
    ]
    
    def __init__(self):
        self.wiki = WikipediaAPI()
        self.news = NewsChecker()
        self.seasonality = SeasonalityAnalyzer()
    
    def discover_trending(self, keywords: List[str] = None) -> List[Dict]:
        """
        Znajduje aktualnie trendujące tematy.
        
        Returns:
            Lista trendujących tematów z źródłem i score
        """
        if keywords is None:
            keywords = self.DARK_DOC_KEYWORDS
        
        trending = []
        
        # Check news for each keyword
        for keyword in keywords[:8]:  # Limit for speed
            news = self.news.get_news_score(keyword)
            
            if news.get('news_score', 0) >= 40:
                trending.append({
                    'topic': keyword.title(),
                    'source': 'news',
                    'score': news['news_score'],
                    'headlines': [h['title'] for h in news.get('recent_headlines', [])[:2]],
                    'recommendation': news.get('recommendation', ''),
                })
        
        # Check seasonality
        current_month = datetime.now().month
        seasonal = self.seasonality.get_seasonal_topics_for_month(current_month)
        
        for topic in seasonal:
            trending.append({
                'topic': topic['topic'],
                'source': 'seasonality',
                'score': 85,
                'reason': topic['reason'],
                'recommendation': '📅 Sezonowy peak właśnie teraz!',
            })
        
        # Check upcoming peaks
        upcoming = self.seasonality.get_upcoming_peaks(months_ahead=2)
        for topic in upcoming:
            if topic['months_away'] == 1:  # Next month
                trending.append({
                    'topic': topic['topic'],
                    'source': 'upcoming_seasonality',
                    'score': 70,
                    'reason': topic['reason'],
                    'recommendation': f"📆 Peak za miesiąc ({topic['month_name']}) - zacznij przygotowania!",
                })
        
        # Sort by score
        trending = sorted(trending, key=lambda x: x['score'], reverse=True)
        
        return trending
    
    def analyze_topic_complete(self, topic: str) -> Dict:
        """
        Kompletna analiza tematu ze wszystkich źródeł.
        """
        # Wikipedia
        wiki_data = self.wiki.get_topic_popularity(topic)
        
        # News
        news_data = self.news.get_news_score(topic)
        
        # Seasonality
        season_data = self.seasonality.analyze_topic_seasonality(topic)
        
        # Calculate combined score
        wiki_score = wiki_data.get('wikipedia_score', 0) * 0.25
        news_score = news_data.get('news_score', 0) * 0.35
        season_bonus = 25 if season_data.get('current_relevance') == 'PEAK' else 10 if season_data.get('current_relevance') == 'PRE_PEAK' else 0
        
        total_score = wiki_score + news_score + season_bonus
        
        return {
            'topic': topic,
            'total_score': round(total_score, 1),
            'wikipedia': wiki_data,
            'news': news_data,
            'seasonality': season_data,
            'verdict': self._get_verdict(total_score, season_data),
        }
    
    def _get_verdict(self, score: float, season_data: Dict) -> str:
        """Generuje werdykt"""
        if season_data.get('current_relevance') == 'PEAK':
            return "🔥 PUBLIKUJ TERAZ - sezonowy peak!"
        elif score >= 70:
            return "🟢 Świetny moment na ten temat"
        elif score >= 40:
            return "🟡 Dobry temat, umiarkowane zainteresowanie"
        else:
            return "🔵 Temat niszowy/evergreen"


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_wiki_api = None
_news_checker = None
_seasonality = None
_trend_discovery = None

def get_wiki_api() -> WikipediaAPI:
    global _wiki_api
    if _wiki_api is None:
        _wiki_api = WikipediaAPI()
    return _wiki_api

def get_news_checker() -> NewsChecker:
    global _news_checker
    if _news_checker is None:
        _news_checker = NewsChecker()
    return _news_checker

def get_seasonality() -> SeasonalityAnalyzer:
    global _seasonality
    if _seasonality is None:
        _seasonality = SeasonalityAnalyzer()
    return _seasonality

def get_trend_discovery() -> TrendDiscovery:
    global _trend_discovery
    if _trend_discovery is None:
        _trend_discovery = TrendDiscovery()
    return _trend_discovery
