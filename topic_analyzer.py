"""
TOPIC ANALYZER MODULE (NOWY dla v4)
====================================
Ocena TEMATU zamiast tytułu:
- Generowanie tytułów z ocenami
- Generowanie obietnic z ocenami
- Analiza konkurencji YouTube
- Viral score prediction
- Podobne hity na kanale
"""

import re
import unicodedata
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from youtubesearchpython import VideosSearch
    YT_SEARCH_AVAILABLE = True
except ImportError:
    YT_SEARCH_AVAILABLE = False


class TitleGenerator:
    """
    Generuje tytuły dla tematu z ocenami i uzasadnieniami.
    Używa templates + AI + wzorców z hitów kanału.
    """
    
    TEMPLATES = {
        'mystery': [
            "Dlaczego {topic} do dziś pozostaje niewyjaśnione?",
            "Tajemnica {topic}: Co naprawdę się wydarzyło?",
            "{topic} - historia która wstrząsnęła światem",
            "Co ukrywa prawda o {topic}?",
            "Mroczna historia {topic}",
        ],
        'shock': [
            "SZOKUJĄCA prawda o {topic}",
            "Nikt nie mówi o tym co NAPRAWDĘ stało się w {topic}",
            "{topic} - fakty które zmienią Twoje postrzeganie",
            "{topic} - największy skandal w historii",
            "Dlaczego media MILCZĄ o {topic}?",
        ],
        'question': [
            "Co tak naprawdę wydarzyło się w {topic}?",
            "Dlaczego {topic} nigdy nie zostało wyjaśnione?",
            "Kto stoi za {topic}?",
            "Czy {topic} było zaplanowane?",
            "Jak doszło do {topic}?",
        ],
        'emotional': [
            "Tragedia {topic}: Historia która łamie serce",
            "{topic} - ostatnie chwile przed katastrofą",
            "Oni wiedzieli że zginą: {topic}",
            "{topic} - nagranie które mrozi krew w żyłach",
            "Relacja ocalałych z {topic}",
        ],
        'number': [
            "5 przerażających faktów o {topic}",
            "{topic}: 7 rzeczy których nie wiedziałeś",
            "3 teorie o {topic} które mogą być prawdą",
            "10 minut które zmieniły historię: {topic}",
        ],
    }

    POWER_WORDS = [
        'zginęli', 'zginęła', 'zginął', 'zakazane', 'zakazany', 'dowód', 'kłamstwo',
        'nagranie', 'prawda', 'sekret', 'afera', 'zbrodnia', 'katastrofa', 'tragedia'
    ]
    FILLER_WORDS = [
        'tajemnicze', 'mroczne', 'szok', 'niewyjaśnione', 'sensacyjne', 'niesamowite'
    ]
    QUESTION_FILLERS = ["o co", "czy to", "jak to"]

    # Prekompilowane wyrażenia regularne (dla wydajności)
    _RE_CONTEXT_NUM = re.compile(r'\d+[\s.\-]*(min|h|godz|lat|osób|ofiar|dni|ciała|km|mln|tys)', re.IGNORECASE)
    _RE_DATE_YEAR = re.compile(r'(19|20)\d{2}')
    _RE_ANY_NUMBER = re.compile(r'\d+')
    _RE_SZOK = re.compile(r'\bszok\w*\b', re.IGNORECASE)
    _RE_SCHOOL_PATTERN = re.compile(r'^(historia|tajemnica|sekret|opowieść|prawda o|wszystko o)', re.IGNORECASE)

    def __init__(self, openai_client=None, channel_data: pd.DataFrame = None):
        self.client = openai_client
        self.channel_data = channel_data
        self.hit_patterns = self._analyze_hits() if channel_data is not None else {}
    
    def _analyze_hits(self) -> Dict:
        """Analizuje wzorce z hitów kanału (PASS videos)"""
        if self.channel_data is None or 'title' not in self.channel_data.columns:
            return {}
        
        df = self.channel_data.copy()
        
        # Ensure labels exist
        if 'label' not in df.columns:
            if 'views' in df.columns:
                median = df['views'].median()
                df['label'] = df['views'].apply(
                    lambda x: 'PASS' if x > median * 1.5 else 'FAIL' if x < median * 0.5 else 'BORDER'
                )
            else:
                return {}
        
        hits = df[df['label'] == 'PASS']['title'].tolist()
        
        if not hits:
            return {}
        
        patterns = {
            'avg_length': sum(len(t) for t in hits) // len(hits),
            'has_number_pct': sum(1 for t in hits if re.search(r'\d', t)) / len(hits),
            'has_question_pct': sum(1 for t in hits if '?' in t) / len(hits),
            'has_colon_pct': sum(1 for t in hits if ':' in t) / len(hits),
            'has_caps_pct': sum(1 for t in hits if re.search(r'\b[A-Z]{2,}\b', t)) / len(hits),
            'trigger_words': self._extract_trigger_words(hits),
            'hit_titles': hits[:10],  # Sample for AI context
        }
        return patterns
    
    def _extract_trigger_words(self, titles: List[str]) -> List[str]:
        """Wyciąga słowa które powtarzają się w hitach (zoptymalizowane)"""
        words = {}
        stopwords = {'i', 'w', 'na', 'do', 'z', 'się', 'to', 'co', 'jak', 'czy', 'że', 'nie', 'o', 'za'}

        # Połącz wszystkie tytuły i wyciągnij słowa jednym regex
        all_text = ' '.join(titles).lower()
        for word in re.findall(r'\w+', all_text):
            if len(word) > 3 and word not in stopwords:
                words[word] = words.get(word, 0) + 1

        return sorted(words.keys(), key=lambda x: words[x], reverse=True)[:30]
    
    def generate(self, topic: str, n: int = 10, use_ai: bool = True) -> List[Dict]:
        """
        Generuje tytuły dla tematu.
        
        Args:
            topic: Temat filmu (np. "Operacja Northwoods")
            n: Liczba tytułów do wygenerowania
            use_ai: Czy używać AI do generowania
            
        Returns:
            Lista {title, score, reasoning, style, source}
        """
        titles = []
        
        # 1. Template-based titles
        for style, templates in self.TEMPLATES.items():
            for template in templates[:2]:  # 2 z każdego stylu
                title = template.format(topic=topic)
                score, reasoning = self._score_title(title)
                reasoning_text = " | ".join(reasoning) if reasoning else "Brak szczególnych cech"
                titles.append({
                    'title': title,
                    'score': score,
                    'reasoning': reasoning_text,
                    'style': style,
                    'source': 'template',
                })
        
        # 2. AI-generated titles
        if use_ai and self.client and len(titles) < n:
            ai_titles = self._generate_ai(topic, n - len(titles))
            titles.extend(ai_titles)
        
        # Sort by score
        titles = sorted(titles, key=lambda x: x['score'], reverse=True)
        
        return titles[:n]
    
    def _score_title(self, title: str) -> Tuple[int, List[str]]:
        """
        Ocenia tytuł i zwraca (score, reasoning).
        """
        score = 50.0
        reasons = []
        text = title.lower()
        words = text.split()
        
        # === 1. Długość (funkcja dzwonowa) ===
        length = len(title)
        dist_from_ideal = abs(length - 52)
        if 40 <= length <= 65:
            score += 10 - (0.4 * dist_from_ideal)
            reasons.append("✅ Dobra długość względem ideału (52 znaki)")
        elif 30 <= length < 40 or 66 <= length <= 75:
            score += 2 - (0.5 * dist_from_ideal)
            reasons.append("⚠️ Długość poza optimum (szybki spadek)")
        else:
            score -= 5 + (0.5 * dist_from_ideal)
            reasons.append("❌ Skrajna długość tytułu")
        
        # === 2. Liczby z kontekstem ===
        if self._RE_CONTEXT_NUM.search(text) or self._RE_DATE_YEAR.search(text):
            score += 10
            reasons.append("✅ Liczba z kontekstem (czas/osoby/data)")
        elif self._RE_ANY_NUMBER.search(text):
            score += 3
            reasons.append("✅ Liczba bez kontekstu")
        
        # === 3. Pytajnik (quality check) ===
        if '?' in title:
            if len(words) < 8 and not any(x in text for x in self.QUESTION_FILLERS):
                score += 5
                reasons.append("✅ Konkretne pytanie")
            else:
                score += 2
                reasons.append("⚠️ Generyczne pytanie")
        
        # === 4. Power words & filler (malejące korzyści) ===
        power_hits = sum(1 for w in self.POWER_WORDS if w in text)
        filler_hits = sum(1 for w in self.FILLER_WORDS if w in text)
        if power_hits:
            score += min(10, power_hits * 5)
            reasons.append("✅ Mocne słowa-klucze")
        if filler_hits:
            score += min(4, filler_hits * 2)
            reasons.append("🟡 Słowa wypełniacze")
        if (power_hits + filler_hits) > 4:
            score -= 5
            reasons.append("⚠️ Nadmiar słów-kluczy (spam)")
        
        # === 5. Interpunkcja i CAPS ===
        caps_words = [w for w in title.split() if w.isupper() and len(w) > 1]
        bad_caps = [w for w in caps_words if len(w) > 4]
        if bad_caps:
            score -= 6
            reasons.append("❌ Krzykliwy CAPS")
        elif caps_words:
            score += 2
            reasons.append("✅ Krótki CAPS (np. UFO)")
        
        if ':' in title:
            score -= 2
            reasons.append("⚠️ Dwukropek zabiera miejsce")
        
        # === 6. Anti-clickbait & anti-school ===
        if self._RE_SZOK.search(text):
            score -= 12
            reasons.append("❌ Clickbaitowe 'szok'")
        if self._RE_SCHOOL_PATTERN.search(text):
            score -= 15
            reasons.append("❌ Szkolny szablon tytułu")
        
        # === 7. Klarowność ===
        if len(words) > 12:
            score -= 5
            reasons.append("⚠️ Za dużo słów")
        if title.count(',') >= 2:
            score -= 3
            reasons.append("⚠️ Za dużo przecinków")
        punctuation_count = sum(1 for ch in title if ch in '!?;,.')
        if punctuation_count > 3:
            score -= 3
            reasons.append("⚠️ Nadmiar interpunkcji")

        # === 8. DNA kanału ===
        if self.hit_patterns:
            trigger_words = self.hit_patterns.get('trigger_words', [])
            found_triggers = [w for w in trigger_words if w in text]
            if found_triggers:
                bonus = min(12, len(found_triggers) * 4)
                score += bonus
                reasons.append(f"✅ DNA kanału: {', '.join(found_triggers[:3])}")
            
            # Match hit patterns
            if self.hit_patterns.get('has_number_pct', 0) > 0.3 and re.search(r'\d', title):
                score += 3
            if self.hit_patterns.get('has_question_pct', 0) > 0.3 and '?' in title:
                score += 3
        
        # Clamp score
        score = max(0, min(100, int(score)))
        
        return score, reasons
    
    def _generate_ai(self, topic: str, n: int) -> List[Dict]:
        """Generuje tytuły przez AI"""
        if not self.client:
            return []
        
        # Build context from hits
        hits_context = ""
        if self.hit_patterns.get('hit_titles'):
            hits_context = "\n\nPrzykłady HITÓW z tego kanału (naśladuj styl):\n"
            hits_context += "\n".join(f"- {h}" for h in self.hit_patterns['hit_titles'][:5])
        
        trigger_context = ""
        if self.hit_patterns.get('trigger_words'):
            trigger_context = f"\n\nSłowa które działają na tym kanale: {', '.join(self.hit_patterns['trigger_words'][:15])}"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"""Jesteś ekspertem od tytułów YouTube w niszy dark documentaries (mroczne dokumenty, tajemnice, zbrodnie, katastrofy).

Generujesz tytuły które:
- Budują CIEKAWOŚĆ (curiosity gap) - widz MUSI kliknąć
- Używają EMOCJI: tajemnica, szok, strach, niedowierzanie
- Są KONKRETNE (liczby, daty, nazwiska)
- Mają 40-65 znaków
- NIE są tandetnym clickbaitem - muszą być prawdziwe
- Pasują do stylu kanału{hits_context}{trigger_context}"""},
                    {"role": "user", "content": f"""Wygeneruj {n} unikalnych tytułów dla tematu: "{topic}"

Każdy tytuł powinien mieć inny styl (pytanie, szok, emocje, liczby, tajemnica).

Odpowiedz TYLKO w formacie JSON:
{{"titles": [
  {{"title": "...", "score": 70-95, "reasoning": "dlaczego dobry", "style": "mystery/shock/emotional/question/number"}}
]}}"""}
                ],
                response_format={"type": "json_object"},
                temperature=0.8,
            )
            
            result = json.loads(response.choices[0].message.content)
            titles = result.get('titles', [])
            
            for t in titles:
                t['source'] = 'ai'
                # Recalculate score with our logic
                calculated_score, calculated_reasoning = self._score_title(t['title'])
                calculated_reasoning_text = (
                    " | ".join(calculated_reasoning)
                    if calculated_reasoning
                    else "Brak szczególnych cech"
                )
                t['calculated_score'] = calculated_score
                # Use max of AI score and calculated score
                t['score'] = max(t.get('score', 0), calculated_score)
                t['reasoning'] = f"{t.get('reasoning', '')} | {calculated_reasoning_text}"
            
            return titles
            
        except Exception as e:
            print(f"AI title generation error: {e}")
            return []


class PromiseGenerator:
    """
    Generuje obietnice (hooki pod tytułem) z ocenami.
    """
    
    TEMPLATES = [
        "To co odkryjesz zmieni Twoje postrzeganie tego tematu na zawsze.",
        "Historia którą przez lata ukrywano przed opinią publiczną.",
        "Fakty które sprawią że już nigdy nie spojrzysz na to tak samo.",
        "Dlaczego odpowiedzialni za to milczą do dziś?",
        "Dowody które zmieniają wszystko co wiedzieliśmy.",
        "To nie był przypadek. To był plan.",
        "Nikt nie mówi o tym co naprawdę się wydarzyło.",
        "Prawda jest znacznie mroczniejsza niż oficjalna wersja.",
        "Co ukrywa się za zamkniętymi drzwiami?",
        "Relacja świadków których nikt nie chciał słuchać.",
        "Dokumenty które miały nigdy nie ujrzeć światła dziennego.",
        "Historia która zmieni Twoje rozumienie świata.",
    ]
    
    def __init__(self, openai_client=None):
        self.client = openai_client
    
    def generate(self, title: str, topic: str, n: int = 5, use_ai: bool = True) -> List[Dict]:
        """
        Generuje obietnice dla tytułu.
        
        Returns:
            Lista {promise, score, reasoning}
        """
        promises = []
        
        # Template-based
        for template in self.TEMPLATES[:n]:
            score, reasoning = self._score_promise(template, title)
            promises.append({
                'promise': template,
                'score': score,
                'reasoning': reasoning,
                'source': 'template',
            })
        
        # AI-generated
        if use_ai and self.client:
            ai_promises = self._generate_ai(title, topic, n)
            promises.extend(ai_promises)
        
        # Sort and return top n
        promises = sorted(promises, key=lambda x: x['score'], reverse=True)
        return promises[:n]
    
    def _score_promise(self, promise: str, title: str) -> Tuple[int, str]:
        """Ocenia obietnicę"""
        score = 50.0
        reasons = []
        text = promise.lower()
        
        # 1. Długość (Narrative Window)
        length = len(promise)
        if 120 <= length <= 220:
            score += 10
            reasons.append("✅ Filmowa długość (120-220 znaków)")
        elif 90 <= length < 120:
            score += 5
            reasons.append("🟡 Dobra długość, ale krótka")
        elif length < 90:
            score -= 10
            reasons.append("❌ Za krótka na klimat")
        elif length > 260:
            score -= 8
            reasons.append("⚠️ Za długa (ściana tekstu)")
        
        # 2. Anti-marketing
        marketing_triggers = ["w tym filmie", "dzisiaj", "opowiem", "przedstawię", "zobaczycie", "zapraszam"]
        if any(x in text for x in marketing_triggers):
            score -= 20
            reasons.append("❌ Marketingowy zwrot")
        
        # 3. Test na konkret (czas/miejsce/obiekt)
        has_time = bool(re.search(r'\d+(\:|\.)\d+|\d+\s*(lat|min|h)', text))
        has_place = any(x in text for x in ["w lesie", "na dnie", "w domu", "pokój", "piwnic", "korytarz"])
        has_object = any(x in text for x in ["raport", "taśm", "nagran", "zdjęci", "dowód", "ciał"])
        if has_time or has_place or has_object:
            score += 8
            reasons.append("✅ Konkret (czas/miejsce/obiekt)")
        else:
            score -= 10
            reasons.append("❌ Brak konkretu")
        
        # 4. Overlap z tytułem
        title_words = set(self._clean_text(title).split())
        promise_words = set(self._clean_text(promise).split())
        common = title_words & promise_words
        if len(common) == 0:
            score += 5
            reasons.append("✅ Dodaje nową wartość")
        elif len(common) > 2:
            score -= 10
            reasons.append("⚠️ Powtarza tytuł")
        
        # 5. Struktura (in medias res)
        if text.startswith(("nie", "nikt", "nigdy", "gdy", "kiedy")):
            score += 5
            reasons.append("✅ Mocny start")
        
        return max(0, min(100, int(score))), ' | '.join(reasons)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Czyści tekst do porównań overlapu."""
        stopwords = {
            'i', 'w', 'na', 'do', 'z', 'się', 'to', 'co', 'jak', 'czy', 'że', 'nie',
            'o', 'za', 'pod', 'nad', 'przez', 'dla', 'od', 'po', 'przed', 'oraz'
        }
        words = re.findall(r'\w+', text.lower())
        cleaned = [w for w in words if len(w) > 3 and w not in stopwords]
        return " ".join(cleaned)
    
    def _generate_ai(self, title: str, topic: str, n: int) -> List[Dict]:
        """Generuje obietnice przez AI"""
        if not self.client:
            return []
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """Jesteś ekspertem od YouTube hooks dla dark documentaries.

Generujesz OBIETNICE (1-2 zdania pod tytułem) które:
- Budują NAPIĘCIE i ciekawość
- NIE zdradzają rozwiązania/końca
- Obiecują WARTOŚĆ (co widz się dowie)
- Używają emocji
- Są KOMPLEMENTARNE do tytułu (nie powtarzają go)
- Są wiarygodne (nie tandetny clickbait)"""},
                    {"role": "user", "content": f"""Tytuł filmu: "{title}"
Temat: "{topic}"

Wygeneruj {n} unikalnych obietnic.

JSON: {{"promises": [{{"promise": "...", "score": 60-90, "reasoning": "dlaczego działa"}}]}}"""}
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            
            result = json.loads(response.choices[0].message.content)
            promises = result.get('promises', [])
            
            for p in promises:
                p['source'] = 'ai'
            
            return promises
            
        except Exception as e:
            print(f"AI promise generation error: {e}")
            return []


class CompetitorAnalyzer:
    """Analizuje konkurencję na YouTube dla tematu"""
    
    def __init__(self):
        self.available = YT_SEARCH_AVAILABLE
    
    def analyze(self, topic: str, max_results: int = 20) -> Dict:
        """
        Analizuje konkurencję dla tematu na YouTube.
        
        Returns:
            Dict z saturation, opportunity_score, top_videos, recommendation
        """
        if not self.available:
            return {
                'error': 'youtube-search-python niedostępne',
                'saturation': 'UNKNOWN',
                'opportunity_score': 50,
                'top_videos': [],
                'recommendation': 'Zainstaluj: pip install youtube-search-python',
            }
        
        result = {
            'topic': topic,
            'total_videos': 0,
            'high_view_videos': 0,
            'recent_videos': 0,
            'top_videos': [],
            'saturation': 'MEDIUM',
            'opportunity_score': 50,
            'recommendation': '',
        }
        
        try:
            # Search for topic
            search = VideosSearch(f"{topic} polski dokumentalny", limit=max_results, region='PL')
            videos = search.result().get('result', [])
            
            result['total_videos'] = len(videos)
            parsed_videos = []

            for vid in videos:
                view_text = vid.get('viewCount', {}).get('text', '0')
                views = self._parse_views(view_text)
                published = vid.get('publishedTime', '')
                days_old = self._parse_days_old(published)
                
                # Count high-view videos
                if views >= 50000:
                    result['high_view_videos'] += 1
                
                # Count recent videos
                if days_old is not None and days_old < 180:
                    result['recent_videos'] += 1
                
                result['top_videos'].append({
                    'title': vid.get('title', ''),
                    'views': views,
                    'channel': vid.get('channel', {}).get('name', ''),
                    'published': published,
                    'duration': vid.get('duration', ''),
                    'link': vid.get('link', ''),
                    'days_old': days_old,
                })
                parsed_videos.append({
                    'title': vid.get('title', ''),
                    'views': views,
                    'days_old': days_old,
                })
            
            # Sort by views
            result['top_videos'] = sorted(result['top_videos'], key=lambda x: x['views'], reverse=True)[:10]

            # Opportunity Score V3 (Demand vs Supply)
            top_5_videos = sorted(parsed_videos, key=lambda x: x['views'], reverse=True)[:5]
            avg_views = np.mean([v['views'] for v in top_5_videos]) if top_5_videos else 0

            demand_score = 0
            if avg_views > 200000:
                demand_score = 100
            elif avg_views > 100000:
                demand_score = 80
            elif avg_views > 50000:
                demand_score = 60
            elif avg_views < 10000:
                demand_score = 10

            recent_count = sum(
                1 for v in parsed_videos[:10]
                if v.get('days_old') is not None and v['days_old'] < 180
            )

            saturation_penalty = 0
            if recent_count >= 4:
                saturation_penalty = 60
            elif recent_count >= 2:
                saturation_penalty = 30
            elif recent_count == 0:
                saturation_penalty = -20

            niche_penalty = 0
            if len(parsed_videos) < 5:
                niche_penalty = 20

            opportunity = demand_score - saturation_penalty - niche_penalty

            top_video = top_5_videos[0] if top_5_videos else None
            if top_video and top_video['views'] > 150000 and (top_video.get('days_old') or 0) > 700:
                opportunity += 25

            result['opportunity_score'] = max(0, min(100, int(opportunity)))

            if saturation_penalty >= 60:
                result['saturation'] = 'HIGH'
            elif saturation_penalty >= 30:
                result['saturation'] = 'MEDIUM'
            else:
                result['saturation'] = 'LOW'

            if demand_score >= 80 and saturation_penalty <= 0:
                result['recommendation'] = "🟢 Wysoki popyt i niska świeża konkurencja - idealna okazja."
            elif demand_score >= 60 and saturation_penalty < 60:
                result['recommendation'] = "🟡 Dobry popyt, ale rynek jest częściowo zajęty. Szukaj unikalnego kąta."
            elif demand_score <= 10:
                result['recommendation'] = "🔴 Słaby popyt - temat może być martwy."
            else:
                result['recommendation'] = "🟠 Umiarkowany popyt lub wysoka świeża konkurencja."
            
        except TypeError as e:
            if "proxies" in str(e):
                result['error'] = str(e)
                result['recommendation'] = (
                    "Błąd kompatybilności youtube-search-python z httpx. "
                    "Zainstaluj: pip install httpx==0.24.1"
                )
            else:
                raise
        
        except Exception as e:
            result['error'] = str(e)
            result['recommendation'] = f"Błąd: {e}"
        
        return result
    
    def _parse_views(self, view_text: str) -> int:
        """Parsuje tekst views na int"""
        try:
            text = view_text.lower().replace(' ', '').replace(',', '').replace('.', '')
            
            multiplier = 1
            if 'mln' in text or 'm' in text:
                multiplier = 1000000
                text = re.sub(r'(mln|m)', '', text)
            elif 'tys' in text or 'k' in text:
                multiplier = 1000
                text = re.sub(r'(tys|k)', '', text)
            
            number = float(re.sub(r'[^\d.]', '', text) or 0)
            return int(number * multiplier)
        except (ValueError, TypeError):
            return 0

    def _parse_days_old(self, published_text: str) -> Optional[int]:
        """Szacuje wiek filmu w dniach na podstawie tekstu."""
        if not published_text:
            return None
        text = published_text.lower()
        match = re.search(r'(\d+)', text)
        if not match:
            return None
        value = int(match.group(1))
        if any(unit in text for unit in ['dzień', 'dni', 'day', 'days']):
            return value
        if any(unit in text for unit in ['tydzień', 'tygodni', 'week', 'weeks']):
            return value * 7
        if any(unit in text for unit in ['miesiąc', 'miesięcy', 'month', 'months']):
            return value * 30
        if any(unit in text for unit in ['rok', 'lata', 'lat', 'year', 'years']):
            return value * 365
        return None


class ViralScorePredictor:
    """Przewiduje potencjał viralowy tematu/tytułu"""
    AUTHORITY_WORDS = ["policja", "rząd", "fbi", "władze", "nasa"]
    NEGATION_WORDS = ["kłam", "ukry", "błąd", "zatusz", "nie"]
    THEMES = {
        "tragedy": ["zginęli", "śmierć", "ofiar", "ciało"],
        "mystery": ["zniknęli", "sygnał", "światło", "dźwięk"],
        "forbidden": ["zakaz", "bunkier", "strefa", "tajne"],
    }
    
    def __init__(self, channel_data: pd.DataFrame = None):
        self.channel_data = channel_data
        self.benchmarks = self._calculate_benchmarks() if channel_data is not None else {}
    
    def _calculate_benchmarks(self) -> Dict:
        """Oblicza benchmarki z danych kanału"""
        if self.channel_data is None or 'views' not in self.channel_data.columns:
            return {}
        
        df = self.channel_data
        return {
            'median_views': df['views'].median(),
            'top_10_pct_views': df['views'].quantile(0.9),
            'avg_retention': df['retention'].mean() if 'retention' in df.columns else None,
        }
    
    def predict(self, title: str, topic: str, competition: Dict = None) -> Dict:
        """
        Przewiduje viral score.
        
        Returns:
            Dict z viral_score (0-100), verdict, factors, recommendation
        """
        score = 50
        factors = []
        
        text = f"{title} {topic}".lower()

        has_authority = any(x in text for x in self.AUTHORITY_WORDS)
        has_negation = any(x in text for x in self.NEGATION_WORDS)
        if has_authority and has_negation:
            score += 20
            factors.append({
                'factor': 'authority_negation',
                'found': [w for w in self.AUTHORITY_WORDS if w in text],
                'bonus': '+20',
            })

        for category, words in self.THEMES.items():
            if any(w in text for w in words):
                score += 10
                factors.append({
                    'factor': category,
                    'found': [w for w in words if w in text],
                    'bonus': '+10',
                })
                break
        
        # Clamp
        score = max(0, min(100, score))
        
        # Verdict
        if score >= 75:
            verdict = "🚀 WYSOKI potencjał viralowy! Ten temat może wybuchnąć."
        elif score >= 60:
            verdict = "📈 DOBRY potencjał - może przyciągnąć nowych widzów."
        elif score >= 45:
            verdict = "➡️ STANDARDOWY zasięg - solidny temat dla stałych widzów."
        else:
            verdict = "📉 NISKI potencjał - rozważ inny kąt lub temat."
        
        return {
            'viral_score': score,
            'verdict': verdict,
            'factors': factors,
            'recommendation': self._generate_recommendation(score, factors),
        }
    
    def _generate_recommendation(self, score: int, factors: List[Dict]) -> str:
        """Generuje rekomendację jak zwiększyć viral score"""
        found_factors = {f['factor'] for f in factors}
        suggestions = []

        if 'authority_negation' not in found_factors:
            suggestions.append("Dodaj konflikt z autorytetem (np. policja/urząd)")
        if not any(f in found_factors for f in ['tragedy', 'mystery', 'forbidden']):
            suggestions.append("Dodaj wyraźny motyw (tragedia/tajemnica/zakaz)")

        if suggestions:
            return "💡 " + " | ".join(suggestions[:2])
        if score >= 70:
            return "✅ Temat ma wyraźne paliwo wiralowe."
        return "🔍 Zbyt mało konfliktu lub mocnego motywu."


class SimilarVideosFinder:
    """Znajduje podobne filmy na kanale"""

    STOPWORDS = {
        'i', 'w', 'na', 'do', 'z', 'się', 'to', 'co', 'jak', 'czy', 'że', 'nie', 'o',
        'za', 'dla', 'od', 'po', 'jest', 'było', 'była', 'byli', 'ten', 'ta', 'te',
        'oraz', 'bez', 'pod', 'nad', 'przez', 'u', 'a', 'albo', 'lub',
    }
    
    def __init__(self, channel_data: pd.DataFrame):
        self.channel_data = channel_data

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Ujednolica tekst: lower + usuń diakrytyki."""
        text = text.lower()
        normalized = unicodedata.normalize('NFKD', text)
        return ''.join(ch for ch in normalized if not unicodedata.combining(ch))

    def _extract_keywords(self, text: str) -> List[str]:
        """Wydobywa słowa kluczowe z tekstu."""
        normalized = self._normalize_text(text)
        tokens = re.findall(r"\w+", normalized, flags=re.UNICODE)
        return [t for t in tokens if len(t) > 3 and t not in self.STOPWORDS]
    
    def find(self, topic: str, title: str = None, top_n: int = 5) -> List[Dict]:
        """Znajduje podobne filmy z kanału"""
        if self.channel_data is None or 'title' not in self.channel_data.columns:
            return []
        
        df = self.channel_data.copy()
        search_text = f"{topic} {title or ''}"
        keywords = self._extract_keywords(search_text)
        
        results = []
        for idx, row in df.iterrows():
            row_title = self._normalize_text(str(row.get('title', '')))
            
            # Count keyword matches
            match_count = sum(1 for kw in keywords if kw in row_title)
            
            if match_count > 0:
                label = row.get('label', 'BORDER')
                views = row.get('views', 0)
                retention = row.get('retention', 0)
                
                # Weight by label
                weight = 2.0 if label == 'PASS' else 0.5 if label == 'FAIL' else 1.0
                
                results.append({
                    'title': row.get('title', ''),
                    'views': views,
                    'retention': retention,
                    'label': label,
                    'similarity_score': match_count * weight,
                    'insight': self._generate_insight(label, views, retention),
                    'video_id': row.get('video_id', ''),
                })
        
        # Sort by similarity
        results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_n]
    
    def _generate_insight(self, label: str, views: int, retention: float) -> str:
        """Generuje insight dla podobnego filmu"""
        if label == 'PASS':
            return f"✅ HIT - {views:,} views, {retention:.0f}% retention. Naśladuj podejście!"
        elif label == 'FAIL':
            return f"❌ SŁABY - {views:,} views. Unikaj podobnego podejścia."
        else:
            return f"🟡 ŚREDNI - {views:,} views, {retention:.0f}% retention."


class TopicEvaluator:
    """
    Główna klasa oceniająca TEMAT.
    Łączy wszystkie moduły w jeden wynik.
    """
    
    def __init__(self, openai_client=None, channel_data: pd.DataFrame = None):
        self.client = openai_client
        self.channel_data = channel_data
        
        # Initialize sub-modules
        self.title_generator = TitleGenerator(openai_client, channel_data)
        self.promise_generator = PromiseGenerator(openai_client)
        self.competitor_analyzer = CompetitorAnalyzer()
        self.viral_predictor = ViralScorePredictor(channel_data)
        self.similar_finder = SimilarVideosFinder(channel_data) if channel_data is not None else None
    
    def evaluate(self, topic: str, n_titles: int = 10, n_promises: int = 5) -> Dict:
        """
        Pełna ocena tematu.
        
        Returns:
            Kompletny wynik z tytułami, obietnicami, konkurencją, viral score, etc.
        """
        result = {
            'topic': topic,
            'timestamp': datetime.now().isoformat(),
            'titles': [],
            'selected_title': None,
            'promises': [],
            'competition': {},
            'viral_score': {},
            'similar_hits': [],
            'overall_score': 0,
            'recommendation': '',
        }
        
        # 1. Generate titles
        result['titles'] = self.title_generator.generate(topic, n=n_titles)
        
        # 2. Generate promises (seed with top title, then evaluate combos)
        seed_title = result['titles'][0]['title'] if result['titles'] else topic
        result['promises'] = self.promise_generator.generate(
            seed_title,
            topic,
            n=n_promises
        )
        
        # 3. Find best combination (title + promise)
        best_combination = None
        best_combo_score = 50.0
        
        for title_obj in result['titles']:
            t_score = title_obj.get('score', 50)
            for promise_obj in result['promises']:
                p_score, _ = self.promise_generator._score_promise(
                    promise_obj.get('promise', ''),
                    title_obj.get('title', '')
                )
                combo_score = (t_score * 0.6) + (p_score * 0.4)
                if combo_score > best_combo_score:
                    best_combo_score = combo_score
                    best_combination = (title_obj, promise_obj.get('promise', ''), t_score, p_score)
        
        if best_combination:
            best_title_obj, best_promise_text, t_score, _ = best_combination
            result['selected_title'] = best_title_obj
            result['best_title'] = best_title_obj.get('title', '')
            result['best_promise'] = best_promise_text
            
            # Re-score promises for selected title and re-order
            rescored_promises = []
            for promise_obj in result['promises']:
                p_score, reasoning = self.promise_generator._score_promise(
                    promise_obj.get('promise', ''),
                    best_title_obj.get('title', '')
                )
                updated = dict(promise_obj)
                updated['score'] = p_score
                updated['reasoning'] = reasoning
                rescored_promises.append(updated)
            rescored_promises = sorted(rescored_promises, key=lambda x: x['score'], reverse=True)
            if best_promise_text:
                for idx, item in enumerate(rescored_promises):
                    if item.get('promise') == best_promise_text:
                        rescored_promises.insert(0, rescored_promises.pop(idx))
                        break
            result['promises'] = rescored_promises
        
        # 4. Analyze competition
        result['competition'] = self.competitor_analyzer.analyze(topic)
        
        # 5. Predict viral score
        best_title = result['selected_title']['title'] if result['selected_title'] else topic
        result['viral_score'] = self.viral_predictor.predict(
            best_title, topic, result['competition']
        )
        
        # 6. Find similar videos
        if self.similar_finder:
            result['similar_hits'] = self.similar_finder.find(topic, best_title)
        
        # 7. Calculate overall score (pair-based)
        competition_score = result['competition'].get('opportunity_score', 50)
        viral_score = result['viral_score'].get('viral_score', 50)
        overall_score = int(
            (best_combo_score * 0.55) +
            (competition_score * 0.30) +
            (viral_score * 0.15)
        )
        if competition_score < 20:
            overall_score = min(overall_score, 65)
        result['overall_score'] = overall_score
        
        # 8. Generate recommendation
        result['recommendation'] = self._generate_recommendation(result)
        
        return result
    
    def _generate_recommendation(self, result: Dict) -> str:
        """Generuje końcową rekomendację"""
        score = result['overall_score']
        competition = result['competition'].get('saturation', 'MEDIUM')
        viral = result['viral_score'].get('viral_score', 50)
        
        if score >= 75:
            rec = "🟢 PUBLIKUJ! Świetny temat z wysokim potencjałem."
        elif score >= 60:
            rec = "🟡 DOBRY temat. Dopracuj tytuł i hook wg sugestii."
        elif score >= 45:
            rec = "🟠 ŚREDNI potencjał. Rozważ inny kąt lub lepszy timing."
        else:
            rec = "🔴 SŁABY temat. Poszukaj lepszego lub zmień podejście."
        
        # Additional notes
        if competition == 'HIGH':
            rec += " ⚠️ Wysoka konkurencja - potrzebujesz unikalnego kąta."
        if competition == 'LOW':
            rec += " ✨ Niska konkurencja - to Twoja szansa!"
        if viral >= 70:
            rec += " 🚀 Wysoki potencjał viralowy!"
        
        return rec


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def get_topic_evaluator(openai_client=None, channel_data=None) -> TopicEvaluator:
    """Factory function to get TopicEvaluator instance"""
    return TopicEvaluator(openai_client, channel_data)
