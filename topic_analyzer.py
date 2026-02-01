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
                titles.append({
                    'title': title,
                    'score': score,
                    'reasoning': reasoning,
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
    
    def _score_title(self, title: str) -> Tuple[int, str]:
        """
        Ocenia tytuł i zwraca (score, reasoning).
        """
        score = 50
        reasons = []
        
        # === Długość ===
        length = len(title)
        if 40 <= length <= 65:
            score += 10
            reasons.append("✅ Optymalna długość (40-65 znaków)")
        elif length > 70:
            score -= 10
            reasons.append("⚠️ Za długi tytuł (>70 znaków)")
        elif length < 30:
            score -= 5
            reasons.append("⚠️ Za krótki tytuł (<30 znaków)")
        
        # === Liczba ===
        if re.search(r'\d', title):
            score += 10
            reasons.append("✅ Zawiera liczbę (konkretność)")
        
        # === Pytanie ===
        if '?' in title:
            score += 8
            reasons.append("✅ Pytanie (buduje ciekawość)")
        
        # === Emocjonalne słowa ===
        emotional_words = [
            'szok', 'przerażając', 'niesamowit', 'tajemnic', 'prawda', 
            'tragedi', 'śmierć', 'zgin', 'mroczn', 'ukryt', 'sekret',
            'wstrząs', 'scandal', 'afera', 'zbrodnia', 'morder'
        ]
        found_emotional = [w for w in emotional_words if w in title.lower()]
        if found_emotional:
            bonus = min(15, len(found_emotional) * 5)
            score += bonus
            reasons.append(f"✅ Emocje: {', '.join(found_emotional[:3])}")
        
        # === CAPS ===
        if re.search(r'\b[A-Z]{2,}\b', title):
            score += 5
            reasons.append("✅ CAPS (zwraca uwagę)")
        
        # === Dwukropek (struktura) ===
        if ':' in title:
            score += 5
            reasons.append("✅ Struktura z dwukropkiem")
        
        # === DNA kanału ===
        if self.hit_patterns:
            trigger_words = self.hit_patterns.get('trigger_words', [])
            found_triggers = [w for w in trigger_words if w in title.lower()]
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
        score = max(0, min(100, score))
        
        return score, ' | '.join(reasons) if reasons else 'Brak szczególnych cech'
    
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
                t['calculated_score'] = calculated_score
                # Use max of AI score and calculated score
                t['score'] = max(t.get('score', 0), calculated_score)
                t['reasoning'] = f"{t.get('reasoning', '')} | {calculated_reasoning}"
            
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
        score = 50
        reasons = []
        
        # Długość
        if 50 <= len(promise) <= 150:
            score += 10
            reasons.append("✅ Dobra długość")
        elif len(promise) > 200:
            score -= 10
            reasons.append("⚠️ Za długa")
        elif len(promise) < 40:
            score -= 5
            reasons.append("⚠️ Za krótka")
        
        # Buduje napięcie
        tension_words = ['ukryt', 'tajemnic', 'prawda', 'odkryj', 'zmieni', 'nigdy', 'nikt', 'sekret', 'mroczn']
        found = [w for w in tension_words if w in promise.lower()]
        if found:
            bonus = min(15, len(found) * 5)
            score += bonus
            reasons.append(f"✅ Napięcie: {', '.join(found[:3])}")
        
        # Nie powtarza tytułu
        title_words = set(w.lower() for w in title.split() if len(w) > 3)
        promise_words = set(w.lower() for w in promise.split() if len(w) > 3)
        overlap = len(title_words & promise_words)
        
        if overlap < 2:
            score += 10
            reasons.append("✅ Dodaje nową wartość (nie powtarza)")
        elif overlap > 3:
            score -= 5
            reasons.append("⚠️ Powtarza słowa z tytułu")
        
        # Konkretność
        if any(word in promise.lower() for word in ['dokument', 'dowod', 'świadek', 'relacj', 'nagran']):
            score += 5
            reasons.append("✅ Konkretna obietnica")
        
        return max(0, min(100, score)), ' | '.join(reasons)
    
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
            
            for vid in videos:
                view_text = vid.get('viewCount', {}).get('text', '0')
                views = self._parse_views(view_text)
                published = vid.get('publishedTime', '')
                
                # Count high-view videos
                if views >= 50000:
                    result['high_view_videos'] += 1
                
                # Count recent videos
                if any(x in published.lower() for x in ['dzień', 'dni', 'day', 'tydzień', 'tygodni', 'week', 'miesiąc', 'month']):
                    result['recent_videos'] += 1
                
                result['top_videos'].append({
                    'title': vid.get('title', ''),
                    'views': views,
                    'channel': vid.get('channel', {}).get('name', ''),
                    'published': published,
                    'duration': vid.get('duration', ''),
                    'link': vid.get('link', ''),
                })
            
            # Sort by views
            result['top_videos'] = sorted(result['top_videos'], key=lambda x: x['views'], reverse=True)[:10]
            
            # Calculate saturation and opportunity
            high_views = result['high_view_videos']
            recent = result['recent_videos']
            
            if high_views >= 5:
                result['saturation'] = 'HIGH'
                result['opportunity_score'] = 25
                result['recommendation'] = "🔴 WYSOKA konkurencja - temat mocno eksploatowany. Potrzebujesz unikalnego kąta lub świeżych informacji."
            elif high_views >= 3:
                result['saturation'] = 'MEDIUM'
                result['opportunity_score'] = 55
                result['recommendation'] = "🟡 ŚREDNIA konkurencja - jest miejsce, ale musisz się wyróżnić."
            else:
                result['saturation'] = 'LOW'
                result['opportunity_score'] = 80
                result['recommendation'] = "🟢 NISKA konkurencja - świetna okazja! Mało filmów o tym temacie."
            
            # Bonus if no recent videos
            if recent == 0 and result['total_videos'] > 0:
                result['opportunity_score'] += 15
                result['recommendation'] += " ✨ Brak świeżych filmów - idealne okno czasowe!"
            
            result['opportunity_score'] = min(100, result['opportunity_score'])
            
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


class ViralScorePredictor:
    """Przewiduje potencjał viralowy tematu/tytułu"""
    
    VIRAL_FACTORS = {
        'emotional_intensity': {
            'keywords': ['szok', 'niesamowit', 'niewiarygod', 'przerażając', 'wstrząsając', 'poruszając'],
            'weight': 15,
        },
        'controversy': {
            'keywords': ['skandal', 'afera', 'oszust', 'kłamst', 'ukrywa', 'cenzur', 'zakazan'],
            'weight': 12,
        },
        'mystery': {
            'keywords': ['tajemnic', 'zagadk', 'niewyjaśnion', 'zaginion', 'sekret', 'odkry'],
            'weight': 10,
        },
        'tragedy': {
            'keywords': ['tragedi', 'śmierć', 'zgin', 'ofiar', 'katastro', 'wypadek'],
            'weight': 10,
        },
        'relatability': {
            'keywords': ['polsk', 'nasz', 'twój', 'każdy'],
            'weight': 8,
        },
        'urgency': {
            'keywords': ['teraz', 'właśnie', 'pilne', 'dziś'],
            'weight': 5,
        },
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
        
        # Check viral factors
        for factor_name, factor_data in self.VIRAL_FACTORS.items():
            keywords = factor_data['keywords']
            weight = factor_data['weight']
            
            found = [kw for kw in keywords if kw in text]
            if found:
                bonus = min(weight, len(found) * (weight // 2))
                score += bonus
                factors.append({
                    'factor': factor_name,
                    'found': found,
                    'bonus': f"+{bonus}",
                })
        
        # Competition factor
        if competition:
            saturation = competition.get('saturation', 'MEDIUM')
            if saturation == 'LOW':
                score += 15
                factors.append({'factor': 'low_competition', 'found': ['Niska konkurencja'], 'bonus': '+15'})
            elif saturation == 'HIGH':
                score -= 10
                factors.append({'factor': 'high_competition', 'found': ['Wysoka konkurencja'], 'bonus': '-10'})
        
        # Length factor
        if 40 <= len(title) <= 65:
            score += 5
            factors.append({'factor': 'optimal_length', 'found': ['Optymalna długość'], 'bonus': '+5'})
        
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
        found_factors = [f['factor'] for f in factors]
        suggestions = []
        
        if 'emotional_intensity' not in found_factors:
            suggestions.append("Dodaj emocjonalne słowa (szokujące, niesamowite)")
        if 'mystery' not in found_factors:
            suggestions.append("Dodaj element tajemnicy")
        if 'controversy' not in found_factors and score < 60:
            suggestions.append("Rozważ kontrowersyjny kąt")
        
        if suggestions:
            return "💡 " + " | ".join(suggestions[:2])
        elif score >= 70:
            return "✅ Wszystkie kluczowe elementy obecne!"
        else:
            return "🔍 Sprawdź konkurencję i timing"


class SimilarVideosFinder:
    """Znajduje podobne filmy na kanale"""
    
    def __init__(self, channel_data: pd.DataFrame):
        self.channel_data = channel_data
    
    def find(self, topic: str, title: str = None, top_n: int = 5) -> List[Dict]:
        """Znajduje podobne filmy z kanału"""
        if self.channel_data is None or 'title' not in self.channel_data.columns:
            return []
        
        df = self.channel_data.copy()
        search_text = f"{topic} {title or ''}".lower()
        keywords = [w for w in search_text.split() if len(w) > 3]
        
        results = []
        for idx, row in df.iterrows():
            row_title = str(row.get('title', '')).lower()
            
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
        
        if result['titles']:
            result['selected_title'] = result['titles'][0]  # Best one
            
            # 2. Generate promises for best title
            result['promises'] = self.promise_generator.generate(
                result['selected_title']['title'],
                topic,
                n=n_promises
            )
        
        # 3. Analyze competition
        result['competition'] = self.competitor_analyzer.analyze(topic)
        
        # 4. Predict viral score
        best_title = result['selected_title']['title'] if result['selected_title'] else topic
        result['viral_score'] = self.viral_predictor.predict(
            best_title, topic, result['competition']
        )
        
        # 5. Find similar videos
        if self.similar_finder:
            result['similar_hits'] = self.similar_finder.find(topic, best_title)
        
        # 6. Calculate overall score
        title_score = result['selected_title']['score'] if result['selected_title'] else 50
        competition_score = result['competition'].get('opportunity_score', 50)
        viral_score = result['viral_score'].get('viral_score', 50)
        
        result['overall_score'] = int(
            title_score * 0.35 +
            competition_score * 0.30 +
            viral_score * 0.35
        )
        
        # 7. Generate recommendation
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
