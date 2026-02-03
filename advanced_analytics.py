"""
ADVANCED ANALYTICS MODULE
==========================
Zaawansowane analizy dla YT Idea Evaluator Pro v2:
1. Hook Analyzer - analiza hooków z transkryptów
2. Google Trends Integration - sprawdzanie trendów
3. Timing Predictor - najlepszy czas publikacji
4. Competition Scanner - analiza konkurencji na YT
5. Packaging DNA - wzorce z hitów
"""

import os
import re
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter
import numpy as np
import pandas as pd

# Opcjonalne importy
try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False

try:
    from youtubesearchpython import VideosSearch
    YT_SEARCH_AVAILABLE = True
except ImportError:
    YT_SEARCH_AVAILABLE = False

# Import PromiseGenerator z topic_analyzer (unikamy duplikacji)
try:
    from topic_analyzer import PromiseGenerator as _TopicPromiseGenerator
    _TOPIC_ANALYZER_AVAILABLE = True
except ImportError:
    _TopicPromiseGenerator = None
    _TOPIC_ANALYZER_AVAILABLE = False


# =============================================================================
# 1. HOOK ANALYZER
# =============================================================================

class HookAnalyzer:
    """
    Analizuje hooki (pierwsze 120s) filmów.
    Wykrywa wzorce, struktury i elementy które działają.
    """
    
    # Wzorce hookowe które działają w dark doc
    HOOK_PATTERNS = {
        "cold_open": {
            "description": "Zaczyna od środka akcji, bez wprowadzenia",
            "triggers": [
                r"^[A-Z][^.!?]*\d+[^.!?]*[.!?]",  # Zaczyna od faktu z liczbą
                r"^\"[^\"]+\"",  # Zaczyna od cytatu
                r"^O \d+:\d+",  # Zaczyna od konkretnej godziny
            ],
            "weight": 1.2
        },
        "pattern_interrupt": {
            "description": "Coś nieoczekiwanego, łamie schemat",
            "triggers": [
                r"ale (nikt|nic|nigdy)",
                r"jednak (okazało się|prawda)",
                r"problem w tym",
                r"i wtedy",
            ],
            "weight": 1.15
        },
        "stakes_immediate": {
            "description": "Od razu pokazuje stawkę/konsekwencje",
            "triggers": [
                r"\d+ (osób|ludzi|ofiar|zgonów)",
                r"(zginęło|zmarło|zaginęło)",
                r"milion(y|ów)",
                r"katastrofa|tragedia|śmierć",
            ],
            "weight": 1.25
        },
        "mystery_setup": {
            "description": "Buduje tajemnicę od pierwszego zdania",
            "triggers": [
                r"nikt nie (wie|wiedział|rozumie)",
                r"do dziś (nie|pozostaje)",
                r"tajemnic",
                r"zagadk",
                r"niewyjaśnion",
            ],
            "weight": 1.2
        },
        "question_hook": {
            "description": "Zaczyna od pytania retorycznego",
            "triggers": [
                r"^(Czy|Co|Jak|Dlaczego|Kiedy|Gdzie|Kto) [^?]+\?",
                r"(zastanawiałeś|myślałeś|słyszałeś)",
            ],
            "weight": 1.1
        },
        "contrast_setup": {
            "description": "Buduje kontrast/paradoks",
            "triggers": [
                r"(wydawało się|wyglądało).*(ale|jednak|mimo)",
                r"z jednej strony.*(z drugiej|ale)",
                r"(wszyscy|nikt).*(ale|jednak|oprócz)",
            ],
            "weight": 1.15
        },
        "time_pressure": {
            "description": "Wprowadza element czasu/pilności",
            "triggers": [
                r"w ciągu (kilku|zaledwie|\d+)",
                r"(minęło|upłynęło|zostało) (tylko|zaledwie)",
                r"\d+ (sekund|minut|godzin|dni)",
                r"(natychmiast|momentalnie|błyskawicznie)",
            ],
            "weight": 1.1
        },
        "direct_address": {
            "description": "Bezpośrednio zwraca się do widza",
            "triggers": [
                r"(pomyśl|wyobraź|zastanów) (się|sobie)",
                r"(pewnie|może|być może) (myślisz|sądzisz|uważasz)",
                r"(twoj|twoi|twoim)",
            ],
            "weight": 1.05
        }
    }
    
    # Red flags w hookach
    HOOK_RED_FLAGS = {
        "slow_start": {
            "description": "Za wolne wprowadzenie",
            "triggers": [
                r"^(Witaj|Cześć|Hej|Dzisiaj)",
                r"^W (tym|dzisiejszym) (filmie|odcinku|materiale)",
                r"^(Zanim|Najpierw|Na początek)",
            ],
            "penalty": -15
        },
        "generic_intro": {
            "description": "Generyczne intro bez hooka",
            "triggers": [
                r"opowiem (ci|wam|o)",
                r"przedstawię (ci|wam)",
                r"historia (którą|o której)",
            ],
            "penalty": -10
        },
        "spoiler_early": {
            "description": "Za szybko zdradza rozwiązanie",
            "triggers": [
                r"(okazało się|odpowiedź|rozwiązanie).{0,50}(było|jest|brzmi)",
                r"(winny|sprawca|przyczyna) (był|była|okazał)",
            ],
            "penalty": -20
        },
        "too_long_setup": {
            "description": "Za długie budowanie kontekstu bez payoff",
            "triggers": [],  # Sprawdzane przez długość bez pattern_interrupt
            "penalty": -10
        }
    }
    
    def __init__(self):
        self.hook_corpus = []  # Lista hooków do analizy
        self.patterns_compiled = self._compile_patterns()
        
    def _compile_patterns(self) -> Dict:
        """Kompiluje regex patterns dla wydajności"""
        compiled = {"positive": {}, "negative": {}}
        
        for name, pattern in self.HOOK_PATTERNS.items():
            compiled["positive"][name] = [
                re.compile(t, re.IGNORECASE | re.MULTILINE) 
                for t in pattern["triggers"]
            ]
        
        for name, pattern in self.HOOK_RED_FLAGS.items():
            compiled["negative"][name] = [
                re.compile(t, re.IGNORECASE | re.MULTILINE) 
                for t in pattern["triggers"]
            ]
        
        return compiled
    
    def analyze_hook(self, hook_text: str) -> Dict:
        """
        Analizuje pojedynczy hook.
        
        Returns:
            Dict z oceną, wykrytymi wzorcami i sugestiami
        """
        if not hook_text or len(hook_text) < 50:
            return {
                "score": 0,
                "patterns_found": [],
                "red_flags": ["TOO_SHORT"],
                "suggestions": ["Hook jest za krótki do analizy"],
                "structure_analysis": None
            }
        
        # Normalizuj tekst (usuń duplikaty z transkrypcji)
        hook_text = self._clean_hook_text(hook_text)
        
        patterns_found = []
        red_flags = []
        base_score = 50
        
        # Sprawdź pozytywne wzorce
        for name, regexes in self.patterns_compiled["positive"].items():
            for regex in regexes:
                if regex.search(hook_text):
                    patterns_found.append({
                        "pattern": name,
                        "description": self.HOOK_PATTERNS[name]["description"],
                        "weight": self.HOOK_PATTERNS[name]["weight"]
                    })
                    base_score += 10 * self.HOOK_PATTERNS[name]["weight"]
                    break  # Jeden pattern raz
        
        # Sprawdź red flags
        for name, regexes in self.patterns_compiled["negative"].items():
            for regex in regexes:
                if regex.search(hook_text):
                    red_flags.append({
                        "flag": name,
                        "description": self.HOOK_RED_FLAGS[name]["description"],
                        "penalty": self.HOOK_RED_FLAGS[name]["penalty"]
                    })
                    base_score += self.HOOK_RED_FLAGS[name]["penalty"]
                    break
        
        # Analiza struktury
        structure = self._analyze_structure(hook_text)
        
        # Dodatkowe punkty za strukturę
        if structure["first_sentence_impact"] >= 7:
            base_score += 10
        if structure["tension_build"]:
            base_score += 8
        
        # Generuj sugestie
        suggestions = self._generate_suggestions(patterns_found, red_flags, structure)
        
        return {
            "score": max(0, min(100, int(base_score))),
            "patterns_found": patterns_found,
            "red_flags": red_flags,
            "suggestions": suggestions,
            "structure_analysis": structure
        }
    
    def _clean_hook_text(self, text: str) -> str:
        """Czyści hook z duplikatów (problem z transkrypcją)"""
        # Usuń powtórzone frazy (częste w auto-transkrypcji)
        words = text.split()
        cleaned = []
        i = 0
        while i < len(words):
            # Sprawdź czy następne 3-5 słów się powtarza
            found_repeat = False
            for window in [5, 4, 3]:
                if i + window * 2 <= len(words):
                    chunk1 = " ".join(words[i:i+window])
                    chunk2 = " ".join(words[i+window:i+window*2])
                    if chunk1.lower() == chunk2.lower():
                        cleaned.extend(words[i:i+window])
                        i += window * 2
                        found_repeat = True
                        break
            if not found_repeat:
                cleaned.append(words[i])
                i += 1
        
        return " ".join(cleaned)
    
    def _analyze_structure(self, hook_text: str) -> Dict:
        """Analizuje strukturę hooka"""
        sentences = re.split(r'[.!?]+', hook_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return {
                "sentence_count": 0,
                "first_sentence_impact": 0,
                "tension_build": False,
                "has_payoff_tease": False
            }
        
        first = sentences[0]
        
        # Impact pierwszego zdania
        first_impact = 5  # baseline
        if re.search(r'\d+', first):
            first_impact += 2  # liczby = konkret
        if len(first) < 80:
            first_impact += 1  # krótkie = punchowe
        if re.search(r'[!?]', first):
            first_impact += 1  # emocja
        if re.search(r'(śmierć|zginął|zaginął|tajemnic|katastrofa)', first, re.I):
            first_impact += 2  # high stakes
        
        # Czy buduje napięcie?
        tension_words = ["ale", "jednak", "mimo", "chociaż", "niestety", "problem"]
        tension_build = any(w in hook_text.lower() for w in tension_words)
        
        # Czy jest teaser payoffu?
        payoff_patterns = [
            r"(za chwilę|później|wkrótce|niedługo)",
            r"(okaże się|przekonasz|zobaczysz|dowiesz)",
            r"(najpierw|zanim|ale wcześniej)"
        ]
        has_payoff = any(re.search(p, hook_text, re.I) for p in payoff_patterns)
        
        return {
            "sentence_count": len(sentences),
            "first_sentence_impact": min(10, first_impact),
            "first_sentence": first[:100],
            "tension_build": tension_build,
            "has_payoff_tease": has_payoff,
            "avg_sentence_length": np.mean([len(s.split()) for s in sentences])
        }
    
    def _generate_suggestions(
        self, 
        patterns: List[Dict], 
        flags: List[Dict], 
        structure: Dict
    ) -> List[str]:
        """Generuje konkretne sugestie poprawy"""
        suggestions = []
        
        pattern_names = [p["pattern"] for p in patterns]
        flag_names = [f["flag"] for f in flags]
        
        # Brak cold_open
        if "cold_open" not in pattern_names and "slow_start" in flag_names:
            suggestions.append(
                "Zacznij od środka akcji - pierwsza scena powinna być jak wejście do filmu w połowie"
            )
        
        # Brak stakes
        if "stakes_immediate" not in pattern_names:
            suggestions.append(
                "Dodaj stawkę w pierwszych 30 sekundach - ile osób zginęło? co było zagrożone?"
            )
        
        # Brak mystery
        if "mystery_setup" not in pattern_names:
            suggestions.append(
                "Zbuduj tajemnicę - 'Nikt nie wie...', 'Do dziś niewyjaśnione...'"
            )
        
        # Słaby pierwszy sentence
        if structure and structure.get("first_sentence_impact", 0) < 6:
            suggestions.append(
                "Przepisz pierwsze zdanie - powinno być punchy, z liczbą lub silnym obrazem"
            )
        
        # Brak napięcia
        if structure and not structure.get("tension_build"):
            suggestions.append(
                "Dodaj kontrast/zwrot - 'Wydawało się idealne... ale nikt nie wiedział, że...'"
            )
        
        # Generic intro detected
        if "generic_intro" in flag_names:
            suggestions.append(
                "Usuń 'W tym filmie opowiem...' - zacznij od POKAZYWANIA nie MÓWIENIA"
            )
        
        if not suggestions:
            suggestions.append("Hook ma dobrą strukturę - minor tweaks mogą go jeszcze wzmocnić")
        
        return suggestions[:5]  # Max 5 sugestii
    
    def train_on_corpus(self, df: pd.DataFrame, hook_col: str = "hook", label_col: str = "label"):
        """
        Trenuje analyzer na korpusie hooków z labelkami.
        Wyciąga wzorce które korelują z PASS vs FAIL.
        """
        if hook_col not in df.columns:
            return {"error": f"Brak kolumny {hook_col}"}
        
        pass_hooks = df[df[label_col] == "PASS"][hook_col].dropna().tolist()
        fail_hooks = df[df[label_col] == "FAIL"][hook_col].dropna().tolist()
        
        # Zbierz statystyki wzorców
        pass_patterns = Counter()
        fail_patterns = Counter()
        
        for hook in pass_hooks:
            hook = self._clean_hook_text(str(hook))
            for name, regexes in self.patterns_compiled["positive"].items():
                if any(r.search(hook) for r in regexes):
                    pass_patterns[name] += 1
        
        for hook in fail_hooks:
            hook = self._clean_hook_text(str(hook))
            for name, regexes in self.patterns_compiled["positive"].items():
                if any(r.search(hook) for r in regexes):
                    fail_patterns[name] += 1
        
        # Normalizuj
        pass_total = len(pass_hooks) or 1
        fail_total = len(fail_hooks) or 1
        
        pattern_lift = {}
        for pattern in self.HOOK_PATTERNS.keys():
            pass_rate = pass_patterns.get(pattern, 0) / pass_total
            fail_rate = fail_patterns.get(pattern, 0) / fail_total
            lift = (pass_rate / (fail_rate + 0.01))  # Unikaj dzielenia przez 0
            pattern_lift[pattern] = {
                "pass_rate": round(pass_rate * 100, 1),
                "fail_rate": round(fail_rate * 100, 1),
                "lift": round(lift, 2)
            }
        
        return {
            "pass_hooks_count": len(pass_hooks),
            "fail_hooks_count": len(fail_hooks),
            "pattern_effectiveness": pattern_lift,
            "top_patterns": sorted(pattern_lift.items(), key=lambda x: x[1]["lift"], reverse=True)[:5]
        }


# =============================================================================
# 2. GOOGLE TRENDS INTEGRATION
# =============================================================================

class TrendsAnalyzer:
    """
    Integracja z Google Trends.
    Sprawdza czy temat jest trending, evergreen czy martwy.
    """
    
    def __init__(self):
        self.pytrends = None
        if PYTRENDS_AVAILABLE:
            try:
                self.pytrends = TrendReq(hl='pl-PL', tz=60)
            except Exception as e:
                print(f"⚠ Nie udało się zainicjalizować pytrends: {e}")
    
    def check_trend(self, keywords: List[str], timeframe: str = "today 3-m") -> Dict:
        """
        Sprawdza trend dla podanych słów kluczowych.
        
        Args:
            keywords: Lista słów kluczowych (max 5)
            timeframe: Zakres czasu ('today 3-m', 'today 12-m', 'today 5-y')
            
        Returns:
            Dict z analizą trendu
        """
        if not PYTRENDS_AVAILABLE or not self.pytrends:
            return self._mock_trend_response(keywords)
        
        try:
            keywords = keywords[:5]  # Max 5
            self.pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='PL')
            
            # Interest over time
            interest_df = self.pytrends.interest_over_time()
            
            if interest_df.empty:
                return {
                    "status": "NO_DATA",
                    "message": "Brak danych dla tych słów kluczowych",
                    "keywords": keywords
                }
            
            results = {}
            for kw in keywords:
                if kw in interest_df.columns:
                    series = interest_df[kw]
                    results[kw] = self._analyze_trend_series(series, kw)
            
            # Overall recommendation
            overall = self._overall_trend_assessment(results)
            
            return {
                "status": "OK",
                "keywords": results,
                "overall": overall,
                "timeframe": timeframe
            }
            
        except Exception as e:
            error_message = str(e)
            if self._is_rate_limit_error(e, error_message):
                return {
                    "status": "RATE_LIMIT",
                    "message": "Przekroczono limit zapytań Google Trends (429). Spróbuj ponownie później.",
                    "keywords": keywords
                }
            return {
                "status": "ERROR",
                "message": error_message,
                "keywords": keywords
            }

    @staticmethod
    def _is_rate_limit_error(exc: Exception, message: str) -> bool:
        if hasattr(exc, "response"):
            response = getattr(exc, "response")
            status_code = getattr(response, "status_code", None)
            if status_code == 429:
                return True
        lowered = message.lower()
        return (
            "429" in lowered
            or "too many requests" in lowered
            or "rate limit" in lowered
        )
    
    def _analyze_trend_series(self, series: pd.Series, keyword: str) -> Dict:
        """Analizuje serię czasową trendu"""
        values = series.values
        
        if len(values) < 4:
            return {"trend": "UNKNOWN", "score": 50}
        
        # Średnie z okresów
        recent = np.mean(values[-4:])  # Ostatni miesiąc
        previous = np.mean(values[-8:-4])  # Poprzedni miesiąc
        old = np.mean(values[:len(values)//2])  # Pierwsza połowa
        
        # Oblicz trend
        recent_change = ((recent - previous) / (previous + 1)) * 100
        overall_change = ((recent - old) / (old + 1)) * 100
        
        # Peak detection
        peak = np.max(values)
        peak_idx = np.argmax(values)
        peak_recent = peak_idx > len(values) * 0.7
        
        # Określ status
        if recent_change > 30 and recent > 50:
            trend = "TRENDING_UP"
            emoji = "🔥"
            score = 90
        elif recent_change > 10:
            trend = "GROWING"
            emoji = "📈"
            score = 75
        elif abs(recent_change) <= 10 and recent > 30:
            trend = "STABLE"
            emoji = "➡️"
            score = 60
        elif recent_change < -20:
            trend = "DECLINING"
            emoji = "📉"
            score = 35
        elif recent < 15:
            trend = "LOW_INTEREST"
            emoji = "💀"
            score = 20
        else:
            trend = "MODERATE"
            emoji = "〰️"
            score = 50
        
        return {
            "trend": trend,
            "emoji": emoji,
            "score": score,
            "current_interest": int(recent),
            "change_vs_last_month": round(recent_change, 1),
            "change_vs_start": round(overall_change, 1),
            "peak_value": int(peak),
            "peak_recent": peak_recent
        }
    
    def _overall_trend_assessment(self, results: Dict) -> Dict:
        """Ogólna ocena trendu"""
        if not results:
            return {"recommendation": "NO_DATA", "score": 50}
        
        avg_score = np.mean([r["score"] for r in results.values()])
        trends = [r["trend"] for r in results.values()]
        
        if "TRENDING_UP" in trends:
            rec = "GOOD_TIMING"
            msg = "🔥 Temat jest HOT - publikuj szybko!"
        elif all(t == "DECLINING" for t in trends):
            rec = "MISSED_WINDOW"
            msg = "📉 Temat przeszedł - rozważ inny kąt lub poczekaj na odrodzenie"
        elif all(t == "LOW_INTEREST" for t in trends):
            rec = "NICHE_RISK"
            msg = "💀 Bardzo niskie zainteresowanie - ryzykowny wybór"
        elif "STABLE" in trends or "MODERATE" in trends:
            rec = "EVERGREEN"
            msg = "➡️ Temat stabilny/evergreen - możesz publikować kiedy chcesz"
        else:
            rec = "NEUTRAL"
            msg = "〰️ Trend neutralny - liczy się jakość packagingu"
        
        return {
            "recommendation": rec,
            "message": msg,
            "score": round(avg_score),
            "trend_bonus": self._calculate_trend_bonus(rec)
        }
    
    def _calculate_trend_bonus(self, recommendation: str) -> int:
        """Bonus/kara do final score na podstawie trendu"""
        bonuses = {
            "GOOD_TIMING": 10,
            "EVERGREEN": 5,
            "NEUTRAL": 0,
            "MISSED_WINDOW": -5,
            "NICHE_RISK": -10,
            "NO_DATA": 0
        }
        return bonuses.get(recommendation, 0)
    
    def _mock_trend_response(self, keywords: List[str]) -> Dict:
        """Mock response gdy pytrends niedostępne"""
        return {
            "status": "MOCK",
            "message": "Google Trends niedostępne - zainstaluj: pip install pytrends",
            "keywords": keywords,
            "overall": {
                "recommendation": "UNKNOWN",
                "message": "Nie można sprawdzić trendu",
                "score": 50,
                "trend_bonus": 0
            }
        }
    
    def extract_keywords_from_title(self, title: str) -> List[str]:
        """Wyciąga słowa kluczowe z tytułu do sprawdzenia trendu"""
        # Usuń stopwords
        stopwords = {
            "i", "w", "na", "do", "z", "o", "że", "to", "się", "nie", "jak",
            "ale", "co", "ten", "ta", "tym", "tej", "tego", "czy", "po", "za",
            "od", "dla", "przy", "przez", "przed", "który", "która", "które",
            "dlaczego", "kiedy", "gdzie", "kto", "co"
        }
        
        # Tokenizuj i filtruj
        words = re.findall(r'\b[A-Za-zżźćńółęąśŻŹĆĄŚĘÓŁŃ]{3,}\b', title.lower())
        keywords = [w for w in words if w not in stopwords]
        
        # Weź top 3-5 najdłuższych (prawdopodobnie najbardziej znaczących)
        keywords = sorted(set(keywords), key=len, reverse=True)[:5]
        
        return keywords


# =============================================================================
# 3. TIMING PREDICTOR
# =============================================================================

class TimingPredictor:
    """
    Analizuje optymalne czasy publikacji na podstawie historii kanału.
    """
    
    def __init__(self):
        self.day_performance = None
        self.topic_seasonality = None
        self.gap_analysis = None
    
    def analyze_timing(self, df: pd.DataFrame, date_col: str = "published") -> Dict:
        """
        Analizuje dane historyczne pod kątem timingu.
        
        Returns:
            Dict z rekomendacjami dot. timingu
        """
        if date_col not in df.columns:
            # Spróbuj alternatywnych nazw
            for alt in ["publishedAt", "published_at", "date", "publishedat"]:
                if alt in df.columns:
                    date_col = alt
                    break
            else:
                return {"error": "Brak kolumny z datą publikacji"}
        
        df = df.copy()
        df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["_date"])
        
        if len(df) < 10:
            return {"error": "Za mało danych (min 10 filmów)"}
        
        results = {
            "day_of_week": self._analyze_day_of_week(df),
            "topic_gaps": self._analyze_topic_gaps(df),
            "seasonal_patterns": self._analyze_seasonality(df),
            "recommendations": []
        }
        
        # Generuj rekomendacje
        results["recommendations"] = self._generate_timing_recommendations(results)
        
        return results
    
    def _analyze_day_of_week(self, df: pd.DataFrame) -> Dict:
        """Analizuje performance wg dnia tygodnia"""
        df["_dow"] = df["_date"].dt.day_name()
        
        day_stats = {}
        days_pl = {
            "Monday": "Poniedziałek",
            "Tuesday": "Wtorek", 
            "Wednesday": "Środa",
            "Thursday": "Czwartek",
            "Friday": "Piątek",
            "Saturday": "Sobota",
            "Sunday": "Niedziela"
        }
        
        for day in days_pl.keys():
            day_df = df[df["_dow"] == day]
            if len(day_df) >= 2 and "views" in df.columns:
                day_stats[days_pl[day]] = {
                    "count": len(day_df),
                    "avg_views": int(day_df["views"].mean()),
                    "median_views": int(day_df["views"].median()),
                    "best_video": day_df.nlargest(1, "views")["title"].iloc[0] if len(day_df) > 0 else None
                }
        
        if not day_stats:
            return {"best_day": None, "analysis": "Za mało danych"}
        
        # Znajdź najlepszy dzień
        best_day = max(day_stats.items(), key=lambda x: x[1]["avg_views"])
        worst_day = min(day_stats.items(), key=lambda x: x[1]["avg_views"])
        worst_avg = worst_day[1]["avg_views"]
        improvement_potential = None
        if worst_avg > 0:
            improvement_potential = round((best_day[1]["avg_views"] / worst_avg - 1) * 100, 1)
        
        return {
            "by_day": day_stats,
            "best_day": best_day[0],
            "best_day_avg": best_day[1]["avg_views"],
            "worst_day": worst_day[0],
            "worst_day_avg": worst_day[1]["avg_views"],
            "improvement_potential": improvement_potential
        }
    
    def _analyze_topic_gaps(self, df: pd.DataFrame) -> Dict:
        """Analizuje przerwy między podobnymi tematami"""
        if "title" not in df.columns:
            return {}
        
        # Wykryj kategorie tematów (uproszczone)
        topic_keywords = {
            "katastrofy": ["katastrofa", "wypadek", "zawalenie", "eksplozja", "pożar"],
            "zbrodnie": ["morderstwo", "zabójstwo", "śmierć", "zaginięcie", "porwanie"],
            "tajemnice": ["tajemnica", "zagadka", "niewyjaśnione", "paranormalne"],
            "afery": ["afera", "skandal", "oszustwo", "korupcja", "spisek"],
            "kulty": ["sekta", "kult", "przywódca", "manipulacja"],
        }
        
        df["_topic"] = df["title"].apply(
            lambda t: self._detect_topic(str(t).lower(), topic_keywords)
        )
        
        # Oblicz średnią przerwę między tematami
        gaps = {}
        for topic in topic_keywords.keys():
            topic_df = df[df["_topic"] == topic].sort_values("_date")
            if len(topic_df) >= 2:
                diffs = topic_df["_date"].diff().dropna()
                avg_gap = diffs.mean().days if len(diffs) > 0 else None
                if avg_gap:
                    last_date = topic_df["_date"].max()
                    # Handle timezone-aware dates
                    if hasattr(last_date, 'tz') and last_date.tz is not None:
                        last_date = last_date.tz_localize(None)
                    try:
                        days_since = (datetime.now() - last_date).days
                    except (TypeError, ValueError):
                        days_since = 0
                    
                    gaps[topic] = {
                        "avg_gap_days": round(avg_gap, 1),
                        "last_video": topic_df["_date"].max().strftime("%Y-%m-%d"),
                        "count": len(topic_df),
                        "days_since_last": days_since
                    }
        
        return gaps
    
    def _detect_topic(self, title: str, topic_keywords: Dict) -> str:
        """Wykrywa temat na podstawie tytułu"""
        for topic, keywords in topic_keywords.items():
            if any(kw in title for kw in keywords):
                return topic
        return "inne"
    
    def _analyze_seasonality(self, df: pd.DataFrame) -> Dict:
        """Analizuje sezonowość"""
        df["_month"] = df["_date"].dt.month
        
        if "views" not in df.columns:
            return {}
        
        month_stats = df.groupby("_month")["views"].agg(["mean", "count"]).round(0)
        
        months_pl = {
            1: "Styczeń", 2: "Luty", 3: "Marzec", 4: "Kwiecień",
            5: "Maj", 6: "Czerwiec", 7: "Lipiec", 8: "Sierpień",
            9: "Wrzesień", 10: "Październik", 11: "Listopad", 12: "Grudzień"
        }
        
        seasonal = {}
        for month, row in month_stats.iterrows():
            if row["count"] >= 2:
                seasonal[months_pl[month]] = {
                    "avg_views": int(row["mean"]),
                    "videos_count": int(row["count"])
                }
        
        if not seasonal:
            return {"pattern": "Za mało danych"}
        
        best_month = max(seasonal.items(), key=lambda x: x[1]["avg_views"])
        worst_month = min(seasonal.items(), key=lambda x: x[1]["avg_views"])
        
        return {
            "by_month": seasonal,
            "best_month": best_month[0],
            "worst_month": worst_month[0],
            "seasonality_strength": round(
                (best_month[1]["avg_views"] / worst_month[1]["avg_views"] - 1) * 100, 1
            )
        }
    
    def _generate_timing_recommendations(self, analysis: Dict) -> List[str]:
        """Generuje rekomendacje dot. timingu"""
        recs = []
        
        # Dzień tygodnia
        dow = analysis.get("day_of_week", {})
        if dow.get("best_day") and dow.get("improvement_potential", 0) > 15:
            recs.append(
                f"📅 Publikuj w {dow['best_day']} - średnio {dow['improvement_potential']:.0f}% więcej views niż w {dow['worst_day']}"
            )
        
        # Luki tematyczne
        gaps = analysis.get("topic_gaps", {})
        for topic, data in gaps.items():
            days_since = data.get("days_since_last", 0)
            avg_gap = data.get("avg_gap_days", 30)
            if days_since > avg_gap * 1.5:
                recs.append(
                    f"🕐 Temat '{topic}' - {days_since} dni od ostatniego. Dobry moment na powrót!"
                )
        
        # Sezonowość
        seasonal = analysis.get("seasonal_patterns", {})
        if seasonal.get("best_month"):
            current_month = datetime.now().strftime("%B")
            months_pl = {
                "January": "Styczeń", "February": "Luty", "March": "Marzec",
                "April": "Kwiecień", "May": "Maj", "June": "Czerwiec",
                "July": "Lipiec", "August": "Sierpień", "September": "Wrzesień",
                "October": "Październik", "November": "Listopad", "December": "Grudzień"
            }
            current_pl = months_pl.get(current_month, current_month)
            if current_pl == seasonal["best_month"]:
                recs.append(
                    f"🎯 {current_pl} to historycznie Twój najlepszy miesiąc - wykorzystaj go!"
                )
        
        if not recs:
            recs.append("📊 Brak wyraźnych wzorców timingowych - publikuj regularnie")
        
        return recs


# =============================================================================
# 4. COMPETITION SCANNER
# =============================================================================

class CompetitionScanner:
    """
    Skanuje konkurencję na YouTube dla danego tematu.
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 3600  # 1 godzina
    
    def scan_competition(self, query: str, days: int = 30) -> Dict:
        """
        Skanuje YouTube w poszukiwaniu konkurencyjnych filmów.
        
        Args:
            query: Temat do wyszukania
            days: Ile dni wstecz sprawdzać
            
        Returns:
            Dict z analizą konkurencji
        """
        cache_key = f"{query}_{days}"
        
        # Sprawdź cache
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if (datetime.now() - cached["timestamp"]).seconds < self.cache_ttl:
                return cached["data"]
        
        if not YT_SEARCH_AVAILABLE:
            return self._mock_competition_response(query)
        
        try:
            # Szukaj filmów
            search = VideosSearch(query, limit=20, language="pl", region="PL")
            results = search.result()
            
            videos = []
            recent_count = 0
            total_views = 0
            
            for item in results.get("result", []):
                video = self._parse_video(item)
                videos.append(video)
                
                # Sprawdź czy recent
                if video.get("days_ago", 999) <= days:
                    recent_count += 1
                
                total_views += video.get("views", 0)
            
            # Analiza
            analysis = self._analyze_competition(videos, recent_count, days, query)
            
            result = {
                "status": "OK",
                "query": query,
                "videos_found": len(videos),
                "recent_videos": recent_count,
                "days_checked": days,
                "analysis": analysis,
                "top_videos": videos[:5]
            }
            
            # Cache
            self.cache[cache_key] = {"timestamp": datetime.now(), "data": result}
            
            return result
        
        except TypeError as e:
            # Znany bug: youtube-search-python + nowszy httpx
            if "proxies" in str(e):
                return {
                    "status": "LIBRARY_BUG",
                    "message": "Błąd kompatybilności youtube-search-python z httpx. Napraw: pip install httpx==0.24.1",
                    "query": query,
                    "analysis": {
                        "saturation": "UNKNOWN",
                        "emoji": "⚠️",
                        "message": "Nie można sprawdzić konkurencji (błąd biblioteki)",
                        "score_bonus": 0
                    }
                }
            raise
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": str(e),
                "query": query,
                "analysis": {
                    "saturation": "UNKNOWN",
                    "emoji": "❓",
                    "message": "Nie można sprawdzić konkurencji",
                    "score_bonus": 0
                }
            }
    
    def _parse_video(self, item: Dict) -> Dict:
        """Parsuje wynik wyszukiwania"""
        # Parsuj views
        views_str = item.get("viewCount", {}).get("short", "0")
        views = self._parse_views(views_str)
        
        # Parsuj datę
        published = item.get("publishedTime", "")
        days_ago = self._parse_days_ago(published)
        
        return {
            "title": item.get("title", ""),
            "channel": item.get("channel", {}).get("name", ""),
            "views": views,
            "views_str": views_str,
            "duration": item.get("duration", ""),
            "published": published,
            "days_ago": days_ago,
            "link": item.get("link", "")
        }
    
    def _parse_views(self, views_str: str) -> int:
        """Parsuje string z views na int"""
        views_str = views_str.lower().replace(" ", "").replace(",", ".")
        
        multipliers = {
            "k": 1000, "tys": 1000,
            "m": 1000000, "mln": 1000000,
            "b": 1000000000, "mld": 1000000000
        }
        
        for suffix, mult in multipliers.items():
            if suffix in views_str:
                try:
                    num = float(views_str.replace(suffix, ""))
                    return int(num * mult)
                except ValueError:
                    continue

        try:
            return int(float(views_str))
        except ValueError:
            return 0
    
    def _parse_days_ago(self, published: str) -> int:
        """Parsuje 'X days ago' na int"""
        published = published.lower()
        
        patterns = {
            r"(\d+)\s*(sekund|second)": lambda x: 0,
            r"(\d+)\s*(minut|minute)": lambda x: 0,
            r"(\d+)\s*(godzin|hour)": lambda x: 0,
            r"(\d+)\s*(dni|day)": lambda x: int(x),
            r"(\d+)\s*(tygodn|week)": lambda x: int(x) * 7,
            r"(\d+)\s*(miesiąc|miesięc|month)": lambda x: int(x) * 30,
            r"(\d+)\s*(rok|year|lat)": lambda x: int(x) * 365,
        }
        
        for pattern, calc in patterns.items():
            match = re.search(pattern, published)
            if match:
                return calc(match.group(1))
        
        return 999  # Unknown
    
    def _analyze_competition(
        self, 
        videos: List[Dict], 
        recent_count: int, 
        days: int,
        query: str
    ) -> Dict:
        """Analizuje poziom konkurencji"""
        
        if recent_count == 0:
            saturation = "BLUE_OCEAN"
            emoji = "🟢"
            message = "Brak konkurencji w ostatnich dniach - świetny moment!"
            score_bonus = 15
        elif recent_count <= 2:
            saturation = "LOW_COMPETITION"
            emoji = "🟢"
            message = "Niska konkurencja - dobra okazja"
            score_bonus = 10
        elif recent_count <= 5:
            saturation = "MODERATE"
            emoji = "🟡"
            message = "Umiarkowana konkurencja - wyróżnij się packagingiem"
            score_bonus = 0
        elif recent_count <= 10:
            saturation = "HIGH_COMPETITION"
            emoji = "🟠"
            message = "Wysoka konkurencja - potrzebujesz unikalnego kąta"
            score_bonus = -5
        else:
            saturation = "OVERSATURATED"
            emoji = "🔴"
            message = "Temat przesycony - rozważ inny kąt lub poczekaj"
            score_bonus = -15
        
        # Sprawdź czy jest viral hit
        viral_threshold = 100000
        virals = [v for v in videos if v["views"] > viral_threshold and v["days_ago"] <= days]
        
        viral_info = None
        if virals:
            top_viral = max(virals, key=lambda x: x["views"])
            viral_info = {
                "title": top_viral["title"],
                "channel": top_viral["channel"],
                "views": top_viral["views"],
                "message": "🔥 Jest viral hit - możesz jechać na fali lub dać kontr-perspektywę"
            }
            score_bonus += 5  # Viral = zainteresowanie tematem
        
        # Średnie views konkurencji
        avg_views = np.mean([v["views"] for v in videos]) if videos else 0
        
        return {
            "saturation": saturation,
            "emoji": emoji,
            "message": message,
            "score_bonus": score_bonus,
            "recent_videos_count": recent_count,
            "avg_competitor_views": int(avg_views),
            "viral_hit": viral_info
        }
    
    def _mock_competition_response(self, query: str) -> Dict:
        """Mock response gdy youtube-search-python niedostępne"""
        return {
            "status": "MOCK",
            "message": "YouTube search niedostępne - zainstaluj: pip install youtube-search-python",
            "query": query,
            "analysis": {
                "saturation": "UNKNOWN",
                "emoji": "❓",
                "message": "Nie można sprawdzić konkurencji",
                "score_bonus": 0
            }
        }


# =============================================================================
# 5. PACKAGING DNA - PATTERN MINING
# =============================================================================

class PackagingDNA:
    """
    Wyciąga wzorce z hitów kanału - "DNA" skutecznego packagingu.
    """
    
    def __init__(self):
        self.title_patterns = None
        self.word_correlations = None
        self.structure_patterns = None
    
    def extract_dna(self, df: pd.DataFrame) -> Dict:
        """
        Analizuje dane kanału i wyciąga wzorce.
        
        Returns:
            Dict z "DNA" packagingu kanału
        """
        if "title" not in df.columns:
            return {"error": "Brak kolumny title"}
        
        if "views" not in df.columns and "label" not in df.columns:
            return {"error": "Potrzebna kolumna views lub label"}
        
        # Podziel na PASS/FAIL jeśli nie ma labelek
        if "label" not in df.columns:
            median_views = df["views"].median()
            df["_label"] = df["views"].apply(lambda x: "PASS" if x > median_views else "FAIL")
        else:
            df["_label"] = df["label"]
        
        results = {
            "title_length": self._analyze_title_length(df),
            "word_triggers": self._find_trigger_words(df),
            "title_structures": self._find_title_structures(df),
            "punctuation_patterns": self._analyze_punctuation(df),
            "emotion_patterns": self._analyze_emotions(df),
            "recommendations": []
        }
        
        results["recommendations"] = self._generate_dna_recommendations(results)
        
        return results
    
    def _analyze_title_length(self, df: pd.DataFrame) -> Dict:
        """Analizuje optymalna długość tytułu"""
        df["_title_len"] = df["title"].apply(lambda x: len(str(x)))
        df["_title_words"] = df["title"].apply(lambda x: len(str(x).split()))
        
        pass_df = df[df["_label"] == "PASS"]
        fail_df = df[df["_label"] == "FAIL"]
        
        return {
            "pass_avg_chars": round(pass_df["_title_len"].mean(), 1),
            "fail_avg_chars": round(fail_df["_title_len"].mean(), 1),
            "pass_avg_words": round(pass_df["_title_words"].mean(), 1),
            "fail_avg_words": round(fail_df["_title_words"].mean(), 1),
            "optimal_chars_range": (
                int(pass_df["_title_len"].quantile(0.25)),
                int(pass_df["_title_len"].quantile(0.75))
            ),
            "optimal_words_range": (
                int(pass_df["_title_words"].quantile(0.25)),
                int(pass_df["_title_words"].quantile(0.75))
            )
        }
    
    def _find_trigger_words(self, df: pd.DataFrame) -> Dict:
        """Znajduje słowa które korelują z sukcesem"""
        # Tokenizuj tytuły
        pass_words = Counter()
        fail_words = Counter()
        
        for _, row in df.iterrows():
            words = re.findall(r'\b[A-Za-zżźćńółęąśŻŹĆĄŚĘÓŁŃ]{3,}\b', str(row["title"]).lower())
            if row["_label"] == "PASS":
                pass_words.update(words)
            else:
                fail_words.update(words)
        
        # Oblicz lift dla każdego słowa
        all_words = set(pass_words.keys()) | set(fail_words.keys())
        pass_total = sum(pass_words.values()) or 1
        fail_total = sum(fail_words.values()) or 1
        
        word_lift = {}
        for word in all_words:
            p_rate = pass_words.get(word, 0) / pass_total
            f_rate = fail_words.get(word, 0) / fail_total
            
            # Tylko słowa z min 2 wystąpieniami
            if pass_words.get(word, 0) + fail_words.get(word, 0) >= 2:
                lift = (p_rate + 0.001) / (f_rate + 0.001)
                word_lift[word] = {
                    "pass_count": pass_words.get(word, 0),
                    "fail_count": fail_words.get(word, 0),
                    "lift": round(lift, 2)
                }
        
        # Top trigger words (wysoki lift = częściej w PASS)
        sorted_words = sorted(word_lift.items(), key=lambda x: x[1]["lift"], reverse=True)
        
        top_triggers = sorted_words[:15]
        avoid_words = sorted_words[-10:]
        
        return {
            "trigger_words": [{"word": w, **d} for w, d in top_triggers],
            "avoid_words": [{"word": w, **d} for w, d in avoid_words],
        }
    
    def _find_title_structures(self, df: pd.DataFrame) -> Dict:
        """Wykrywa struktury tytułów które działają"""
        structures = {
            "question": r"\?$",
            "colon_structure": r":.+",
            "parenthesis": r"\([^)]+\)",
            "number_start": r"^\d+",
            "number_anywhere": r"\d+",
            "ellipsis": r"\.{3}|…",
            "caps_word": r"\b[A-ZĄĆĘŁŃÓŚŹŻ]{2,}\b",
            "quote": r'["\']',
        }
        
        pass_df = df[df["_label"] == "PASS"]
        fail_df = df[df["_label"] == "FAIL"]
        
        results = {}
        for name, pattern in structures.items():
            pass_match = pass_df["title"].apply(lambda x: bool(re.search(pattern, str(x)))).mean()
            fail_match = fail_df["title"].apply(lambda x: bool(re.search(pattern, str(x)))).mean()
            
            results[name] = {
                "pass_rate": round(pass_match * 100, 1),
                "fail_rate": round(fail_match * 100, 1),
                "lift": round((pass_match + 0.01) / (fail_match + 0.01), 2)
            }
        
        # Sortuj po lift
        sorted_structures = sorted(results.items(), key=lambda x: x[1]["lift"], reverse=True)
        
        return {
            "structures": dict(sorted_structures),
            "best_structures": [s[0] for s in sorted_structures if s[1]["lift"] > 1.2][:3],
            "avoid_structures": [s[0] for s in sorted_structures if s[1]["lift"] < 0.8]
        }
    
    def _analyze_punctuation(self, df: pd.DataFrame) -> Dict:
        """Analizuje użycie interpunkcji"""
        pass_df = df[df["_label"] == "PASS"]
        fail_df = df[df["_label"] == "FAIL"]
        
        punct_patterns = {
            "question_mark": r"\?",
            "exclamation": r"!",
            "colon": r":",
            "ellipsis": r"\.{3}|…",
            "parenthesis": r"\(",
        }
        
        results = {}
        for name, pattern in punct_patterns.items():
            pass_rate = pass_df["title"].apply(lambda x: bool(re.search(pattern, str(x)))).mean()
            fail_rate = fail_df["title"].apply(lambda x: bool(re.search(pattern, str(x)))).mean()
            
            results[name] = {
                "pass_rate": round(pass_rate * 100, 1),
                "fail_rate": round(fail_rate * 100, 1),
            }
        
        return results
    
    def _analyze_emotions(self, df: pd.DataFrame) -> Dict:
        """Analizuje emocje w tytułach"""
        emotion_words = {
            "strach": ["śmierć", "zginął", "zabił", "przerażając", "koszmar", "makabryczn"],
            "tajemnica": ["tajemnic", "zagadk", "niewyjaśnion", "ukryt", "sekret"],
            "szok": ["szok", "niewiarygod", "niesamowit", "zdumiewając"],
            "gniew": ["skandal", "afera", "oszust", "kłamst", "zdrad"],
            "smutek": ["tragedi", "dramat", "rozpaczy", "samotność"],
        }
        
        pass_df = df[df["_label"] == "PASS"]
        fail_df = df[df["_label"] == "FAIL"]
        
        results = {}
        for emotion, words in emotion_words.items():
            pattern = "|".join(words)
            pass_rate = pass_df["title"].apply(
                lambda x: bool(re.search(pattern, str(x).lower()))
            ).mean()
            fail_rate = fail_df["title"].apply(
                lambda x: bool(re.search(pattern, str(x).lower()))
            ).mean()
            
            results[emotion] = {
                "pass_rate": round(pass_rate * 100, 1),
                "fail_rate": round(fail_rate * 100, 1),
                "lift": round((pass_rate + 0.01) / (fail_rate + 0.01), 2)
            }
        
        best_emotion = max(results.items(), key=lambda x: x[1]["lift"])
        
        return {
            "emotions": results,
            "best_emotion": best_emotion[0],
            "best_emotion_lift": best_emotion[1]["lift"]
        }
    
    def _generate_dna_recommendations(self, analysis: Dict) -> List[str]:
        """Generuje rekomendacje na podstawie DNA"""
        recs = []
        
        # Długość tytułu
        title_len = analysis.get("title_length", {})
        if title_len.get("optimal_chars_range"):
            low, high = title_len["optimal_chars_range"]
            recs.append(f"📏 Optymalna długość tytułu: {low}-{high} znaków")
        
        # Trigger words
        triggers = analysis.get("word_triggers", {}).get("trigger_words", [])
        if triggers:
            top_words = [t["word"] for t in triggers[:5]]
            recs.append(f"🎯 Słowa-triggery z Twoich hitów: {', '.join(top_words)}")
        
        # Struktury
        structures = analysis.get("title_structures", {}).get("best_structures", [])
        structure_names = {
            "question": "pytanie (?)",
            "colon_structure": "dwukropek (:)",
            "number_anywhere": "liczby",
            "caps_word": "CAPS",
            "parenthesis": "nawias ()"
        }
        if structures:
            struct_pl = [structure_names.get(s, s) for s in structures]
            recs.append(f"🏗️ Skuteczne struktury: {', '.join(struct_pl)}")
        
        # Emocje
        emotions = analysis.get("emotion_patterns", {})
        if emotions.get("best_emotion"):
            recs.append(f"💫 Najskuteczniejsza emocja: {emotions['best_emotion']} (lift: {emotions['best_emotion_lift']}x)")
        
        return recs


# =============================================================================
# GŁÓWNA KLASA ŁĄCZĄCA WSZYSTKO
# =============================================================================

class PromiseGenerator:
    """
    Adapter dla PromiseGenerator z topic_analyzer.
    Zachowuje wsteczną kompatybilność z metodą generate_from_title().
    """

    def __init__(self, openai_client=None):
        self.client = openai_client
        # Użyj implementacji z topic_analyzer jeśli dostępna
        if _TOPIC_ANALYZER_AVAILABLE and _TopicPromiseGenerator is not None:
            self._impl = _TopicPromiseGenerator(openai_client)
        else:
            self._impl = None

    def generate_from_title(self, title: str, n: int = 5, use_ai: bool = True) -> List[Dict]:
        """
        Generuje propozycje obietnic na podstawie tytułu.

        Returns:
            Lista {promise, score, style/source}
        """
        if self._impl is not None:
            # Deleguj do implementacji z topic_analyzer
            # Metoda generate() przyjmuje: title, topic, n, use_ai
            results = self._impl.generate(title=title, topic=title, n=n, use_ai=use_ai)
            # Mapuj 'source' -> 'style' dla wstecznej kompatybilności
            for r in results:
                if 'source' in r and 'style' not in r:
                    r['style'] = r['source']
            return results
        else:
            # Fallback - prosta implementacja bez AI
            return self._fallback_generate(title, n)

    def _fallback_generate(self, title: str, n: int) -> List[Dict]:
        """Prosta implementacja bez zależności od topic_analyzer"""
        templates = [
            "To, co odkryjesz, zmieni Twoje postrzeganie tego tematu.",
            "Historia, która przez lata była ukrywana przed opinią publiczną.",
            "Co naprawdę kryje się za oficjalną wersją wydarzeń?",
            "Fakty, które sprawią, że już nigdy nie spojrzysz na to tak samo.",
            "Prawda jest znacznie mroczniejsza niż oficjalna wersja.",
        ]
        return [{"promise": t, "score": 60, "style": "template"} for t in templates[:n]]


class ABTitleTester:
    """Testuje który tytuł ma większy potencjał CTR"""
    
    CTR_FACTORS = {
        "number_in_title": 1.15,
        "question_mark": 1.10,
        "caps_word": 1.08,
        "emotional_word": 1.20,
        "mystery_word": 1.18,
        "short_title": 1.05,  # < 50 chars
        "colon_structure": 1.12,
    }
    
    EMOTIONAL_WORDS = [
        "szok", "niesamowit", "niewiarygod", "przerażając", "tragedi",
        "tajemnic", "ukryt", "prawda", "sekret", "skandal", "śmierć"
    ]
    
    MYSTERY_WORDS = [
        "dlaczego", "jak", "kto", "co naprawdę", "prawda o", "tajemnica",
        "niewyjaśnion", "zagadk", "zaginion"
    ]
    
    def __init__(self, channel_data: pd.DataFrame = None):
        self.channel_data = channel_data
        self.learned_patterns = {}
        if channel_data is not None:
            self._learn_patterns()
    
    def _learn_patterns(self):
        """Uczy się wzorców z danych kanału"""
        if self.channel_data is None or 'title' not in self.channel_data.columns:
            return
        
        df = self.channel_data.copy()
        if 'views' not in df.columns:
            return
        
        median_views = df['views'].median()
        df['is_hit'] = df['views'] > median_views
        
        # Analizuj patterns
        import re
        
        for pattern_name, check_func in [
            ("number", lambda t: bool(re.search(r'\d+', t))),
            ("question", lambda t: '?' in t),
            ("colon", lambda t: ':' in t),
            ("caps", lambda t: bool(re.search(r'\b[A-Z]{2,}\b', t))),
            ("emotional", lambda t: any(w in t.lower() for w in self.EMOTIONAL_WORDS)),
            ("mystery", lambda t: any(w in t.lower() for w in self.MYSTERY_WORDS)),
        ]:
            has_pattern = df['title'].apply(check_func)
            hit_rate_with = df[has_pattern]['is_hit'].mean() if has_pattern.sum() > 0 else 0.5
            hit_rate_without = df[~has_pattern]['is_hit'].mean() if (~has_pattern).sum() > 0 else 0.5
            
            if hit_rate_without > 0:
                self.learned_patterns[pattern_name] = hit_rate_with / hit_rate_without
    
    def compare(self, title_a: str, title_b: str) -> Dict:
        """
        Porównuje dwa tytuły i przewiduje który lepszy.
        
        Returns:
            Dict z analizą i rekomendacją
        """
        score_a = self._calculate_ctr_score(title_a)
        score_b = self._calculate_ctr_score(title_b)
        
        analysis_a = self._analyze_title(title_a)
        analysis_b = self._analyze_title(title_b)
        
        winner = "A" if score_a > score_b else "B" if score_b > score_a else "TIE"
        confidence = abs(score_a - score_b) / max(score_a, score_b) * 100
        
        return {
            "title_a": {
                "title": title_a,
                "score": round(score_a, 1),
                "factors": analysis_a
            },
            "title_b": {
                "title": title_b,
                "score": round(score_b, 1),
                "factors": analysis_b
            },
            "winner": winner,
            "confidence": round(confidence, 1),
            "recommendation": self._get_recommendation(title_a, title_b, analysis_a, analysis_b)
        }
    
    def _calculate_ctr_score(self, title: str) -> float:
        """Oblicza score CTR dla tytułu"""
        import re
        
        base_score = 50.0
        
        # Sprawdź faktory
        if re.search(r'\d+', title):
            factor = self.learned_patterns.get("number", self.CTR_FACTORS["number_in_title"])
            base_score *= factor
        
        if '?' in title:
            factor = self.learned_patterns.get("question", self.CTR_FACTORS["question_mark"])
            base_score *= factor
        
        if re.search(r'\b[A-Z]{2,}\b', title):
            factor = self.learned_patterns.get("caps", self.CTR_FACTORS["caps_word"])
            base_score *= factor
        
        if any(w in title.lower() for w in self.EMOTIONAL_WORDS):
            factor = self.learned_patterns.get("emotional", self.CTR_FACTORS["emotional_word"])
            base_score *= factor
        
        if any(w in title.lower() for w in self.MYSTERY_WORDS):
            factor = self.learned_patterns.get("mystery", self.CTR_FACTORS["mystery_word"])
            base_score *= factor
        
        if ':' in title:
            factor = self.learned_patterns.get("colon", self.CTR_FACTORS["colon_structure"])
            base_score *= factor
        
        if len(title) < 50:
            base_score *= self.CTR_FACTORS["short_title"]
        
        return min(100, base_score)
    
    def _analyze_title(self, title: str) -> List[str]:
        """Analizuje co tytuł ma/nie ma"""
        import re
        
        factors = []
        
        if re.search(r'\d+', title):
            factors.append("✅ Zawiera liczbę")
        else:
            factors.append("❌ Brak liczby")
        
        if '?' in title:
            factors.append("✅ Pytanie (buduje ciekawość)")
        
        if any(w in title.lower() for w in self.EMOTIONAL_WORDS):
            factors.append("✅ Słowa emocjonalne")
        else:
            factors.append("❌ Brak emocji")
        
        if any(w in title.lower() for w in self.MYSTERY_WORDS):
            factors.append("✅ Element tajemnicy")
        
        if len(title) > 60:
            factors.append("⚠️ Za długi (>60 znaków)")
        
        return factors
    
    def _get_recommendation(self, title_a: str, title_b: str, 
                           analysis_a: List, analysis_b: List) -> str:
        """Generuje rekomendację"""
        a_good = sum(1 for a in analysis_a if a.startswith("✅"))
        b_good = sum(1 for a in analysis_b if a.startswith("✅"))
        
        if a_good > b_good:
            return f"Tytuł A ma więcej pozytywnych elementów ({a_good} vs {b_good})"
        elif b_good > a_good:
            return f"Tytuł B ma więcej pozytywnych elementów ({b_good} vs {a_good})"
        else:
            return "Oba tytuły są porównywalne - testuj miniaturkami"


class ContentGapFinder:
    """Znajduje luki w contencie - popularne tematy których nie robiłeś"""
    
    NICHE_TOPICS = {
        "dark_doc_pl": [
            "katastrofy lotnicze", "seryjni mordercy", "zaginięcia", "sekty",
            "afery polityczne", "niewyjaśnione sprawy", "wypadki", "oszustwa",
            "teorie spiskowe", "tajne operacje", "katastrofy naturalne",
            "zbrodnie wojenne", "eksperymenty", "mafia", "korupcja"
        ]
    }
    
    def __init__(self, channel_data: pd.DataFrame = None):
        self.channel_data = channel_data
        self.my_topics = set()
        if channel_data is not None:
            self._extract_my_topics()
    
    def _extract_my_topics(self):
        """Wyciąga tematy z moich filmów"""
        if self.channel_data is None or 'title' not in self.channel_data.columns:
            return
        
        for title in self.channel_data['title']:
            title_lower = str(title).lower()
            for topic_list in self.NICHE_TOPICS.values():
                for topic in topic_list:
                    if topic in title_lower:
                        self.my_topics.add(topic)
    
    def find_gaps(self, niche: str = "dark_doc_pl") -> List[Dict]:
        """
        Znajduje tematy których nie robiłeś.
        
        Returns:
            Lista {topic, popularity, your_coverage, recommendation}
        """
        topics = self.NICHE_TOPICS.get(niche, [])
        
        gaps = []
        for topic in topics:
            covered = topic in self.my_topics
            
            # Oblicz coverage
            coverage = 0
            if self.channel_data is not None and 'title' in self.channel_data.columns:
                coverage = sum(1 for t in self.channel_data['title'] 
                              if topic in str(t).lower())
            
            gaps.append({
                "topic": topic,
                "your_videos": coverage,
                "covered": covered,
                "recommendation": "🟢 Nieeksploatowany temat!" if not covered else f"🟡 Masz {coverage} filmów"
            })
        
        # Sortuj - nierobione na górze
        return sorted(gaps, key=lambda x: (x["covered"], x["your_videos"]))
    
    def suggest_ideas(self, gap_topic: str, n: int = 3) -> List[str]:
        """Sugeruje pomysły na filmy dla danego tematu-luki"""
        templates = [
            f"Największa {gap_topic} w historii Polski",
            f"5 niewyjaśnionych przypadków: {gap_topic}",
            f"{gap_topic.title()}: Co naprawdę się wydarzyło?",
            f"Dlaczego nikt nie mówi o tej {gap_topic}?",
            f"Mroczna historia: {gap_topic} która wstrząsnęła światem",
        ]
        return templates[:n]


class WtopaAnalyzer:
    """Analizuje dlaczego film nie wystrzelił"""
    
    def __init__(self, openai_client=None, channel_data: pd.DataFrame = None):
        self.client = openai_client
        self.channel_data = channel_data
        self.benchmarks = self._calculate_benchmarks()
    
    def _calculate_benchmarks(self) -> Dict:
        """Oblicza benchmarki kanału"""
        if self.channel_data is None:
            return {}
        
        df = self.channel_data
        
        return {
            "median_views": df['views'].median() if 'views' in df.columns else 0,
            "avg_views": df['views'].mean() if 'views' in df.columns else 0,
            "median_retention": df['retention'].median() if 'retention' in df.columns else 0,
        }
    
    def analyze(self, title: str, views: int, retention: float = None, 
                script: str = None) -> Dict:
        """
        Analizuje dlaczego film nie wystrzelił.
        
        Args:
            title: Tytuł filmu
            views: Rzeczywiste views
            retention: Rzeczywista retencja
            script: Pełny skrypt (opcjonalnie)
            
        Returns:
            Dict z analizą i rekomendacjami
        """
        analysis = {
            "title": title,
            "views": views,
            "retention": retention,
            "problems": [],
            "recommendations": [],
            "verdict": "",
        }
        
        # 1. Porównaj z benchmarkami
        median_views = self.benchmarks.get("median_views", 50000)
        
        performance = views / median_views if median_views else 0
        
        if performance < 0.3:
            analysis["verdict"] = "🔴 DUŻO PONIŻEJ ŚREDNIEJ"
        elif performance < 0.7:
            analysis["verdict"] = "🟠 PONIŻEJ ŚREDNIEJ"
        elif performance < 1.0:
            analysis["verdict"] = "🟡 LEKKO PONIŻEJ"
        else:
            analysis["verdict"] = "🟢 OK (nie jest wtopą)"
            return analysis
        
        # 2. Analiza tytułu
        title_problems = self._analyze_title_problems(title)
        analysis["problems"].extend(title_problems)
        
        # 3. Analiza retencji
        if retention:
            retention_problems = self._analyze_retention_problems(retention)
            analysis["problems"].extend(retention_problems)
        
        # 4. AI analiza (jeśli dostępna)
        if self.client:
            ai_analysis = self._ai_analyze(title, views, retention, script)
            if ai_analysis:
                analysis["ai_diagnosis"] = ai_analysis.get("diagnosis", "")
                analysis["recommendations"].extend(ai_analysis.get("recommendations", []))
        
        # 5. Generuj rekomendacje
        if not analysis["recommendations"]:
            analysis["recommendations"] = self._generate_recommendations(analysis["problems"])
        
        return analysis
    
    def _analyze_title_problems(self, title: str) -> List[str]:
        """Analizuje problemy z tytułem"""
        problems = []
        
        if len(title) > 70:
            problems.append("📏 Tytuł za długi (>70 znaków) - może być ucięty")
        
        if len(title) < 20:
            problems.append("📏 Tytuł za krótki - brak kontekstu")
        
        import re
        if not re.search(r'\d+', title):
            problems.append("🔢 Brak liczby w tytule - mniej konkretny")
        
        emotional_words = ["szok", "niesamowit", "przerażając", "tajemnic", "prawda"]
        if not any(w in title.lower() for w in emotional_words):
            problems.append("😐 Brak emocjonalnych słów - tytuł może być nudny")
        
        if title[0].islower():
            problems.append("📝 Tytuł nie zaczyna się wielką literą")
        
        return problems
    
    def _analyze_retention_problems(self, retention: float) -> List[str]:
        """Analizuje problemy z retencją"""
        problems = []
        
        median_ret = self.benchmarks.get("median_retention", 35)
        
        if retention < 20:
            problems.append("📉 BARDZO NISKA retencja (<20%) - problem z hookiem lub całym contentem")
        elif retention < 30:
            problems.append("📉 Niska retencja (<30%) - widzowie odchodzą za wcześnie")
        elif retention < median_ret * 0.8:
            problems.append(f"📉 Retencja poniżej średniej kanału ({retention:.0f}% vs {median_ret:.0f}%)")
        
        return problems
    
    def _ai_analyze(self, title: str, views: int, retention: float, script: str) -> Optional[Dict]:
        """AI analiza wtopy"""
        if not self.client:
            return None
        
        try:
            script_part = script[:3000] if script else "Brak skryptu"
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """Jesteś ekspertem od YouTube dark documentaries.
Analizujesz dlaczego film nie odniósł sukcesu.

Bądź konkretny i bezpośredni. Wskaż dokładne problemy."""},
                    {"role": "user", "content": f"""Przeanalizuj ten film który nie wystrzelił:

Tytuł: {title}
Views: {views:,}
Retencja: {retention}%

Fragment skryptu:
{script_part}

Odpowiedz w JSON:
{{"diagnosis": "Główny problem w 1-2 zdaniach", "recommendations": ["konkretna rada 1", "konkretna rada 2", "konkretna rada 3"]}}"""}
                ],
                response_format={"type": "json_object"}
            )
            
            import json
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"AI analysis error: {e}")
            return None
    
    def _generate_recommendations(self, problems: List[str]) -> List[str]:
        """Generuje rekomendacje na podstawie problemów"""
        recs = []
        
        for problem in problems:
            if "długi" in problem:
                recs.append("Skróć tytuł do max 60 znaków")
            if "liczby" in problem.lower():
                recs.append("Dodaj konkretną liczbę (rok, ilość ofiar, dni)")
            if "emocj" in problem.lower():
                recs.append("Dodaj emocjonalne słowo: tajemnica, szok, prawda")
            if "retencja" in problem.lower():
                recs.append("Przepracuj pierwsze 30 sekund - mocniejszy hook")
            if "hook" in problem.lower():
                recs.append("Zacznij od środka akcji, nie od wprowadzenia")
        
        return recs[:5]


class SeriesAnalyzer:
    """Analizuje które serie/tematy działają najlepiej"""
    
    def __init__(self, channel_data: pd.DataFrame):
        self.channel_data = channel_data
        self.series = self._detect_series()
    
    def _detect_series(self) -> Dict[str, List[Dict]]:
        """Wykrywa serie na podstawie tytułów"""
        if self.channel_data is None or 'title' not in self.channel_data.columns:
            return {}
        
        import re
        
        series = {}
        
        for _, row in self.channel_data.iterrows():
            title = str(row.get('title', ''))
            views = row.get('views', 0)
            
            # Szukaj wzorców serii
            patterns = [
                r'^(.+?)\s*[#|:]\s*(\d+)',  # "Seria: 1" lub "Seria #1"
                r'^(.+?)\s+cz\.?\s*(\d+)',  # "Seria cz. 1"
                r'^(.+?)\s+część\s+(\d+)',  # "Seria część 1"
                r'^(.+?)\s+odc\.?\s*(\d+)',  # "Seria odc 1"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, title, re.IGNORECASE)
                if match:
                    series_name = match.group(1).strip()
                    episode = int(match.group(2))
                    
                    if series_name not in series:
                        series[series_name] = []
                    
                    series[series_name].append({
                        'title': title,
                        'episode': episode,
                        'views': views,
                    })
                    break
        
        return series
    
    def get_series_performance(self) -> List[Dict]:
        """Zwraca performance każdej serii"""
        results = []
        
        for name, episodes in self.series.items():
            if len(episodes) < 2:
                continue  # Nie jest serią
            
            views = [e['views'] for e in episodes]
            
            results.append({
                'name': name,
                'episodes': len(episodes),
                'total_views': sum(views),
                'avg_views': sum(views) // len(views),
                'best_episode': max(episodes, key=lambda x: x['views']),
                'worst_episode': min(episodes, key=lambda x: x['views']),
                'trend': self._calculate_trend(episodes),
            })
        
        return sorted(results, key=lambda x: x['avg_views'], reverse=True)
    
    def _calculate_trend(self, episodes: List[Dict]) -> str:
        """Oblicza trend serii"""
        if len(episodes) < 2:
            return "—"
        
        sorted_eps = sorted(episodes, key=lambda x: x['episode'])
        first_half = sorted_eps[:len(sorted_eps)//2]
        second_half = sorted_eps[len(sorted_eps)//2:]
        
        avg_first = sum(e['views'] for e in first_half) / len(first_half)
        avg_second = sum(e['views'] for e in second_half) / len(second_half)
        
        change = (avg_second - avg_first) / avg_first * 100 if avg_first else 0
        
        if change > 20:
            return "📈 Rośnie"
        elif change < -20:
            return "📉 Spada"
        else:
            return "➡️ Stabilna"
    
    def get_recommendations(self) -> List[str]:
        """Rekomendacje na podstawie analizy serii"""
        recs = []
        
        perf = self.get_series_performance()
        
        if perf:
            best = perf[0]
            recs.append(f"🏆 Najlepsza seria: '{best['name']}' (avg {best['avg_views']:,} views)")
            
            for s in perf:
                if s['trend'] == "📈 Rośnie":
                    recs.append(f"📈 Kontynuuj '{s['name']}' - trend rosnący")
                elif s['trend'] == "📉 Spada" and s['episodes'] > 3:
                    recs.append(f"⚠️ Rozważ zakończenie '{s['name']}' - trend spadkowy")
        
        return recs


class AdvancedAnalytics:
    """
    Główna klasa łącząca wszystkie moduły analityczne.
    """
    
    def __init__(self, openai_client=None):
        self.hook_analyzer = HookAnalyzer()
        self.trends_analyzer = TrendsAnalyzer()
        self.timing_predictor = TimingPredictor()
        self.competition_scanner = CompetitionScanner()
        self.packaging_dna = PackagingDNA()
        
        # Nowe moduły
        self.promise_generator = PromiseGenerator(openai_client)
        self.ab_tester = None  # Wymaga danych
        self.content_gap_finder = None  # Wymaga danych
        self.wtopa_analyzer = WtopaAnalyzer(openai_client)
        self.series_analyzer = None  # Wymaga danych
        
        self._df = None
        self._dna_cache = None
        self._timing_cache = None
        self._client = openai_client
    
    def load_data(self, df: pd.DataFrame):
        """Ładuje dane kanału"""
        self._df = df
        self._dna_cache = None
        self._timing_cache = None
        
        # Inicjalizuj moduły wymagające danych
        self.ab_tester = ABTitleTester(df)
        self.content_gap_finder = ContentGapFinder(df)
        self.wtopa_analyzer = WtopaAnalyzer(self._client, df)
        self.series_analyzer = SeriesAnalyzer(df)
        
        return self
    
    def set_openai_client(self, client):
        """Ustawia klienta OpenAI"""
        self._client = client
        self.promise_generator = PromiseGenerator(client)
        self.wtopa_analyzer = WtopaAnalyzer(client, self._df)
    
    def get_packaging_dna(self, force_refresh: bool = False) -> Dict:
        """Zwraca DNA packagingu (cache'owane)"""
        if self._dna_cache is None or force_refresh:
            if self._df is not None:
                self._dna_cache = self.packaging_dna.extract_dna(self._df)
            else:
                self._dna_cache = {"error": "Brak danych"}
        return self._dna_cache
    
    def get_timing_analysis(self, force_refresh: bool = False) -> Dict:
        """Zwraca analizę timingu (cache'owana)"""
        if self._timing_cache is None or force_refresh:
            if self._df is not None:
                self._timing_cache = self.timing_predictor.analyze_timing(self._df)
            else:
                self._timing_cache = {"error": "Brak danych"}
        return self._timing_cache
    
    def analyze_idea(self, title: str, promise: str = "", hook: str = "") -> Dict:
        """
        Pełna analiza pomysłu z wszystkimi modułami.
        
        Returns:
            Dict z wynikami wszystkich analiz
        """
        results = {
            "title": title,
            "promise": promise,
            "analyses": {},
            "total_bonus": 0,
            "recommendations": []
        }
        
        # 1. Hook analysis (jeśli podany)
        if hook:
            hook_result = self.hook_analyzer.analyze_hook(hook)
            results["analyses"]["hook"] = hook_result
            results["recommendations"].extend(hook_result.get("suggestions", []))
        
        # 2. Trends
        keywords = self.trends_analyzer.extract_keywords_from_title(title)
        if keywords:
            trends_result = self.trends_analyzer.check_trend(keywords)
            results["analyses"]["trends"] = trends_result
            results["total_bonus"] += trends_result.get("overall", {}).get("trend_bonus", 0)
        
        # 3. Competition
        search_query = " ".join(keywords[:3]) if keywords else title[:50]
        comp_result = self.competition_scanner.scan_competition(search_query)
        results["analyses"]["competition"] = comp_result
        results["total_bonus"] += comp_result.get("analysis", {}).get("score_bonus", 0)
        
        # 4. DNA check (jeśli mamy dane)
        if self._df is not None:
            dna = self.get_packaging_dna()
            dna_score = self._check_against_dna(title, dna)
            results["analyses"]["dna_match"] = dna_score
            results["total_bonus"] += dna_score.get("bonus", 0)
        
        # 5. Timing (jeśli mamy dane)
        if self._df is not None:
            timing = self.get_timing_analysis()
            results["analyses"]["timing"] = timing
            results["recommendations"].extend(timing.get("recommendations", []))
        
        return results
    
    def _check_against_dna(self, title: str, dna: Dict) -> Dict:
        """Sprawdza tytuł względem DNA kanału"""
        score = 0
        matches = []
        
        # Sprawdź długość
        title_len = dna.get("title_length", {})
        optimal_range = title_len.get("optimal_chars_range", (30, 70))
        if optimal_range[0] <= len(title) <= optimal_range[1]:
            score += 5
            matches.append("✅ Długość w optymalnym zakresie")
        
        # Sprawdź trigger words
        triggers = dna.get("word_triggers", {}).get("trigger_words", [])
        trigger_words = [t["word"] for t in triggers[:10]]
        title_lower = title.lower()
        found_triggers = [w for w in trigger_words if w in title_lower]
        if found_triggers:
            score += len(found_triggers) * 3
            matches.append(f"✅ Zawiera trigger words: {', '.join(found_triggers)}")
        
        # Sprawdź struktury
        best_structures = dna.get("title_structures", {}).get("best_structures", [])
        structures_found = []
        if "question" in best_structures and "?" in title:
            structures_found.append("pytanie")
            score += 5
        if "colon_structure" in best_structures and ":" in title:
            structures_found.append("dwukropek")
            score += 5
        if "number_anywhere" in best_structures and re.search(r'\d', title):
            structures_found.append("liczba")
            score += 5
        
        if structures_found:
            matches.append(f"✅ Używa skutecznych struktur: {', '.join(structures_found)}")
        
        return {
            "bonus": min(score, 20),  # Max 20 punktów
            "matches": matches,
            "score_breakdown": score
        }
    
    def train_hook_analyzer(self) -> Dict:
        """Trenuje hook analyzer na danych kanału"""
        if self._df is None:
            return {"error": "Brak danych"}
        
        # Szukaj kolumny z hookami
        hook_col = None
        for col in ["hook_120s", "hook", "intro"]:
            if col in self._df.columns:
                hook_col = col
                break
        
        if not hook_col:
            return {"error": "Brak kolumny z hookami"}
        
        return self.hook_analyzer.train_on_corpus(self._df, hook_col=hook_col)
    
    # =================
    # NOWE METODY
    # =================
    
    def generate_promises(self, title: str, n: int = 5) -> List[Dict]:
        """Generuje propozycje obietnic"""
        return self.promise_generator.generate_from_title(title, n)
    
    def ab_test_titles(self, title_a: str, title_b: str) -> Dict:
        """Porównuje dwa tytuły"""
        if self.ab_tester:
            return self.ab_tester.compare(title_a, title_b)
        return {"error": "Załaduj dane kanału"}
    
    def find_content_gaps(self) -> List[Dict]:
        """Znajduje luki w contencie"""
        if self.content_gap_finder:
            return self.content_gap_finder.find_gaps()
        return []
    
    def analyze_wtopa(self, title: str, views: int, retention: float = None, 
                      script: str = None) -> Dict:
        """Analizuje dlaczego film nie wystrzelił"""
        return self.wtopa_analyzer.analyze(title, views, retention, script)
    
    def get_series_analysis(self) -> Dict:
        """Zwraca analizę serii"""
        if self.series_analyzer:
            return {
                "series": self.series_analyzer.get_series_performance(),
                "recommendations": self.series_analyzer.get_recommendations()
            }
        return {"error": "Załaduj dane kanału"}
    
    def get_optimal_upload_calendar(self) -> Dict:
        """Generuje optymalny kalendarz publikacji"""
        timing = self.get_timing_analysis()
        gaps = timing.get("topic_gaps", {})
        dow = timing.get("day_of_week", {})
        
        recommendations = []
        
        # Najlepszy dzień
        if dow.get("best_day"):
            recommendations.append({
                "type": "day",
                "recommendation": f"Publikuj w {dow['best_day']}",
                "reason": f"Średnio {dow['improvement_potential']:.0f}% więcej views"
            })
        
        # Tematy do zrobienia
        for topic, data in gaps.items():
            days_since = data.get("days_since_last", 0)
            avg_gap = data.get("avg_gap_days", 30)
            
            if days_since > avg_gap:
                recommendations.append({
                    "type": "topic",
                    "recommendation": f"Czas na film o: {topic}",
                    "reason": f"{days_since} dni od ostatniego (średnia: {avg_gap:.0f})"
                })
        
        return {
            "recommendations": recommendations,
            "best_day": dow.get("best_day"),
            "topic_gaps": gaps
        }
    
    def compare_ideas(self, ideas: List[Dict]) -> Dict:
        """
        Porównuje wiele pomysłów i tworzy ranking.
        
        Args:
            ideas: Lista {title, promise}
            
        Returns:
            Dict z rankingiem i porównaniem
        """
        results = []
        
        for idea in ideas:
            title = idea.get("title", "")
            promise = idea.get("promise", "")
            
            analysis = self.analyze_idea(title, promise)
            
            # Quick score based on DNA + trends + competition
            quick_score = 50 + analysis.get("total_bonus", 0)
            
            # A/B score
            if self.ab_tester:
                ab = self.ab_tester._calculate_ctr_score(title)
                quick_score = (quick_score + ab) / 2
            
            results.append({
                "title": title,
                "promise": promise,
                "score": round(quick_score, 1),
                "bonus": analysis.get("total_bonus", 0),
                "analysis": analysis
            })
        
        # Sortuj po score
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # Dodaj ranking
        for i, r in enumerate(results, 1):
            r["rank"] = i
        
        return {
            "ranking": results,
            "winner": results[0] if results else None,
            "comparison_summary": self._generate_comparison_summary(results)
        }
    
    def _generate_comparison_summary(self, results: List[Dict]) -> str:
        """Generuje podsumowanie porównania"""
        if not results:
            return ""
        
        if len(results) == 1:
            return f"Tylko jeden pomysł do oceny: {results[0]['score']}/100"
        
        winner = results[0]
        runnerup = results[1]
        
        diff = winner["score"] - runnerup["score"]
        
        if diff > 15:
            return f"🏆 '{winner['title'][:30]}...' wyraźnie wygrywa (+{diff:.0f} pkt)"
        elif diff > 5:
            return f"🥇 '{winner['title'][:30]}...' lekko lepszy (+{diff:.0f} pkt)"
        else:
            return f"🤝 Wyrównany wynik - oba pomysły podobne (różnica {diff:.0f} pkt)"
