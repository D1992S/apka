"""
CONFIG MANAGER
===============
Zarządza konfiguracją i pamięcią lokalną aplikacji.
- Zapamiętuje API keys
- Przechowuje historię ocen
- Zarządza Idea Vault
- Tracking accuracy
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import hashlib


# Ścieżki do plików konfiguracyjnych
CONFIG_DIR = Path("./app_data")
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "evaluation_history.json"
IDEA_VAULT_FILE = CONFIG_DIR / "idea_vault.json"
TRACKING_FILE = CONFIG_DIR / "tracking_accuracy.json"
TREND_ALERTS_FILE = CONFIG_DIR / "trend_alerts.json"
SERIES_MAP_FILE = CONFIG_DIR / "series_map.json"
COMPETITORS_FILE = CONFIG_DIR / "competitors.json"



def ensure_config_dir():
    """Tworzy folder konfiguracyjny jeśli nie istnieje"""
    CONFIG_DIR.mkdir(exist_ok=True)


def _safe_load_json(path: Path, default: Any, label: str):
    """Bezpieczne ładowanie JSON z automatycznym backupem uszkodzonych plików."""
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = path.with_suffix(f".corrupt-{timestamp}.json")
        try:
            path.rename(backup)
            print(
                f"⚠ Nie udało się wczytać {label}: {e}. "
                f"Plik przeniesiono do {backup.name}."
            )
        except OSError:
            print(f"⚠ Nie udało się wczytać {label}: {e}.")
        return default


# =============================================================================
# KONFIGURACJA APLIKACJI
# =============================================================================

class AppConfig:
    """Zarządza konfiguracją aplikacji"""
    
    DEFAULT_CONFIG = {
        "openai_api_key": "",
        "google_ai_api_key": "",
        "openai_enabled": True,
        "google_enabled": True,
        "niche_keywords": ["tajemnice", "zagadki", "spiski", "ufo", "katastrofy"],
        "youtube_credentials_path": "",
        "youtube_api_key": "",
        "auto_sync_on_start": False,
        "dark_mode": True,
        "default_judges": 2,
        "default_topn": 5,
        "default_optimize_variants": False,
        "channel_id": "",
        "last_sync": None,
        "llm_provider": "openai",
        "openai_model": "auto",
        "google_model": "auto",
    }
    
    def __init__(self):
        ensure_config_dir()
        self.config = self._load()

    def _normalize_config(self, saved: Dict) -> Dict:
        """Ujednolica konfigurację i naprawia nieprawidłowe typy."""
        normalized = {**self.DEFAULT_CONFIG, **saved}
        type_expectations = {
            "openai_api_key": str,
            "google_ai_api_key": str,
            "openai_enabled": bool,
            "google_enabled": bool,
            "niche_keywords": list,
            "youtube_credentials_path": str,
            "youtube_api_key": str,
            "auto_sync_on_start": bool,
            "dark_mode": bool,
            "default_judges": int,
            "default_topn": int,
            "default_optimize_variants": bool,
            "channel_id": str,
            "llm_provider": str,
            "openai_model": str,
            "google_model": str,
        }
        for key, expected_type in type_expectations.items():
            value = normalized.get(key)
            if value is None:
                continue
            if not isinstance(value, expected_type):
                print(
                    f"⚠ Nieprawidłowy typ dla '{key}': "
                    f"{type(value).__name__}, przywracam domyślną wartość."
                )
                normalized[key] = self.DEFAULT_CONFIG.get(key)
        return normalized

    def _load(self) -> Dict:
        """Ładuje konfigurację z pliku"""
        saved = _safe_load_json(CONFIG_FILE, {}, "config.json")
        if isinstance(saved, dict):
            return self._normalize_config(saved)
        print("⚠ config.json ma nieprawidłowy format. Używam domyślnych ustawień.")
        return self.DEFAULT_CONFIG.copy()
    
    def save(self):
        """Zapisuje konfigurację do pliku"""
        ensure_config_dir()
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except OSError as e:
            print(f"⚠ Nie udało się zapisać config.json: {e}")
    
    def get(self, key: str, default=None):
        """Pobiera wartość konfiguracji"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Ustawia wartość i zapisuje"""
        self.config[key] = value
        self.save()
    
    def get_api_key(self) -> str:
        """Pobiera OpenAI API key"""
        return self.config.get("openai_api_key", "")
    
    def set_api_key(self, key: str):
        """Zapisuje OpenAI API key"""
        self.set("openai_api_key", key)

    def get_google_api_key(self) -> str:
        """Pobiera Google AI Studio API key"""
        return self.config.get("google_ai_api_key", "")

    def set_google_api_key(self, key: str):
        """Zapisuje Google AI Studio API key"""
        self.set("google_ai_api_key", key)

    def get_youtube_api_key(self) -> str:
        """Pobiera YouTube API key"""
        return self.config.get("youtube_api_key", "")

    def set_youtube_api_key(self, key: str):
        """Zapisuje YouTube API key"""
        self.set("youtube_api_key", key)


# =============================================================================
# HISTORIA OCEN
# =============================================================================

class EvaluationHistory:
    """Zarządza historią wszystkich ocen"""
    
    def __init__(self):
        ensure_config_dir()
        self.history = self._load()
    
    def _load(self) -> List[Dict]:
        """Ładuje historię z pliku"""
        data = _safe_load_json(HISTORY_FILE, [], "evaluation_history.json")
        if isinstance(data, list):
            return data
        print("⚠ evaluation_history.json ma nieprawidłowy format. Używam pustej historii.")
        return []
    
    def save(self):
        """Zapisuje historię do pliku"""
        ensure_config_dir()
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False, default=str)
        except OSError as e:
            print(f"⚠ Nie udało się zapisać historii: {e}")
    
    def add(self, evaluation: Dict):
        """Dodaje ocenę do historii"""
        # Dodaj timestamp i ID
        entry = {
            "id": hashlib.md5(f"{evaluation.get('title', '')}{datetime.now().isoformat()}".encode()).hexdigest()[:8],
            "timestamp": datetime.now().isoformat(),
            "payload": evaluation,
            "title": evaluation.get("title", ""),
            "promise": evaluation.get("promise", ""),
            "final_score": evaluation.get("final_score", 0),
            "final_score_with_bonus": evaluation.get("final_score_with_bonus", evaluation.get("final_score", 0)),
            "final_verdict": evaluation.get("final_verdict", ""),
            "data_score": evaluation.get("data_score", 0),
            "llm_score": evaluation.get("llm_score", 0),
            "risk_penalty": evaluation.get("risk_penalty", 0),
            "risk_flags": evaluation.get("risk_flags", []),
            "advanced_bonus": evaluation.get("advanced_bonus", 0),
            "dimensions": evaluation.get("dimensions", {}),
            "improvements": evaluation.get("improvements", []),
            "title_variants": evaluation.get("title_variants", []),
            "promise_variants": evaluation.get("promise_variants", []),
            "why": evaluation.get("why", ""),
            "suggested_hook_angle": evaluation.get("suggested_hook_angle", ""),
            "target_emotion": evaluation.get("target_emotion", ""),
            "predicted_metrics": evaluation.get("predicted_metrics", {}),
            "advanced_insights": evaluation.get("advanced_insights", {}),
            "tags": evaluation.get("tags", []),
            "status": evaluation.get("status", ""),
            # Tracking - do uzupełnienia po publikacji
            "published": False,
            "actual_views": None,
            "actual_retention": None,
            "prediction_accuracy": None,
        }
        
        self.history.insert(0, entry)  # Najnowsze na górze
        self.save()
        return entry["id"]
    
    def get_all(self) -> List[Dict]:
        """Zwraca całą historię"""
        return self.history
    
    def get_recent(self, n: int = 20) -> List[Dict]:
        """Zwraca ostatnie N ocen"""
        return self.history[:n]
    
    def get_by_id(self, eval_id: str) -> Optional[Dict]:
        """Pobiera ocenę po ID"""
        for entry in self.history:
            if entry.get("id") == eval_id:
                return entry
        return None
    
    def update_tracking(self, eval_id: str, actual_views: int, actual_retention: float = None):
        """Aktualizuje dane po publikacji"""
        for entry in self.history:
            if entry.get("id") == eval_id:
                entry["published"] = True
                entry["actual_views"] = actual_views
                entry["actual_retention"] = actual_retention
                
                # Oblicz accuracy
                predicted = entry.get("predicted_metrics", {})
                pred_views = predicted.get("views_estimate", 0)
                if pred_views and actual_views:
                    # Accuracy jako % różnicy (zabezpieczenie przed dzieleniem przez 0)
                    max_val = max(pred_views, actual_views, 1)
                    diff = abs(pred_views - actual_views) / max_val
                    entry["prediction_accuracy"] = round((1 - diff) * 100, 1)
                
                self.save()
                return True
        return False
    
    def get_tracking_stats(self) -> Dict:
        """Statystyki accuracy predykcji"""
        published = [e for e in self.history if e.get("published")]
        
        if not published:
            return {"total": 0, "message": "Brak opublikowanych filmów z tracking"}
        
        accuracies = [e["prediction_accuracy"] for e in published if e.get("prediction_accuracy")]
        
        # Analiza PASS vs FAIL
        pass_correct = sum(1 for e in published 
                          if e.get("final_verdict") == "PASS" 
                          and e.get("actual_views", 0) > 50000)
        fail_correct = sum(1 for e in published 
                          if e.get("final_verdict") == "FAIL" 
                          and e.get("actual_views", 0) < 20000)
        
        pass_total = sum(1 for e in published if e.get("final_verdict") == "PASS")
        fail_total = sum(1 for e in published if e.get("final_verdict") == "FAIL")
        
        return {
            "total_tracked": len(published),
            "avg_accuracy": round(sum(accuracies) / len(accuracies), 1) if accuracies else 0,
            "pass_accuracy": round(pass_correct / pass_total * 100, 1) if pass_total else 0,
            "fail_accuracy": round(fail_correct / fail_total * 100, 1) if fail_total else 0,
            "published_entries": published
        }
    
    def search(self, query: str) -> List[Dict]:
        """Wyszukuje w historii"""
        query = query.lower()
        return [
            e for e in self.history 
            if query in e.get("title", "").lower() 
            or query in e.get("promise", "").lower()
        ]
    
    def export_to_csv(self) -> str:
        """Eksportuje historię do CSV"""
        import csv
        import io
        
        output = io.StringIO()
        if not self.history:
            return ""
        
        fieldnames = ["timestamp", "title", "promise", "final_score", "final_verdict", 
                      "data_score", "llm_score", "risk_penalty", "advanced_bonus",
                      "published", "actual_views", "prediction_accuracy"]
        
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(self.history)
        
        return output.getvalue()

    def export_to_json(self) -> str:
        """Eksportuje historię do JSON"""
        return json.dumps(self.history, indent=2, ensure_ascii=False, default=str)

    def delete(self, entry_id: str) -> bool:
        """Usuwa pojedynczy wpis z historii"""
        before = len(self.history)
        self.history = [h for h in self.history if h.get("id") != entry_id]
        if len(self.history) != before:
            self.save()
            return True
        return False

    def clear(self):
        """Czyści całą historię"""
        self.history = []
        self.save()

# =============================================================================
# IDEA VAULT
# =============================================================================


class IdeaVault:
    """Przechowuje pomysły na później"""
    
    def __init__(self):
        ensure_config_dir()
        self.ideas = self._load()
    
    def _load(self) -> List[Dict]:
        """Ładuje vault z pliku"""
        if IDEA_VAULT_FILE.exists():
            try:
                with open(IDEA_VAULT_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                print(f"⚠ Nie udało się wczytać Idea Vault: {e}")
        return []
    
    def save(self):
        """Zapisuje vault do pliku"""
        ensure_config_dir()
        with open(IDEA_VAULT_FILE, "w", encoding="utf-8") as f:
            json.dump(self.ideas, f, indent=2, ensure_ascii=False)
    
    def add(self, title: str, promise: str = "", score: int = 0,
            reason: str = "", tags: List[str] = None, remind_when: str = None,
            topic: str = "", payload: Dict[str, Any] = None, status: str = "new",
            **kwargs):
        """Dodaje pomysł do vault"""
        idea = {
            "id": hashlib.md5(f"{title}{datetime.now().isoformat()}".encode()).hexdigest()[:8],
            "added": datetime.now().isoformat(),
            "title": title,
            "topic": topic,
            "promise": promise,
            "score": score,
            "reason": reason,  # Dlaczego zapisany na później
            "tags": tags or [],
            "remind_when": remind_when,  # "trending", "30_days", "season_winter" etc.
            "status": status or "new",  # new, shortlisted, scripted, used, discarded
            "notes": "",
            "payload": payload or {},
            "extra": kwargs or {},
        }
        self.ideas.append(idea)
        self.save()
        return idea["id"]
    
    def get_all(self, status: str = None) -> List[Dict]:
        """Zwraca pomysły (opcjonalnie filtrowane)"""
        if status:
            if status == "new":
                return [i for i in self.ideas if i.get("status") in ["new", "waiting"]]
            return [i for i in self.ideas if i.get("status") == status]
        return self.ideas
    
    def update_status(self, idea_id: str, status: str, notes: str = None):
        """Aktualizuje status pomysłu"""
        for idea in self.ideas:
            if idea.get("id") == idea_id:
                idea["status"] = status
                if notes:
                    idea["notes"] = notes
                self.save()
                return True
        return False

    def update_metadata(self, idea_id: str, tags: List[str] = None, status: str = None, notes: str = None):
        """Aktualizuje tagi/status/notes pomysłu"""
        for idea in self.ideas:
            if idea.get("id") == idea_id:
                if tags is not None:
                    idea["tags"] = tags
                if status:
                    idea["status"] = status
                if notes is not None:
                    idea["notes"] = notes
                self.save()
                return True
        return False
    
    

    def get_by_id(self, idea_id: str) -> Optional[Dict]:
        """Zwraca pojedynczy pomysł po ID"""
        for idea in self.ideas:
            if idea.get("id") == idea_id:
                return idea
        return None

    def update_payload(self, idea_id: str, payload: Dict[str, Any]) -> bool:
        """Nadpisuje payload (pełny wynik analizy)"""
        for idea in self.ideas:
            if idea.get("id") == idea_id:
                idea["payload"] = payload or {}
                self.save()
                return True
        return False

    def remove(self, idea_id: str) -> bool:
        """Usuwa pomysł"""
        before = len(self.ideas)
        self.ideas = [i for i in self.ideas if i.get("id") != idea_id]
        if len(self.ideas) != before:
            self.save()
            return True
        return False

    def check_reminders(self, trending_topics: List[str] = None) -> List[Dict]:
        """Sprawdza które pomysły powinny być przypomniane"""
        reminders: List[Dict] = []
        trending_topics = trending_topics or []

        for idea in self.ideas:
            if idea.get("status") not in ["waiting", "new"]:
                continue

            remind = idea.get("remind_when", "")

            # Trending reminder
            if remind == "trending" and trending_topics:
                for tag in idea.get("tags", []):
                    if any(tag.lower() in t.lower() for t in trending_topics):
                        reminders.append({**idea, "reminder_reason": f"Temat '{tag}' zaczął trendować!"})
                        break

            # Time-based reminder: np. "30_days"
            elif remind and remind.endswith("_days"):
                try:
                    days = int(remind.replace("_days", ""))
                    added = datetime.fromisoformat(idea["added"])
                    if (datetime.now() - added).days >= days:
                        reminders.append({**idea, "reminder_reason": f"Minęło {days} dni od zapisania"})
                except Exception:
                    pass

        return reminders

# =============================================================================
# TREND ALERTS
# =============================================================================



# =============================================================================
# MANUALNE ASSETY: SERIE i KONKURENCJA
# =============================================================================

class SeriesManager:
    """Manualny mapping: film -> seria. Przechowywane lokalnie w JSON."""

    def __init__(self):
        ensure_config_dir()
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if SERIES_MAP_FILE.exists():
            try:
                with open(SERIES_MAP_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                print(f"⚠ Nie udało się wczytać mapy serii: {e}")
        return {"series": {}, "video_to_series": {}}

    def save(self):
        ensure_config_dir()
        with open(SERIES_MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def list_series(self) -> List[str]:
        return sorted(list(self.data.get("series", {}).keys()))

    def set_series(self, series_name: str, video_ids: List[str]):
        series_name = (series_name or "").strip()
        if not series_name:
            return
        video_ids = [v for v in (video_ids or []) if v]
        self.data.setdefault("series", {})[series_name] = video_ids
        # rebuild index
        self.data["video_to_series"] = {}
        for s, vids in self.data.get("series", {}).items():
            for vid in vids:
                self.data["video_to_series"][vid] = s
        self.save()

    def add_videos_to_series(self, series_name: str, video_ids: List[str]):
        series_name = (series_name or "").strip()
        if not series_name:
            return
        cur = set(self.data.get("series", {}).get(series_name, []))
        for vid in (video_ids or []):
            if vid:
                cur.add(vid)
                self.data.setdefault("video_to_series", {})[vid] = series_name
        self.data.setdefault("series", {})[series_name] = sorted(list(cur))
        self.save()

    def remove_series(self, series_name: str) -> bool:
        if series_name in self.data.get("series", {}):
            vids = self.data["series"].pop(series_name)
            # remove reverse mapping
            for vid in vids:
                if self.data.get("video_to_series", {}).get(vid) == series_name:
                    self.data["video_to_series"].pop(vid, None)
            self.save()
            return True
        return False

    def get_series_for_video(self, video_id: str) -> str:
        return self.data.get("video_to_series", {}).get(video_id, "")

    def get_videos_for_series(self, series_name: str) -> List[str]:
        return self.data.get("series", {}).get(series_name, [])


class CompetitorManager:
    """Lista kanałów konkurencji do trackingu (lokalny JSON)."""

    def __init__(self):
        ensure_config_dir()
        self.competitors = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if COMPETITORS_FILE.exists():
            try:
                with open(COMPETITORS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                print(f"⚠ Nie udało się wczytać listy konkurencji: {e}")
        return []

    def save(self):
        ensure_config_dir()
        with open(COMPETITORS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.competitors, f, indent=2, ensure_ascii=False)

    def add(self, name: str, channel_id: str, niche_notes: str = "") -> str:
        name = (name or "").strip()
        channel_id = (channel_id or "").strip()
        if not channel_id:
            return ""
        cid = hashlib.md5(f"{channel_id}{datetime.now().isoformat()}".encode()).hexdigest()[:8]
        self.competitors.append({
            "id": cid,
            "name": name or channel_id,
            "channel_id": channel_id,
            "notes": niche_notes,
            "added": datetime.now().isoformat()
        })
        self.save()
        return cid

    def remove(self, competitor_id: str) -> bool:
        before = len(self.competitors)
        self.competitors = [c for c in self.competitors if c.get("id") != competitor_id]
        if len(self.competitors) != before:
            self.save()
            return True
        return False

    def list_all(self) -> List[Dict[str, Any]]:
        return self.competitors

class TrendAlerts:
    """Zarządza alertami trendów"""
    
    def __init__(self):
        ensure_config_dir()
        self.alerts = self._load()
    
    def _load(self) -> List[Dict]:
        """Ładuje alerty z pliku"""
        if TREND_ALERTS_FILE.exists():
            try:
                with open(TREND_ALERTS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (IOError, json.JSONDecodeError) as e:
                print(f"⚠ Nie udało się wczytać alertów trendów: {e}")
        return []
    
    def save(self):
        """Zapisuje alerty do pliku"""
        ensure_config_dir()
        with open(TREND_ALERTS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.alerts, f, indent=2, ensure_ascii=False)
    
    def add_topic(self, topic: str, threshold: int = 50):
        """Dodaje temat do monitorowania"""
        if any(a["topic"].lower() == topic.lower() for a in self.alerts):
            return False  # Już istnieje
        
        alert = {
            "id": hashlib.md5(f"{topic}{datetime.now().isoformat()}".encode()).hexdigest()[:8],
            "topic": topic,
            "added": datetime.now().isoformat(),
            "threshold": threshold,  # Poziom interest do alertu
            "last_check": None,
            "last_value": None,
            "triggered": False,
            "history": [],
        }
        self.alerts.append(alert)
        self.save()
        return True
    
    def remove_topic(self, topic_id: str):
        """Usuwa temat z monitorowania"""
        self.alerts = [a for a in self.alerts if a.get("id") != topic_id]
        self.save()
    
    def get_all(self) -> List[Dict]:
        """Zwraca wszystkie monitorowane tematy"""
        return self.alerts
    
    def update_check(self, topic_id: str, current_value: int, is_trending: bool):
        """Aktualizuje wynik sprawdzenia"""
        for alert in self.alerts:
            if alert.get("id") == topic_id:
                alert["last_check"] = datetime.now().isoformat()
                alert["last_value"] = current_value
                alert["triggered"] = is_trending
                alert["history"].append({
                    "date": datetime.now().isoformat(),
                    "value": current_value
                })
                # Ogranicz historię do 30 punktów
                alert["history"] = alert["history"][-30:]
                self.save()
                return True
        return False
    
    def get_triggered(self) -> List[Dict]:
        """Zwraca tematy które aktualnie trendują"""
        return [a for a in self.alerts if a.get("triggered")]


# =============================================================================
# FUNKCJE POMOCNICZE
# =============================================================================



def get_series_manager():
    """Factory: SeriesManager"""
    return SeriesManager()

def get_competitor_manager():
    """Factory: CompetitorManager"""
    return CompetitorManager()

def generate_report(evaluation: Dict) -> str:
    """Generuje tekstowy raport do skopiowania"""
    report = []
    report.append("=" * 60)
    report.append("📊 RAPORT OCENY POMYSŁU NA FILM")
    report.append("=" * 60)
    report.append("")
    report.append(f"📅 Data: {evaluation.get('timestamp', datetime.now().isoformat())[:10]}")
    report.append("")
    report.append(f"📹 TYTUŁ: {evaluation.get('title', '')}")
    report.append(f"💬 OBIETNICA: {evaluation.get('promise', '')}")
    report.append("")
    report.append("-" * 60)
    report.append("WERDYKT")
    report.append("-" * 60)
    report.append(f"🎯 Wynik: {evaluation.get('final_score_with_bonus', evaluation.get('final_score', 0))}/100")
    report.append(f"📊 Werdykt: {evaluation.get('final_verdict', '')}")
    report.append("")
    report.append(f"   • Data Score (ML): {evaluation.get('data_score', 0)}")
    report.append(f"   • LLM Score: {evaluation.get('llm_score', 0)}")
    report.append(f"   • Kara za ryzyko: -{evaluation.get('risk_penalty', 0)}")
    report.append(f"   • Bonus z analiz: +{evaluation.get('advanced_bonus', 0)}")
    report.append("")
    report.append("-" * 60)
    report.append("DIAGNOZA")
    report.append("-" * 60)
    report.append(evaluation.get('why', ''))
    report.append("")
    
    # Ryzyka
    risks = evaluation.get('risk_flags', [])
    if risks:
        report.append(f"🚩 RYZYKA: {', '.join(risks)}")
        report.append("")
    
    # Wymiary
    dims = evaluation.get('dimensions', {})
    if dims:
        report.append("-" * 60)
        report.append("WYMIARY OCENY")
        report.append("-" * 60)
        for dim, val in dims.items():
            bar = "█" * int(val / 10) + "░" * (10 - int(val / 10))
            report.append(f"   {dim}: {bar} {val}/100")
        report.append("")
    
    # Ulepszenia
    improvements = evaluation.get('improvements', [])
    if improvements:
        report.append("-" * 60)
        report.append("CO POPRAWIĆ")
        report.append("-" * 60)
        for i, imp in enumerate(improvements, 1):
            report.append(f"   {i}. {imp}")
        report.append("")
    
    # Warianty tytułów
    title_variants = evaluation.get('title_variants', [])
    if title_variants:
        report.append("-" * 60)
        report.append("WARIANTY TYTUŁU")
        report.append("-" * 60)
        for var in title_variants:
            report.append(f"   • {var}")
        report.append("")
    
    # Warianty obietnic
    promise_variants = evaluation.get('promise_variants', [])
    if promise_variants:
        report.append("-" * 60)
        report.append("WARIANTY OBIETNICY")
        report.append("-" * 60)
        for var in promise_variants:
            report.append(f"   • {var}")
        report.append("")
    
    # Hook
    hook = evaluation.get('suggested_hook_angle', '')
    emotion = evaluation.get('target_emotion', '')
    if hook or emotion:
        report.append("-" * 60)
        report.append("HOOK")
        report.append("-" * 60)
        if hook:
            report.append(f"   Sugerowany kąt: {hook}")
        if emotion:
            report.append(f"   Docelowa emocja: {emotion}")
        report.append("")
    
    report.append("=" * 60)
    report.append("Wygenerowano przez YT Idea Evaluator Pro")
    report.append("=" * 60)
    
    return "\n".join(report)


# =============================================================================
# SINGLETON INSTANCES
# =============================================================================

_config = None
_history = None
_vault = None
_alerts = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def get_history() -> EvaluationHistory:
    global _history
    if _history is None:
        _history = EvaluationHistory()
    return _history


def get_vault() -> IdeaVault:
    global _vault
    if _vault is None:
        _vault = IdeaVault()
    return _vault


def get_alerts() -> TrendAlerts:
    global _alerts
    if _alerts is None:
        _alerts = TrendAlerts()
    return _alerts
