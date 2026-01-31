"""
YT IDEA EVALUATOR PRO v2
========================
Profesjonalne narzędzie do oceny pomysłów na mroczne dokumenty YouTube.
NOWOŚĆ w v2: Integracja z REALNYMI metrykami filmów (views, retention, watch time).

Autor: Dla Dawida
Wersja: 2.0 (metrics-aware)
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from collections import Counter
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, r2_score
from openai import OpenAI
from pydantic import BaseModel, Field


# =============================================================================
# KONFIGURACJA
# =============================================================================

class Config:
    """Centralna konfiguracja narzędzia v2"""
    
    # Modele OpenAI
    EMB_MODEL = os.getenv("EMB_MODEL", "text-embedding-3-large")
    JUDGE_MODELS = [
        os.getenv("JUDGE_MODEL", "gpt-4o"),
        "gpt-4o-mini"  # fallback
    ]
    
    # Ścieżki
    CACHE_DIR = "./emb_cache"
    DATA_DIR = "./data"
    
    # Progi decyzyjne (kalibrowane)
    THRESHOLD_PASS = 68
    THRESHOLD_BORDER = 52
    
    # Wagi scoringu v2
    WEIGHT_DATA = 0.30      # Waga modelu klasyfikacji (PASS/FAIL)
    WEIGHT_METRICS = 0.25   # NOWE: Waga modeli regresji (views/retention)
    WEIGHT_LLM = 0.45       # Waga oceny LLM
    
    # Parametry retrievalu
    DEFAULT_TOPN = 5
    DEFAULT_N_JUDGES = 2
    
    # Auto-labeling thresholds (jeśli brak labelek)
    AUTO_VIEWS_PASS = 50000
    AUTO_VIEWS_FAIL = 15000
    AUTO_RETENTION_PASS = 45.0
    AUTO_RETENTION_FAIL = 25.0


# =============================================================================
# SYSTEM PROMPT v2 - Z KONTEKSTEM METRYK
# =============================================================================

SYSTEM_JUDGE_V2 = """Jesteś BEZLITOSNYM selekcjonerem pomysłów na odcinki YouTube dla kanału typu MROCZNY DOKUMENT ŚLEDCZY (styl: LEMMiNO, Nexpo, SunnyV2, Coffeezilla).

## TWOJA ROLA
Oceniasz WYŁĄCZNIE potencjał PACKAGING (tytuł + obietnica) pod kątem:
- Czy to zatrzyma scrollowanie?
- Czy obietnica buduje napięcie, które wymusza kliknięcie?
- Czy temat pasuje do niszy mrocznych dokumentów?

## KONTEKST METRYK KANAŁU
Masz dostęp do REALNYCH statystyk z kanału:
- Views: ile wyświetleń zdobywają podobne filmy w pierwszych 7 dniach
- Retention: % obejrzanego filmu (wyższy = lepszy)
- Percentyle pokazują jak filmy wypadają względem CAŁEGO kanału

UWAGA: Twoja predykcja metryk powinna być REALISTYCZNA względem benchmarków kanału!

## KRYTERIA OCENY (0-100 każde)

### 1. CURIOSITY GAP (Luka ciekawości)
- 90-100: Nieznośna luka - muszę wiedzieć, nie mogę scrollować dalej
- 70-89: Silna luka - intryguje, ale mogę się oprzeć
- 50-69: Przeciętna - "ciekawe", ale nie pilne
- 30-49: Słaba - przewidywalne, nudne
- 0-29: Brak luki - zero napięcia

### 2. SPECIFICITY (Konkretność)
- 90-100: Laser - precyzyjna obietnica, konkretne detale
- 70-89: Ostre - jasne co dostanę
- 50-69: Mgławe - ogólnikowe sformułowania
- 30-49: Rozmyte - nie wiem czego się spodziewać
- 0-29: Abstrakt - nic konkretnego

### 3. DARK_NICHE_FIT (Dopasowanie do niszy)
- 90-100: Idealny temat na mroczny dokument (tajemnice, zbrodnie, systemy, kulty, afery)
- 70-89: Dobry fit - można zrobić ciemno
- 50-69: Neutralny - wymaga przeróbki kąta
- 30-49: Słaby fit - raczej nie dla tej niszy
- 0-29: Zły fit - to nie jest temat na dark doc

### 4. HOOK_POTENTIAL (Potencjał na hook)
- 90-100: Mam gotowy opening w głowie - wiem jak zacząć
- 70-89: Widzę kilka opcji na mocne otwarcie
- 50-69: Trudne do zhookowania, ale możliwe
- 30-49: Nie widzę mocnego otwarcia
- 0-29: Nie da się tego zahookować

### 5. SHAREABILITY (Wiralowość)
- 90-100: "Musisz to obejrzeć" - naturalny przekaz dalej
- 70-89: Wysoka szansa na share
- 50-69: Może ktoś podeśle
- 30-49: Mało prawdopodobne
- 0-29: Zero potencjału wiralowego

### 6. TITLE_CRAFT (Rzemiosło tytułu)
- 90-100: Mistrzowski - każde słowo pracuje
- 70-89: Dobry - drobne poprawki
- 50-69: Przeciętny - wymaga pracy
- 30-49: Słaby - do przepisania
- 0-29: Tragiczny - od zera

## CZERWONE FLAGI (risk_flags)
Oznacz WSZYSTKIE które występują:
- "CLICKBAIT_BACKFIRE" - obietnica za duża, rozczarowanie pewne
- "DEMONETIZATION_RISK" - temat może być zablokowany/ograniczony
- "OVERSATURATED" - temat zrobiony 1000 razy
- "TOO_NICHE" - zbyt wąska grupa docelowa
- "NO_STAKES" - brak stawki, nikogo to nie obchodzi
- "VAGUE_PROMISE" - obietnica nic nie mówi
- "GENERIC_TITLE" - tytuł jak setki innych
- "WRONG_NICHE" - to nie jest temat na dark doc
- "NO_TENSION" - zero napięcia w zestawieniu tytuł+obietnica
- "SPOILER_IN_TITLE" - tytuł zdradza punchline
- "LOW_RETENTION_PATTERN" - temat trudny do utrzymania uwagi
- "LOW_VIEWS_PATTERN" - podobne tematy miały słabe views

## PREDYKCJA METRYK (WYMAGANE)
Na podstawie podobnych filmów z kanału i ich wyników, MUSISZ podać:
- predicted_views_percentile: 0-100 (w jakim percentylu kanału wyląduje ten film?)
- predicted_retention: 0-100 (jaki % filmu obejrzy przeciętny widz?)
- metrics_analysis: Krótkie wyjaśnienie predykcji
- similar_hit_to_emulate: Który HIT z historii najlepiej naśladować i dlaczego

## FORMAT ODPOWIEDZI
Użyj DOKŁADNIE schematu Structured Outputs. Bądź BRUTALNIE szczery w "why".

## GENEROWANIE WARIANTÓW
Dla title_variants użyj RÓŻNYCH strategii:
1. Mystery angle (ukryta prawda)
2. Numbers/specificity (konkretne liczby)
3. Contrast/paradox (sprzeczność)
4. Stakes/consequence (konsekwencje)
5. Time element (kiedy, jak długo)
6. Personal angle (czyjaś historia)

Dla promise_variants:
- Każda obietnica musi budować INNY rodzaj napięcia
- Unikaj powtarzania tej samej struktury zdania
- Mix: pytania, twierdzenia, kontrasty, cliffhangery
"""


# =============================================================================
# MODELE DANYCH (Pydantic) - v2 z metrykami
# =============================================================================

class IdeaEvaluationV2(BaseModel):
    """Pełna ocena pomysłu przez LLM - wersja z predykcją metryk"""
    
    # Werdykt główny
    verdict: str = Field(description="PASS | BORDER | FAIL")
    packaging_score: int = Field(ge=0, le=100, description="Ocena ogólna packagingu")
    confidence: int = Field(ge=0, le=100, description="Pewność oceny")
    
    # Szczegółowe wymiary (0-100)
    curiosity_gap: int = Field(ge=0, le=100)
    specificity: int = Field(ge=0, le=100)
    dark_niche_fit: int = Field(ge=0, le=100)
    hook_potential: int = Field(ge=0, le=100)
    shareability: int = Field(ge=0, le=100)
    title_craft: int = Field(ge=0, le=100)
    
    # NOWE: Predykcja metryk
    predicted_views_percentile: int = Field(ge=0, le=100, description="Przewidywany percentyl views na kanale")
    predicted_retention: int = Field(ge=0, le=100, description="Przewidywana retencja %")
    metrics_analysis: str = Field(description="Analiza podobieństwa do hitów/wtop pod kątem metryk")
    similar_hit_to_emulate: str = Field(description="Który HIT z historii najlepiej naśladować i dlaczego")
    
    # Analiza
    why: str = Field(description="1-3 zdania, BRUTALNIE szczera diagnoza")
    risk_flags: List[str] = Field(default_factory=list)
    
    # Rekomendacje
    improvements: List[str] = Field(min_length=3, max_length=8)
    title_variants: List[str] = Field(min_length=6, max_length=12)
    promise_variants: List[str] = Field(min_length=6, max_length=12)
    
    # Dodatkowe insights
    suggested_hook_angle: str = Field(description="Sugerowany kąt na opening/hook")
    target_emotion: str = Field(description="Główna emocja do wywołania")


class ChannelBenchmarks(BaseModel):
    """Statystyki benchmarkowe kanału"""
    
    views_min: int = 0
    views_p25: int = 0
    views_median: int = 0
    views_p75: int = 0
    views_p90: int = 0
    views_max: int = 0
    views_mean: float = 0.0
    
    retention_min: float = 0.0
    retention_median: float = 0.0
    retention_max: float = 0.0
    retention_mean: float = 0.0
    
    total_videos: int = 0


# =============================================================================
# GŁÓWNA KLASA - YT IDEA EVALUATOR v2
# =============================================================================

class YTIdeaEvaluatorV2:
    """
    Narzędzie do oceny pomysłów na YouTube z integracją metryk.
    
    NOWOŚĆ w v2:
    - Ładowanie realnych metryk (views, retention)
    - Modele regresji do predykcji metryk
    - Kontekst metryk dla LLM
    - Combined scoring z trzech źródeł
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.client = None
        
        # Dane
        self.df_data = None
        self.X_hist = None
        self.benchmarks: ChannelBenchmarks = None
        
        # Modele
        self.clf_label = None         # LogisticRegression na PASS/FAIL
        self.reg_views = None         # Ridge na log_views
        self.reg_retention = None     # Ridge na retention
        
        self._initialized = False
        
    def initialize(self, api_key: str = None):
        """Inicjalizacja klienta OpenAI"""
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Brak OPENAI_API_KEY!")
        
        self.client = OpenAI(api_key=api_key)
        os.makedirs(self.config.CACHE_DIR, exist_ok=True)
        print(f"✓ Initialized v2 | Embedding: {self.config.EMB_MODEL} | Judges: {self.config.JUDGE_MODELS}")
        
    # -------------------------------------------------------------------------
    # ŁADOWANIE DANYCH
    # -------------------------------------------------------------------------
        
    def load_data(
        self,
        data_path: str,
        label_col: str = "label",
        title_col: str = "title",
        views_col: str = None,
        retention_col: str = None,
        auto_label: bool = True
    ):
        """
        Załaduj dane z metrykami.
        
        Automatycznie rozpoznaje kolumny:
        - views_0_7d, viewCount, views -> views
        - avgViewPercentage_0_7d, avgViewPercentage -> retention
        - label, LABEL -> label (opcjonalne, może być auto-wygenerowane)
        
        Args:
            data_path: Ścieżka do CSV
            label_col: Nazwa kolumny z labelkami (opcjonalna)
            title_col: Nazwa kolumny z tytułami
            views_col: Override kolumny views
            retention_col: Override kolumny retention
            auto_label: Czy generować labelki z metryk jeśli brak
        """
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} rows from {data_path}")
        print(f"Columns: {list(df.columns)}")
        
        # Normalizuj kolumny
        df = self._normalize_columns(df, title_col, views_col, retention_col)
        
        # Auto-labeling jeśli brak
        if label_col not in df.columns or df[label_col].isna().all():
            if auto_label and "views" in df.columns:
                df = self._auto_label(df)
                print(f"✓ Auto-labeled based on metrics")
            else:
                raise ValueError(f"Brak kolumny '{label_col}' i nie można auto-labelować")
        else:
            df["label"] = df[label_col].fillna("BORDER")
        
        # Filtruj valid labels
        df = df[df["label"].isin(["PASS", "FAIL", "BORDER"])].copy()
        
        # Buduj tekst do embeddingu
        df["packaging_text"] = df.apply(self._build_packaging_text, axis=1)
        
        # Oblicz benchmarki
        self.benchmarks = self._compute_benchmarks(df)
        
        self.df_data = df
        stats = df["label"].value_counts().to_dict()
        print(f"✓ Loaded {len(df)} videos | Labels: {stats}")
        print(f"✓ Benchmarks: median views={self.benchmarks.views_median:,}, median retention={self.benchmarks.retention_median:.1f}%")
        
        return self
    
    def _normalize_columns(
        self,
        df: pd.DataFrame,
        title_col: str,
        views_col: str = None,
        retention_col: str = None
    ) -> pd.DataFrame:
        """Normalizuje nazwy kolumn do standardowych"""
        
        # Title
        if title_col and title_col in df.columns:
            df["title"] = df[title_col]
        elif "title" not in df.columns:
            for c in ["title_api", "Title", "TITLE"]:
                if c in df.columns:
                    df["title"] = df[c]
                    break
        
        # Views
        if views_col and views_col in df.columns:
            df["views"] = pd.to_numeric(df[views_col], errors="coerce")
        elif "views" not in df.columns:
            for c in ["views_0_7d", "viewCount", "Views", "VIEWS"]:
                if c in df.columns:
                    df["views"] = pd.to_numeric(df[c], errors="coerce")
                    break
        
        # Retention
        if retention_col and retention_col in df.columns:
            df["retention"] = pd.to_numeric(df[retention_col], errors="coerce")
        elif "retention" not in df.columns:
            for c in ["avgViewPercentage_0_7d", "avgViewPercentage", "Retention", "RETENTION", "avg_view_percentage"]:
                if c in df.columns:
                    df["retention"] = pd.to_numeric(df[c], errors="coerce")
                    break
        
        # Log-transform views if not present
        if "views" in df.columns and "log_views" not in df.columns:
            df["log_views"] = np.log1p(df["views"].fillna(0))
        
        return df
    
    def _auto_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generuje labelki na podstawie metryk"""
        cfg = self.config
        
        def get_label(row):
            v = row.get("views", 0) or 0
            r = row.get("retention", 0) or 0
            
            # PASS: wysokie views LUB wysoka retencja
            if v >= cfg.AUTO_VIEWS_PASS or r >= cfg.AUTO_RETENTION_PASS:
                return "PASS"
            # FAIL: niskie views I niska retencja
            if v < cfg.AUTO_VIEWS_FAIL and r < cfg.AUTO_RETENTION_FAIL:
                return "FAIL"
            return "BORDER"
        
        df["label"] = df.apply(get_label, axis=1)
        return df
    
    def _build_packaging_text(self, row) -> str:
        """Buduje tekst do embeddingu z tytułu i obietnicy"""
        title = str(row.get("title", "") or "").strip()
        promise = str(row.get("promise", "") or row.get("description", "") or "").strip()
        if promise:
            return f"TYTUŁ: {title}\nOBIETNICA: {promise}"[:8000]
        return f"TYTUŁ: {title}"[:8000]
    
    def _compute_benchmarks(self, df: pd.DataFrame) -> ChannelBenchmarks:
        """Oblicza statystyki benchmarkowe kanału"""
        views = df["views"].dropna() if "views" in df.columns else pd.Series([0])
        retention = df["retention"].dropna() if "retention" in df.columns else pd.Series([0])
        
        return ChannelBenchmarks(
            views_min=int(views.min()) if len(views) else 0,
            views_p25=int(views.quantile(0.25)) if len(views) else 0,
            views_median=int(views.median()) if len(views) else 0,
            views_p75=int(views.quantile(0.75)) if len(views) else 0,
            views_p90=int(views.quantile(0.90)) if len(views) else 0,
            views_max=int(views.max()) if len(views) else 0,
            views_mean=float(views.mean()) if len(views) else 0.0,
            retention_min=float(retention.min()) if len(retention) else 0.0,
            retention_median=float(retention.median()) if len(retention) else 0.0,
            retention_max=float(retention.max()) if len(retention) else 0.0,
            retention_mean=float(retention.mean()) if len(retention) else 0.0,
            total_videos=len(df)
        )
    
    # -------------------------------------------------------------------------
    # EMBEDDINGI
    # -------------------------------------------------------------------------
    
    def build_embeddings(self, force_rebuild: bool = False):
        """Buduje embeddingi dla wszystkich historycznych przykładów"""
        if self.df_data is None:
            raise RuntimeError("Najpierw załaduj dane: load_data()")
        
        texts = self.df_data["packaging_text"].tolist()
        self.X_hist = self._embed_texts(texts, force_rebuild=force_rebuild)
        print(f"✓ Embeddings: {self.X_hist.shape}")
        
        return self
    
    def _embed_texts(self, texts: List[str], force_rebuild: bool = False) -> np.ndarray:
        """Embeduje teksty z cache'owaniem"""
        vecs = [None] * len(texts)
        to_fetch = []
        idx_map = []
        
        for i, t in enumerate(texts):
            t = str(t)[:8000]
            h = hashlib.sha256((self.config.EMB_MODEL + "|" + t).encode()).hexdigest()
            p = os.path.join(self.config.CACHE_DIR, h + ".npy")
            
            if not force_rebuild and os.path.exists(p):
                vecs[i] = np.load(p)
            else:
                to_fetch.append(t)
                idx_map.append((i, p))
        
        if to_fetch:
            batch_size = 64
            for start in tqdm(range(0, len(to_fetch), batch_size), desc="Embedding"):
                chunk = to_fetch[start:start+batch_size]
                r = self.client.embeddings.create(model=self.config.EMB_MODEL, input=chunk)
                for j, item in enumerate(r.data):
                    v = np.array(item.embedding, dtype="float32")
                    i, path = idx_map[start + j]
                    np.save(path, v)
                    vecs[i] = v
        
        X = np.vstack(vecs).astype("float32")
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        return X
    
    # -------------------------------------------------------------------------
    # TRENOWANIE MODELI
    # -------------------------------------------------------------------------
    
    def train_models(self):
        """Trenuje wszystkie trzy modele: klasyfikator + 2 regresory"""
        if self.X_hist is None:
            raise RuntimeError("Najpierw zbuduj embeddingi: build_embeddings()")
        
        print("\n=== TRENOWANIE MODELI ===")
        
        # 1. Klasyfikator PASS/FAIL
        self._train_label_classifier()
        
        # 2. Regresor views
        self._train_views_regressor()
        
        # 3. Regresor retention
        self._train_retention_regressor()
        
        self._initialized = True
        print("\n✓ Wszystkie modele wytrenowane!")
        
        return self
    
    def _train_label_classifier(self):
        """Trenuje model LogisticRegression na PASS/FAIL"""
        mask = self.df_data["label"].isin(["PASS", "FAIL"])
        X = self.X_hist[mask]
        y = (self.df_data[mask]["label"] == "PASS").astype(int).values
        
        if len(y) < 10:
            print("⚠ Za mało danych PASS/FAIL, pomijam klasyfikator")
            return
        
        self.clf_label = LogisticRegression(max_iter=2000, class_weight="balanced")
        skf = StratifiedKFold(n_splits=min(5, len(y)//2), shuffle=True, random_state=42)
        
        aucs = []
        for tr, te in skf.split(X, y):
            self.clf_label.fit(X[tr], y[tr])
            p = self.clf_label.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
        
        self.clf_label.fit(X, y)
        print(f"✓ Label Classifier | CV AUC: {np.mean(aucs):.3f} ({[round(a,3) for a in aucs]})")
    
    def _train_views_regressor(self):
        """Trenuje Ridge regression na log(views)"""
        if "log_views" not in self.df_data.columns:
            print("⚠ Brak kolumny log_views, pomijam regresor views")
            return
        
        mask = self.df_data["log_views"].notna()
        X = self.X_hist[mask]
        y = self.df_data.loc[mask, "log_views"].values
        
        if len(y) < 10:
            print("⚠ Za mało danych views, pomijam regresor")
            return
        
        self.reg_views = Ridge(alpha=1.0)
        kf = KFold(n_splits=min(5, len(y)//2), shuffle=True, random_state=42)
        
        r2s = []
        for tr, te in kf.split(X):
            self.reg_views.fit(X[tr], y[tr])
            p = self.reg_views.predict(X[te])
            r2s.append(r2_score(y[te], p))
        
        self.reg_views.fit(X, y)
        print(f"✓ Views Regressor | CV R²: {np.mean(r2s):.3f} ({[round(r,3) for r in r2s]})")
    
    def _train_retention_regressor(self):
        """Trenuje Ridge regression na retention %"""
        if "retention" not in self.df_data.columns:
            print("⚠ Brak kolumny retention, pomijam regresor")
            return
        
        mask = self.df_data["retention"].notna() & (self.df_data["retention"] > 0)
        X = self.X_hist[mask]
        y = self.df_data.loc[mask, "retention"].values
        
        if len(y) < 10:
            print("⚠ Za mało danych retention, pomijam regresor")
            return
        
        self.reg_retention = Ridge(alpha=1.0)
        kf = KFold(n_splits=min(5, len(y)//2), shuffle=True, random_state=42)
        
        r2s = []
        for tr, te in kf.split(X):
            self.reg_retention.fit(X[tr], y[tr])
            p = self.reg_retention.predict(X[te])
            r2s.append(r2_score(y[te], p))
        
        self.reg_retention.fit(X, y)
        print(f"✓ Retention Regressor | CV R²: {np.mean(r2s):.3f} ({[round(r,3) for r in r2s]})")
    
    # -------------------------------------------------------------------------
    # RETRIEVAL I KONTEKST
    # -------------------------------------------------------------------------
    
    def _retrieve_examples(self, title: str, promise: str = "", topn: int = 5) -> Tuple:
        """Znajduje podobne PASS i FAIL z historii wraz z metrykami"""
        text = f"TYTUŁ: {title}\nOBIETNICA: {promise}".strip()[:8000]
        v = self._embed_texts([text])[0]
        sims = self.X_hist @ v
        
        tmp = self.df_data.copy()
        tmp["sim"] = sims
        
        top_pass = tmp[tmp["label"] == "PASS"].nlargest(topn, "sim")
        top_fail = tmp[tmp["label"] == "FAIL"].nlargest(topn, "sim")
        top_all = tmp.nlargest(topn * 2, "sim")
        
        return v, top_pass, top_fail, top_all
    
    def _format_metrics_context(self, top_pass, top_fail, top_all) -> str:
        """Formatuje kontekst z metrykami dla LLM"""
        
        # Benchmarki kanału
        bench = self.benchmarks
        context = f"""
=== BENCHMARKI KANAŁU ({bench.total_videos} filmów) ===
Views (7 dni):
  - Minimum: {bench.views_min:,}
  - Percentyl 25: {bench.views_p25:,}
  - Mediana: {bench.views_median:,}
  - Percentyl 75: {bench.views_p75:,}
  - Percentyl 90: {bench.views_p90:,}
  - Maximum: {bench.views_max:,}
  - Średnia: {bench.views_mean:,.0f}

Retention:
  - Minimum: {bench.retention_min:.1f}%
  - Mediana: {bench.retention_median:.1f}%
  - Maximum: {bench.retention_max:.1f}%
  - Średnia: {bench.retention_mean:.1f}%

=== PODOBNE HITY (PASS) ===
"""
        
        # Podobne PASS z metrykami
        metric_cols = ["title", "views", "retention", "sim"]
        available_cols = [c for c in metric_cols if c in top_pass.columns]
        
        for _, row in top_pass.head(5).iterrows():
            views = row.get("views", "?")
            retention = row.get("retention", "?")
            sim = row.get("sim", 0)
            if pd.notna(views):
                views = f"{int(views):,}"
            if pd.notna(retention):
                retention = f"{float(retention):.1f}%"
            context += f"• {row['title'][:60]} | Views: {views} | Ret: {retention} | Sim: {sim:.2f}\n"
        
        context += "\n=== PODOBNE WTOPY (FAIL) ===\n"
        
        for _, row in top_fail.head(5).iterrows():
            views = row.get("views", "?")
            retention = row.get("retention", "?")
            sim = row.get("sim", 0)
            if pd.notna(views):
                views = f"{int(views):,}"
            if pd.notna(retention):
                retention = f"{float(retention):.1f}%"
            context += f"• {row['title'][:60]} | Views: {views} | Ret: {retention} | Sim: {sim:.2f}\n"
        
        return context
    
    # -------------------------------------------------------------------------
    # OCENA LLM
    # -------------------------------------------------------------------------
    
    def _judge_once(self, title: str, promise: str, context: str, model: str) -> IdeaEvaluationV2:
        """Pojedyncza ocena przez LLM"""
        user_msg = f"""NOWY POMYSŁ DO OCENY:

TYTUŁ: {title}
OBIETNICA: {promise}

{context}

Oceń ten pomysł według wszystkich kryteriów. 
PAMIĘTAJ: Musisz podać predicted_views_percentile i predicted_retention na podstawie benchmarków!
Bądź BRUTALNIE szczery."""

        response = self.client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_JUDGE_V2},
                {"role": "user", "content": user_msg}
            ],
            response_format=IdeaEvaluationV2
        )
        
        return response.choices[0].message.parsed
    
    def _judge_with_ensemble(self, title: str, promise: str, context: str, n_judges: int = 2) -> Tuple:
        """Ocena z ensemble kilku wywołań"""
        results = []
        
        for _ in range(n_judges):
            for model in self.config.JUDGE_MODELS:
                try:
                    r = self._judge_once(title, promise, context, model)
                    results.append(r)
                    break
                except Exception as e:
                    print(f"⚠ Judge error ({model}): {e}")
                    continue
        
        if not results:
            raise RuntimeError("Nie udało się uzyskać oceny od żadnego modelu")
        
        # Agregacja: mediana score, najczęstszy verdict
        scores = [r.packaging_score for r in results]
        verdicts = [r.verdict for r in results]
        
        med_score = int(np.median(scores))
        final_verdict = Counter(verdicts).most_common(1)[0][0]
        
        # Wybierz najbliższy do mediany
        best = min(results, key=lambda r: abs(r.packaging_score - med_score))
        best.packaging_score = med_score
        best.verdict = final_verdict
        
        return best, results
    
    # -------------------------------------------------------------------------
    # SCORING
    # -------------------------------------------------------------------------
    
    def _calculate_risk_penalty(self, evaluation: IdeaEvaluationV2) -> float:
        """Oblicza karę za ryzyka"""
        risk_weights = {
            "CLICKBAIT_BACKFIRE": 15,
            "DEMONETIZATION_RISK": 12,
            "OVERSATURATED": 10,
            "NO_STAKES": 10,
            "VAGUE_PROMISE": 8,
            "GENERIC_TITLE": 8,
            "WRONG_NICHE": 15,
            "NO_TENSION": 10,
            "TOO_NICHE": 5,
            "SPOILER_IN_TITLE": 7,
            "LOW_RETENTION_PATTERN": 8,
            "LOW_VIEWS_PATTERN": 8,
        }
        
        penalty = 0
        for flag in evaluation.risk_flags:
            penalty += risk_weights.get(flag, 5)
        
        return min(penalty, 30)
    
    def _predict_metrics(self, v: np.ndarray) -> Dict[str, float]:
        """Predykcja metryk z modeli regresji"""
        result = {
            "pred_log_views": None,
            "pred_views": None,
            "pred_views_percentile": None,
            "pred_retention": None,
        }
        
        # Views
        if self.reg_views is not None:
            pred_log = float(self.reg_views.predict([v])[0])
            pred_views = np.expm1(pred_log)
            result["pred_log_views"] = pred_log
            result["pred_views"] = pred_views
            
            # Percentyl względem benchmarków
            if self.benchmarks and self.benchmarks.views_median > 0:
                all_views = self.df_data["views"].dropna()
                percentile = (all_views < pred_views).mean() * 100
                result["pred_views_percentile"] = min(100, max(0, percentile))
        
        # Retention
        if self.reg_retention is not None:
            pred_ret = float(self.reg_retention.predict([v])[0])
            result["pred_retention"] = min(100, max(0, pred_ret))
        
        return result
    
    def _combined_score(
        self,
        data_score: float,
        metrics_pred: Dict,
        llm_score: int,
        risk_penalty: float
    ) -> Tuple[float, str]:
        """Oblicza finalny score z wszystkich składników"""
        
        cfg = self.config
        
        # Składniki
        comp_data = data_score * 100  # 0-100
        comp_llm = llm_score          # 0-100
        
        # Metrics score: średnia z percentylu views i retention
        comp_metrics = 50  # default
        vp = metrics_pred.get("pred_views_percentile")
        rp = metrics_pred.get("pred_retention")
        
        if vp is not None and rp is not None:
            comp_metrics = (vp * 0.6 + rp * 0.4)  # views ważniejsze
        elif vp is not None:
            comp_metrics = vp
        elif rp is not None:
            comp_metrics = rp
        # else: zostaje default 50
        
        # Ważona suma
        base = (
            cfg.WEIGHT_DATA * comp_data +
            cfg.WEIGHT_METRICS * comp_metrics +
            cfg.WEIGHT_LLM * comp_llm
        )
        
        # Normalizuj i odejmij karę
        total_weight = cfg.WEIGHT_DATA + cfg.WEIGHT_METRICS + cfg.WEIGHT_LLM
        normalized = base / total_weight
        final = max(0, normalized - risk_penalty)
        
        # Werdykt
        if final >= cfg.THRESHOLD_PASS:
            verdict = "PASS"
        elif final >= cfg.THRESHOLD_BORDER:
            verdict = "BORDER"
        else:
            verdict = "FAIL"
        
        return round(final, 1), verdict
    
    # -------------------------------------------------------------------------
    # GŁÓWNA FUNKCJA OCENY
    # -------------------------------------------------------------------------
    
    def evaluate(
        self,
        title: str,
        promise: str = "",
        topn: int = None,
        n_judges: int = None,
        optimize: bool = False
    ) -> Dict[str, Any]:
        """
        Główna funkcja oceny pomysłu z metrykami.
        
        Args:
            title: Tytuł filmu
            promise: Obietnica/opis (opcjonalnie)
            topn: Ile podobnych przykładów pobrać
            n_judges: Ile razy odpytać LLM
            optimize: Czy generować i oceniać warianty
            
        Returns:
            Słownik z pełną oceną włącznie z predykcją metryk
        """
        if not self._initialized:
            raise RuntimeError("Najpierw zainicjalizuj i wytrenuj model!")
        
        topn = topn or self.config.DEFAULT_TOPN
        n_judges = n_judges or self.config.DEFAULT_N_JUDGES
        
        # 1. Retrieve similar z metrykami
        v, top_pass, top_fail, top_all = self._retrieve_examples(title, promise, topn)
        
        # 2. Data score (klasyfikator)
        data_score = 0.5
        if self.clf_label is not None:
            data_score = float(self.clf_label.predict_proba([v])[0, 1])
        
        # 3. Predykcja metryk (regresory)
        metrics_pred = self._predict_metrics(v)
        
        # 4. Kontekst z metrykami dla LLM
        context = self._format_metrics_context(top_pass, top_fail, top_all)
        
        # 5. LLM judge
        llm_best, llm_all = self._judge_with_ensemble(title, promise, context, n_judges)
        
        # 6. Risk penalty
        risk_penalty = self._calculate_risk_penalty(llm_best)
        
        # 7. Combined score
        final_score, final_verdict = self._combined_score(
            data_score, metrics_pred, llm_best.packaging_score, risk_penalty
        )
        
        # Build result
        result = {
            # Podstawowe
            "title": title,
            "promise": promise,
            "timestamp": datetime.now().isoformat(),
            
            # Scores
            "final_score": final_score,
            "final_verdict": final_verdict,
            "data_score": round(data_score * 100, 1),
            "llm_score": llm_best.packaging_score,
            "risk_penalty": risk_penalty,
            
            # NOWE: Predykcja metryk
            "predicted_metrics": {
                "views_percentile": round(metrics_pred.get("pred_views_percentile") or 0, 1),
                "views_estimate": int(metrics_pred.get("pred_views") or 0),
                "retention_estimate": round(metrics_pred.get("pred_retention") or 0, 1),
            },
            "llm_metrics_prediction": {
                "views_percentile": llm_best.predicted_views_percentile,
                "retention": llm_best.predicted_retention,
                "analysis": llm_best.metrics_analysis,
                "hit_to_emulate": llm_best.similar_hit_to_emulate,
            },
            
            # Szczegółowe wymiary
            "dimensions": {
                "curiosity_gap": llm_best.curiosity_gap,
                "specificity": llm_best.specificity,
                "dark_niche_fit": llm_best.dark_niche_fit,
                "hook_potential": llm_best.hook_potential,
                "shareability": llm_best.shareability,
                "title_craft": llm_best.title_craft,
            },
            
            # Analiza
            "confidence": llm_best.confidence,
            "why": llm_best.why,
            "risk_flags": llm_best.risk_flags,
            "suggested_hook_angle": llm_best.suggested_hook_angle,
            "target_emotion": llm_best.target_emotion,
            
            # Rekomendacje
            "improvements": llm_best.improvements,
            "title_variants": llm_best.title_variants,
            "promise_variants": llm_best.promise_variants,
            
            # Kontekst
            "similar_pass": self._format_similar_df(top_pass, 3),
            "similar_fail": self._format_similar_df(top_fail, 3),
            
            # Benchmarki
            "channel_benchmarks": self.benchmarks.model_dump() if self.benchmarks else {},
            
            # Raw
            "_raw_judges": [r.model_dump() for r in llm_all]
        }
        
        # Optymalizacja wariantów
        if optimize:
            result["optimized_variants"] = self._optimize_variants(
                llm_best.title_variants,
                llm_best.promise_variants,
                title, promise, topn
            )
        
        return result
    
    def _format_similar_df(self, df, n: int) -> List[Dict]:
        """Formatuje podobne filmy do listy słowników"""
        cols = ["title", "sim"]
        if "views" in df.columns:
            cols.append("views")
        if "retention" in df.columns:
            cols.append("retention")
        
        rows = []
        for _, row in df.head(n).iterrows():
            r = {"title": row["title"], "sim": round(row["sim"], 3)}
            if "views" in row and pd.notna(row["views"]):
                r["views"] = int(row["views"])
            if "retention" in row and pd.notna(row["retention"]):
                r["retention"] = round(row["retention"], 1)
            rows.append(r)
        return rows
    
    def _optimize_variants(
        self,
        title_variants: List[str],
        promise_variants: List[str],
        orig_title: str,
        orig_promise: str,
        topn: int
    ) -> List[Dict]:
        """Ocenia warianty i zwraca ranking z predykcją metryk"""
        candidates = []
        
        # Kombinacje
        for t in title_variants[:6]:
            candidates.append((t, orig_promise))
        for p in promise_variants[:6]:
            candidates.append((orig_title, p))
        for t in title_variants[:3]:
            for p in promise_variants[:3]:
                candidates.append((t, p))
        
        # Deduplikacja
        seen = set()
        unique = []
        for t, p in candidates:
            key = (t.strip(), p.strip())
            if key not in seen:
                seen.add(key)
                unique.append(key)
        
        # Szybka ocena z metrykami
        scored = []
        for t, p in unique[:15]:
            v, _, _, _ = self._retrieve_examples(t, p, topn)
            
            # Data score
            ds = 0.5
            if self.clf_label is not None:
                ds = float(self.clf_label.predict_proba([v])[0, 1]) * 100
            
            # Metrics prediction
            mp = self._predict_metrics(v)
            
            scored.append({
                "title": t,
                "promise": p,
                "quick_score": round(ds, 1),
                "pred_views_percentile": round(mp.get("pred_views_percentile") or 0, 1),
                "pred_retention": round(mp.get("pred_retention") or 0, 1),
            })
        
        return sorted(scored, key=lambda x: x["quick_score"], reverse=True)[:10]
    
    def evaluate_batch(self, ideas: List[Dict], **kwargs) -> pd.DataFrame:
        """Ocena wielu pomysłów jednocześnie"""
        results = []
        for idea in tqdm(ideas, desc="Evaluating"):
            title = idea.get("title", "")
            promise = idea.get("promise", "")
            try:
                r = self.evaluate(title, promise, **kwargs)
                results.append(r)
            except Exception as e:
                results.append({
                    "title": title,
                    "promise": promise,
                    "error": str(e)
                })
        
        return pd.DataFrame(results)


# =============================================================================
# FUNKCJE POMOCNICZE
# =============================================================================

def quick_setup(
    data_path: str,
    api_key: str = None
) -> YTIdeaEvaluatorV2:
    """
    Szybka inicjalizacja evaluatora z jednym wywołaniem.
    
    Przykład:
        evaluator = quick_setup("yt_labeled_pass_fail.csv")
        result = evaluator.evaluate("Tytuł", "Obietnica")
    """
    evaluator = YTIdeaEvaluatorV2()
    evaluator.initialize(api_key)
    evaluator.load_data(data_path)
    evaluator.build_embeddings()
    evaluator.train_models()
    
    return evaluator


def format_result(result: Dict, verbose: bool = True) -> str:
    """Formatuje wynik oceny do czytelnego stringa"""
    
    lines = [
        "=" * 60,
        f"📊 OCENA: {result['title'][:50]}...",
        "=" * 60,
        "",
        f"🎯 WERDYKT: {result['final_verdict']} ({result['final_score']}/100)",
        f"   • Data Score: {result['data_score']}",
        f"   • LLM Score: {result['llm_score']}",
        f"   • Risk Penalty: -{result['risk_penalty']}",
        "",
        "📈 PREDYKCJA METRYK:",
        f"   • Views percentyl (model): {result['predicted_metrics']['views_percentile']}%",
        f"   • Views estimate: {result['predicted_metrics']['views_estimate']:,}",
        f"   • Retention estimate: {result['predicted_metrics']['retention_estimate']}%",
        "",
        f"   • Views percentyl (LLM): {result['llm_metrics_prediction']['views_percentile']}%",
        f"   • Retention (LLM): {result['llm_metrics_prediction']['retention']}%",
        "",
        "📐 WYMIARY:",
    ]
    
    for dim, val in result['dimensions'].items():
        bar = "█" * (val // 10) + "░" * (10 - val // 10)
        lines.append(f"   {dim:20} {bar} {val}")
    
    lines.extend([
        "",
        f"💬 DIAGNOZA: {result['why']}",
        "",
        f"🚩 RYZYKA: {', '.join(result['risk_flags']) or 'brak'}",
        "",
        "✨ ULEPSZENIA:",
    ])
    
    for i, imp in enumerate(result['improvements'][:5], 1):
        lines.append(f"   {i}. {imp}")
    
    if verbose:
        lines.extend([
            "",
            "📝 WARIANTY TYTUŁÓW:",
        ])
        for i, t in enumerate(result['title_variants'][:6], 1):
            lines.append(f"   {i}. {t}")
        
        lines.extend([
            "",
            f"🎯 HIT DO NAŚLADOWANIA: {result['llm_metrics_prediction']['hit_to_emulate']}",
            "",
            f"🎣 HOOK ANGLE: {result['suggested_hook_angle']}",
            f"💫 TARGET EMOTION: {result['target_emotion']}",
        ])
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


# =============================================================================
# PRZYKŁAD UŻYCIA
# =============================================================================

if __name__ == "__main__":
    # Przykładowe użycie
    print("""
=== YT IDEA EVALUATOR PRO v2 ===
Narzędzie do oceny pomysłów z integracją metryk kanału.

Przykład użycia:
    
    from yt_idea_evaluator_pro_v2 import quick_setup, format_result
    
    # Inicjalizacja z danymi
    evaluator = quick_setup("yt_labeled_pass_fail.csv")
    
    # Ocena pomysłu
    result = evaluator.evaluate(
        title="Tajemnica Zaginęła na Zawsze?",
        promise="Nikt nie wie co naprawdę wydarzyło się tamtej nocy."
    )
    
    # Wyświetl wynik
    print(format_result(result))
    
    # Z optymalizacją wariantów
    result = evaluator.evaluate(..., optimize=True)
""")
