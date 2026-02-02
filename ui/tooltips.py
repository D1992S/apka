"""Tooltip definitions for the UI."""

TOOLTIPS = {
    "judges": """
**Liczba sędziów LLM**

Ile razy model GPT oceni Twój pomysł. Więcej = dokładniejsza ocena, ale wolniejsza i droższa.

- **1 sędzia**: Szybko, tanie, ale może być niestabilne
- **2 sędziów**: Dobry balans (zalecane)
- **3 sędziów**: Najdokładniejsze, ale 3x dłużej
""",
    "topn": """
**Podobne przykłady**

Ile Twoich filmów model weźmie pod uwagę jako kontekst.

- **3-5**: Szybkie, ogólne porównanie
- **5-7**: Dobry balans (zalecane)
- **8-10**: Głębsza analiza, wolniejsze
""",
    "optimize": """
**Optymalizuj warianty**

Gdy włączone, model wygeneruje warianty tytułu i oceni każdy z nich osobno, szukając najlepszego.

⚠️ Wydłuża czas oceny 2-3x
""",
    "data_score": """
**Data Score (ML)**

Ocena z modelu Machine Learning trenowanego na TWOICH danych.

Model nauczył się wzorców z Twoich hitów vs wtop i przewiduje czy nowy pomysł pasuje do wzorca sukcesu.

- Używa embeddingów OpenAI
- Trenowany na Ridge Regression i LogisticRegression
- Im więcej danych, tym dokładniejszy
""",
    "llm_score": """
**LLM Score**

Ocena od GPT-4o który analizuje:
- Curiosity gap (czy buduje ciekawość)
- Specyficzność (czy jest konkretny)
- Dark niche fit (czy pasuje do niszy)
- Hook potential (potencjał na mocny hook)
- Shareability (czy ludzie będą udostępniać)
- Title craft (jakość tytułu)
""",
    "risk_penalty": """
**Kara za ryzyko**

Punkty odjęte za wykryte ryzyka:
- CLICKBAIT_BACKFIRE: Tytuł obiecuje za dużo
- OVERSATURATED: Temat przesycony
- TOO_NICHE: Za wąski temat
- WEAK_HOOK: Słaby potencjał na hook
- LOW_SHAREABILITY: Niska viralowość
- TITLE_TOO_LONG/SHORT: Problem z długością
- NO_CLEAR_PROMISE: Brak obietnicy
- CONTROVERSIAL: Ryzykowny temat
""",
    "trend_bonus": """
**Bonus/Kara za Trend**

Sprawdza Google Trends:
- 🔥 +10: Temat HOT, trending up
- ➡️ +5: Evergreen, stabilny
- 📉 -5: Trend spadkowy
- 💀 -10: Temat martwy
""",
    "topic_overall": """
**Overall Score**

Składa się z:
- 35%: siła najlepszego tytułu
- 30%: Opportunity z analizy konkurencji
- 35%: Viral Score (predykcja viralowości)
- Korekty: bonus/penalty trendów + dopasowanie do hitów kanału
""",
    "topic_viral": """
**Viral Score**

Predykcja potencjału viralowości:
- atrakcyjność tytułu
- dynamika tematu
- dopasowanie do niszy

Skala 0–100: im wyżej, tym lepiej.
""",
    "topic_trend": """
**Trend Score**

Ocena trendu wyszukiwań i sezonowości:
- kierunek trendu (UP/DOWN)
- poziom zainteresowania
- sezonowość tematu

Wyżej = większy wiatr w plecy.
""",
    "topic_opportunity": """
**Opportunity**

Analiza konkurencji:
- nasycenie tematu vs popyt
- porównanie podobnych filmów i ich performance

Wyżej = łatwiej się przebić.
""",
    "competition_bonus": """
**Bonus/Kara za Konkurencję**

Skanuje YouTube:
- 🟢 +15: Blue ocean, brak konkurencji
- 🟢 +10: Niska konkurencja
- 🟡 0: Umiarkowana
- 🟠 -5: Wysoka konkurencja
- 🔴 -15: Temat przesycony
""",
    "dna_bonus": """
**Bonus za DNA Match**

Sprawdza czy tytuł pasuje do wzorców Twoich hitów:
- Optymalna długość
- Trigger words z Twoich hitów
- Struktury które działają
- Max +20 punktów
""",
    "channel_views": """
**Views (wyświetlenia)**

Najprościej pozyskać:
- YouTube Studio → Analytics → eksportuj dane (CSV)
- YouTube Data API: pole `viewCount` dla każdego filmu

Dlaczego ważne:
- To główny sygnał popytu i baza do prognoz.
""",
    "channel_retention": """
**Retention (retencja)**

Najprościej pozyskać:
- YouTube Studio → Analytics → zakładka „Zaangażowanie”
- Eksportuj średnią retencję (%) per film

Dlaczego ważne:
- Retencja wpływa na rekomendacje i viral score.
""",
    "channel_label": """
**Label (PASS/BORDER/FAIL)**

Jak uzupełnić:
- Oznacz ręcznie po wynikach (np. 75+ = PASS, 60–74 = BORDER, <60 = FAIL)
- Możesz dodać własne etykiety po analizie

Dlaczego ważne:
- Model lepiej rozpoznaje wzorce hitów vs wtop.
""",
    "channel_title": """
**Title (tytuł filmu)**

Jak pozyskać:
- YouTube Data API: pole `title`
- Eksport z YouTube Studio (CSV)

Dlaczego ważne:
- Bez tytułu nie zbudujemy kontekstu ani embeddingów.
""",
    "channel_published_at": """
**Published At (data publikacji)**

Jak pozyskać:
- YouTube Data API: pole `publishedAt`
- Eksport z YouTube Studio

Dlaczego ważne:
- Umożliwia analizy trendu w czasie i prognozy.
""",
}


def show_tooltip(key: str) -> str:
    """Return tooltip text for a given key."""
    return TOOLTIPS.get(key, "")
