# Propozycja uporządkowania struktury folderów

## Aktualny stan
Wszystkie pliki leżą w katalogu głównym — kod, raporty, notebooki, dokumentacja, logi, eksperymenty. Trudno znaleźć cokolwiek.

## Proponowana struktura

```
TenisPredictionModel/
│
├── README.md                              # główny opis projektu (1 strona, co tu jest)
├── requirements.txt                       # zależności Pythona (do zrobienia)
├── .gitignore
│
├── src/                                   # ★ CAŁY KOD MODELI
│   ├── main_48_cech.py                    # baseline
│   ├── main_48_cech_modelslice.py         # diagnostyka slicingu
│   ├── main_48_cech_sliceaware.py         # wariant slice-aware (shotgun)
│   ├── main_48_cech_sliceaware_bestof5_v1.py  # wariant Bo5 (NAJLEPSZY)
│   ├── main_48_cech_sliceaware_qfserve_v3.py  # wariant QF + serve
│   ├── main_48_cech_slicecompare.py       # porównywarka 4 wariantów
│   ├── main_48_cech_seedstability.py      # test stabilności seedów
│   └── experiments_archive/               # eksperymenty boczne (nie używane w głównym pipeline)
│       ├── main_48_cech_10wykonan.py
│       └── main_48_cech_ewma.py
│
├── notebooks/                             # ★ JUPYTER NOTEBOOKS
│   ├── TPM_Experiment_ModelSlice.ipynb
│   ├── TPM_Experiment_SliceAware.ipynb
│   ├── TPM_Experiment_SliceAware_BestOf5_v1.ipynb
│   ├── TPM_Experiment_SliceAware_QFServe_v3.ipynb
│   └── archive/                           # stare notebooki z `old/`
│       ├── TPM_Experiment_Classifiers.ipynb
│       ├── TPM_Experiment_Stability.ipynb
│       └── TPM_Experiment_Window_Size.ipynb
│
├── data/                                  # ★ DANE WEJSCIOWE
│   └── sample_data/                       # (lub przenieś zawartość 1 poziom wyżej)
│       ├── 2018.csv
│       ├── 2019.csv
│       ├── 2020.csv
│       ├── 2021.csv
│       ├── 2022.csv
│       ├── 2023.csv
│       └── 2024.csv
│
├── docs/                                  # ★ DOKUMENTACJA OPISOWA (dla promotora)
│   ├── ai_model_slice.md                  # główny opis Model Slicing dla promotora
│   ├── opis_main_48_cech.md               # opis baseline'u
│   ├── opis_main_48_cech_modelslice.md
│   ├── opis_main_48_cech_sliceaware.md
│   ├── opis_main_48_cech_sliceaware_bestof5_v1.md
│   ├── opis_main_48_cech_sliceaware_qfserve_v3.md
│   ├── opis_main_48_cech_slicecompare.md
│   └── papers/                            # ★ literatura, artykuły, prace
│       ├── GuideAI25_2.pdf                # Model Slicing for Responsible AI
│       └── dryja_thesis.pdf
│
├── reports/                               # ★ RAPORTY EKSPERYMENTÓW (wyniki, analizy)
│   ├── RAPORT_main_48_cech_sliceaware_i_slicecompare.md
│   ├── RAPORT_main_48_cech_vs_modelslice.md
│   ├── RAPORT_main_48_cech_warianty_slice_podsumowanie.md
│   └── outputs/                           # ★ wygenerowane raporty (XLSX, CSV, PNG)
│       ├── slice_comparison_all_variants.xlsx
│       ├── slice_comparison_baseline_vs_sliceaware.xlsx
│       ├── slice_comparison_baseline_vs_sliceaware.csv
│       └── reliability_diagram.png        # jak będzie generowany
│
├── logs/                                  # ★ LOGI URUCHOMIEŃ
│   ├── baseline_run.log
│   └── slicecompare_run.log
│
└── .venv/                                 # środowisko Pythona (gitignore)
```

## Co jest co — krótko

| Folder | Co tam trzymać |
|---|---|
| `src/` | Wszystkie pliki `.py` z kodem modeli i pipeline'em |
| `src/experiments_archive/` | Boczne eksperymenty których NIE używasz w głównym flow (np. 10wykonan, ewma) — żeby nie myliły z aktywnymi |
| `notebooks/` | Wszystkie `.ipynb` — Twoje + przeniesione z `old/` |
| `data/` | Surowe dane wejściowe — CSV z Jeff Sackmann |
| `docs/` | Dokumentacja OPISOWA — co i jak działa (dla promotora) |
| `docs/papers/` | PDF-y artykułów naukowych |
| `reports/` | Raporty z eksperymentów (analiza wyników, wnioski) — markdownowe |
| `reports/outputs/` | Pliki wynikowe generowane przez kod: XLSX, CSV, PNG |
| `logs/` | Stdout z uruchomień (dla debugowania) |

## Komendy do reorganizacji (PowerShell)

```powershell
# Z katalogu c:\Users\stasi\PycharmProjects\TenisPredictionModel uruchom:

# Utwórz strukturę
New-Item -ItemType Directory -Force src, src\experiments_archive, notebooks, notebooks\archive, docs, docs\papers, reports, reports\outputs, logs, data | Out-Null

# Przenieś kod
Move-Item main_48_cech.py, main_48_cech_modelslice.py, main_48_cech_sliceaware.py, main_48_cech_sliceaware_bestof5_v1.py, main_48_cech_sliceaware_qfserve_v3.py, main_48_cech_slicecompare.py, main_48_cech_seedstability.py src\
Move-Item main_48_cech_10wykonan.py, main_48_cech_ewma.py src\experiments_archive\

# Przenieś notebooki
Move-Item TPM_Experiment_*.ipynb notebooks\
Move-Item old\*.ipynb notebooks\archive\
Remove-Item old -Recurse -Force

# Przenieś dokumentację
Move-Item ai_model_slice.md, opis_*.md docs\
Move-Item GuideAI25_2.pdf, dryja_thesis.pdf docs\papers\

# Przenieś raporty
Move-Item RAPORT_*.md reports\
Move-Item slice_comparison_*.xlsx, slice_comparison_*.csv reports\outputs\

# Przenieś logi
Move-Item *.log logs\

# Przenieś dane (jeśli chcesz)
Move-Item sample_data data\

# Sprzątanie cache
Remove-Item __pycache__ -Recurse -Force
```

## ⚠ Co trzeba pamiętać po przenosinach

1. **Ścieżki w kodzie** — pliki `main_48_cech*.py` używają `pd.read_csv('sample_data/2024.csv')`. Po przeniesieniu do `src/` trzeba albo:
   - **Opcja A (prostsza)**: zmienić ścieżki na `'../data/sample_data/2024.csv'`
   - **Opcja B (lepsza)**: dodać na początku `BASE_DIR = Path(__file__).resolve().parent.parent` i używać `BASE_DIR / "data" / "sample_data" / "2024.csv"`

2. **Sub-process w `slicecompare.py`** — używa `runpy.run_path("main_48_cech_sliceaware.py")`. Po przeniesieniu wszystkich `.py` do `src/`, ścieżki w `MODELS` dict w slicecompare.py powinny działać (bo wszystkie są w tym samym `src/`).

3. **Notebooki** — używają `runpy.run_path("main_48_cech.py")`. Po przeniesieniu, trzeba zmienić na `runpy.run_path("../src/main_48_cech.py")`.

4. **Sample data import** — `main_48_cech_seedstability.py` używa `Path(__file__).with_name("main_48_cech.py")` — to ZAWSZE działa, niezależnie od katalogu.

5. **README.md** — warto stworzyć na samej górze, opisujący w 1 stronie co tu jest, jak uruchomić, jakie są wyniki. Nadrzędne narzędzie dla każdego (w tym promotora) wchodzącego do projektu.

## Krótszy alternative — jeśli wolisz minimum zmian

Jeśli nie chcesz ruszać kodu (ścieżek), zostaw `.py` w katalogu głównym, ale przenieś:

```
TenisPredictionModel/
├── *.py                  # zostają w głównym
├── sample_data/          # zostaje
├── notebooks/            # przeniesione .ipynb
├── docs/                 # opis_*.md, ai_model_slice.md, *.pdf
├── reports/              # RAPORT_*.md, *.xlsx, *.csv
├── logs/                 # *.log
└── old/                  # zostaje jako archiwum
```

To minimum-effort reorganizacja — uporządkuje wzrokowo bez ruszania kodu.

## Co zostawiam Tobie do decyzji

- Czy `data/sample_data/` ma 2 poziomy (`data/sample_data/`) czy 1 (`data/`)? Drugi prostszy, ale jak dodasz inne źródła danych w przyszłości (np. odds, weather), to `data/sample_data/` lepiej skaluje.
- Czy `notebooks/archive/` używać dla starych eksperymentów, czy zostawić `old/` jako jest. Archive zakomunikowane lepiej, że są STARE.
- `requirements.txt` — czy go potrzebujesz. Dla promotora ułatwia odtworzenie środowiska.
