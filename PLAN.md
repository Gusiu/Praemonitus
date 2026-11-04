# Praemonitus - plan projektu

Projekt zaliczeniowy z przedmiotu Systemy sztucznej inteligencji. Nazwa od łacińskiego *praemonitus, praemunitus* - "uprzedzony, uzbrojony": przewidzieć awarię, zanim się zdarzy.

> **Dla Claude Code:** przeczytaj cały dokument przed pierwszą zmianą w kodzie. Kamienie milowe realizuj po kolei (sekcja 10), po każdym zatrzymaj się i zapytaj. Decyzji z sekcji 2 nie zmieniaj bez pytania. Granice zakresu: sekcja 13. Pułapki: sekcja 14 - wróć do niej przed każdym kamieniem. Każdą decyzję techniczną dopisuj od razu do `docs/obrona.md` (sekcja 8.3).

---

## 1. Co budujemy

Aplikację, która na podstawie parametrów procesu obróbki (wariant produktu, temperatura powietrza i procesu, prędkość obrotowa, moment obrotowy, zużycie narzędzia) przewiduje, czy proces zakończy się awarią maszyny. Model jest trenowany na publicznym zbiorze AI4I 2020 z UCI. Aplikacja waliduje dane wejściowe, zwraca prawdopodobieństwo awarii i decyzję, zapisuje każde wywołanie do SQLite i pokazuje historię.

- **Problem:** klasyfikacja binarna, zmienna docelowa `Machine failure` (0/1).
- **Jedna obserwacja:** jeden przebieg procesu z jego parametrami i etykietą.
- **Dla kogo wynik:** dział utrzymania ruchu - decyzja, czy zatrzymać proces i sprawdzić maszynę.

Karta przedmiotu i specyfikacja projektu końcowego wymagają **klasyfikacji**, a lista tematów w "Ćwiczeniach i projektach" dopuszcza też regresję (tematy B i F). Klasyfikacja spełnia oba źródła. Odnotuj to w raporcie jednym zdaniem.

---

## 2. Decyzje (nie zmieniaj bez pytania)

1. **Dane:** AI4I 2020 Predictive Maintenance Dataset (UCI, CC BY 4.0) - dane syntetyczne, bez danych osobowych, z opisanymi regułami powstawania awarii (sekcja 4.3).
2. **Stos:** Python 3.12, pandas, NumPy, scikit-learn (>= 1.5, potrzebny `TunedThresholdClassifierCV`), matplotlib, seaborn, joblib, sqlite3 (biblioteka standardowa), Pydantic, Streamlit, pytest, Jupyter. Wersje przypięte w `requirements.txt`.
3. **Protokół oceny:** najpierw podział stratyfikowany 80/20 (`random_state=42`), dopiero potem cokolwiek innego. Wybór modelu i progu wyłącznie walidacją krzyżową na zbiorze treningowym (`StratifiedKFold(5, shuffle=True, random_state=42)`, te same foldy dla wszystkich modeli). Zbiór testowy używany **raz**, po zamrożeniu wyboru.
4. **Miary:** główna PR-AUC (average precision); do tego recall, precision i F1 klasy "awaria" przy wybranym progu oraz macierz pomyłek. Accuracy raportujemy tylko po to, żeby pokazać, że wprowadza w błąd.
5. **Próg decyzyjny:** dobierany kosztem błędów, koszt przeoczonej awarii (FN) do fałszywego alarmu (FP) = 10:1. To założenie, nie fakt - raport pokazuje wrażliwość wyniku dla 5:1, 10:1 i 20:1.
6. **Przeciek danych:** `UDI`, `Product ID`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF` nigdy nie są cechami. Flagi trybów awarii służą wyłącznie do analizy błędów po fakcie.
7. **Cechy pochodne liczone wewnątrz Pipeline:** aplikacja podaje surowe wartości, model sam liczy różnicę temperatur, moc i obciążenie narzędzia. Ten sam kod w treningu i w aplikacji, więc nie ma rozjazdu.
8. **Aplikacja:** wspólny rdzeń `ml/predictor.py` dla interfejsu Streamlit (`app.py`) i konsoli (`cli.py`); baza w `database.py`, wyłącznie zapytania parametryzowane.
9. **Raport:** jedno źródło `reports/raport.md`, z którego pandoc generuje `raport.docx` i `raport.pdf`.

---

## 3. Macierz zgodności

Wiersze to lista kontrolna z "Ćwiczeń i projektów" (pkt 7) oraz wymagania z karty i specyfikacji projektu końcowego, których lista nie powtarza. Przed tagiem `v1.0` każdy wiersz musi mieć działający dowód.

| Wymaganie | Realizacja | Gdzie |
|---|---|---|
| Problem i zmienna docelowa | sekcja 1, rozdział 1 raportu | `reports/raport.md`, notebook rozdz. 1 |
| Źródło, licencja, struktura danych | źródło, licencja, DOI, suma SHA-256, słownik kolumn | `data/SOURCE.md`, notebook rozdz. 2 |
| Braki, duplikaty, typy, rozkład klas | EDA + kontrola spójności etykiety z flagami trybów | notebook rozdz. 3 |
| Podział przed dopasowaniem transformacji | podział jako pierwszy krok, wszystkie transformacje w Pipeline | `ml/data.py`, `ml/train.py`, test `test_no_leakage.py` |
| Kilka modeli na porównywalnych danych | 4 modele x 2 zestawy cech + model bazowy, te same foldy | notebook rozdz. 5, `ml/train.py` |
| Miary dobrane do problemu i zinterpretowane | PR-AUC, recall/precision przy progu kosztowym, macierz pomyłek | notebook rozdz. 5-6 |
| Analiza wybranych błędów (specyfikacja, etap 3) | FN i FP według trybów awarii, przykłady rekordów | notebook rozdz. 7 |
| Wybór modelu nie z jednej liczby (specyfikacja, etap 3) | tabela kryteriów | notebook rozdz. 6 |
| Zapis i ponowne wczytanie modelu | `model.joblib` + karta modelu; test identycznych predykcji po wczytaniu | `models/`, `tests/test_model_io.py` |
| Aplikacja waliduje dane i pokazuje predykcję | walidacja Pydantic, prawdopodobieństwo, decyzja, flaga "poza zakresem" | `ml/validation.py`, `app.py`, `cli.py` |
| Baza: czas, dane wejściowe, wynik; odczyt historii | tabela `predictions`, widok historii, eksport CSV | `database.py` |
| Bez zbędnych danych osobowych (specyfikacja, etap 4) | dane syntetyczne, baza nie zapisuje użytkownika | `database.py`, raport |
| Instrukcja uruchomienia, raport, prezentacja | README, `raport.pdf` + `raport.docx`, scenariusz pokazu | katalog główny, `docs/pokaz.md` |
| Ograniczenia i źródła błędów | rozdział raportu + przygotowanie do obrony | `reports/raport.md`, `docs/obrona.md` |

**Ponad minimum** (zadania rozszerzające z ćwiczeń, wykorzystane w projekcie): słownik kolumn (ćw. 2), korelacje i redundancja cech (ćw. 3), walidacja krzyżowa i strojenie (ćw. 5), Pipeline z cechami liczbowymi i kategorycznymi (ćw. 7), obsługa błędnych danych wejściowych (ćw. 7), identyfikator modelu w bazie i eksport historii do CSV (ćw. 8), interfejs Streamlit (rozszerzenie środowiska).

---

## 4. Dane (etap 1)

### 4.1 Źródło

- UCI Machine Learning Repository: *AI4I 2020 Predictive Maintenance Dataset*, DOI `10.24432/C5HS5C`, przekazany 29.08.2020.
- Licencja CC BY 4.0 - wolno kopiować i dołączyć plik do repozytorium pod warunkiem podania autorstwa i źródła.
- 10 000 rekordów, według UCI bez braków. Plik `ai4i2020.csv`, ok. 510 KB.
- Praca źródłowa: S. Matzka, *Explainable Artificial Intelligence for Predictive Maintenance Applications*, International Conference on Artificial Intelligence for Industries (AI4I), 2020.
- Dane są **syntetyczne**, co spełnia zalecenie kursu dotyczące danych bez danych osobowych.

### 4.2 Pozyskanie

UCI jest niedostępne z sesji chmurowej (sprawdzone 4.11.2026), więc plik pobrał użytkownik ze strony UCI i jest już w `data/raw/ai4i2020.csv`, w postaci oryginalnej. `data/SOURCE.md` zawiera pełne przypisanie (tytuł, UCI, DOI, licencja, praca źródłowa), datę pobrania i SHA-256. Test `tests/test_data.py` sprawdza sumę, liczbę wierszy i nazwy kolumn. Kod nigdy nie pobiera danych z sieci.

Plik ma kodowanie UTF-8 z BOM i końce linii CRLF. `.gitattributes` oznacza go jako `-text`, żeby git na Windowsie i Linuksie nie zmieniał końców linii (inaczej suma SHA-256 przestałaby się zgadzać). Wczytanie: `pd.read_csv(..., encoding="utf-8-sig")`.

### 4.3 Kolumny i reguły powstawania awarii

Kolumny (sprawdź dokładne nazwy w pliku): `UDI`, `Product ID`, `Type` (L/M/H), `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.

Przy wczytaniu nazwy zamieniamy na snake_case: `type`, `air_temp_k`, `process_temp_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`, `machine_failure`.

Reguły z opisu zbioru na UCI - to one tworzą etykietę:

| Tryb | Reguła | Liczba przypadków wg UCI |
|---|---|---|
| TWF - zużycie narzędzia | narzędzie wymieniane albo psuje się w losowym momencie przy zużyciu 200-240 min | 120 (69 wymian, 51 awarii) |
| HDF - odprowadzanie ciepła | różnica temperatur procesu i powietrza < 8,6 K **i** prędkość < 1380 rpm | 115 |
| PWF - moc | moc (moment x prędkość kątowa w rad/s) < 3500 W albo > 9000 W | 95 |
| OSF - przeciążenie | zużycie narzędzia x moment > 11 000 minNm (L), 12 000 (M), 13 000 (H) | 98 |
| RNF - losowa | 0,1% szans na awarię niezależnie od parametrów | 5 |

`Machine failure = 1`, gdy prawdziwy jest co najmniej jeden tryb. Według UCI model nie wie, który tryb spowodował awarię.

### 4.4 Fakty o pliku (sprawdzone 4.11.2026 na pliku z repozytorium)

EDA w notebooku musi te liczby policzyć sama - tu są po to, żeby było wiadomo, czego się spodziewać.

- 10 000 wierszy, 14 kolumn, bez braków i bez duplikatów; `UDI` i `Product ID` unikalne, `Product ID` zaczyna się literą wariantu z `Type`.
- **Awarie: 339 (3,39%).** Model, który zawsze odpowiada "brak awarii", ma accuracy 96,61%.
- Warianty: L 6000, M 2997, H 1003; odsetek awarii L 3,92%, M 2,77%, H 2,09%.
- Zakresy: temperatura powietrza 295,3-304,5 K, procesu 305,7-313,8 K, prędkość 1168-2886 rpm, moment 3,8-76,6 Nm, zużycie 0-253 min. Temperatura procesu zawsze wyższa od powietrza (różnica 7,6-12,1 K), korelacja obu temperatur 0,876.
- **Reguły HDF, PWF i OSF z tabeli 4.3 odtwarzają flagi dokładnie** (zgodność 10 000/10 000 dla każdej) i wyjaśniają 287 z 339 awarii (84,7%). Pozostałe 52 awarie: 43 z flagą TWF i 9 bez żadnej flagi.

**Rozbieżności między opisem UCI a plikiem** - opisz je w raporcie, nie poprawiaj danych:

| Element | Opis na UCI | Plik |
|---|---|---|
| TWF | 120 przypadków, w tym 51 awarii; okno 200-240 min | 46 flag; zużycie przy fladze 198-253 min |
| RNF | 5 przypadków | 19 flag, z czego tylko 1 przy awarii |
| Spójność etykiety | awaria = co najmniej jeden tryb | 9 awarii bez żadnej flagi; 18 flag RNF bez awarii |

Decyzja: `Machine failure` zostaje etykietą bez zmian. Rozbieżności idą do rozdziału o ograniczeniach danych, a 9 awarii bez flagi do analizy błędów jako przypadki, których żadna reguła nie wyjaśnia.

---

## 5. Analiza i przygotowanie danych (etap 2)

Notebook `notebooks/analiza_i_model.ipynb`, rozdziały 1-4, logika w `ml/data.py`:

1. **Problem** - opis z sekcji 1.
2. **Dane** - źródło, licencja, słownik kolumn (nazwa, typ, jednostka, opis, rola: cecha, etykieta, identyfikator, wykluczona).
3. **Jakość danych:**
   - typy, braki, duplikaty całych wierszy, unikalność `UDI` i `Product ID`,
   - rozkład klas ogółem i według `Type`,
   - wartości odstające (wykresy pudełkowe według klasy),
   - **spójność etykiety:** zestawienie `Machine failure` z flagami trybów oraz porównanie z opisem UCI (spodziewane wyniki w sekcji 4.4). Policz, opisz i uzasadnij decyzję - etykieta zostaje bez zmian, bo to ona jest celem.
4. **EDA:** co najmniej trzy wykresy - histogramy cech, wykresy pudełkowe według klasy, prędkość obrotowa vs moment z kolorem awarii (widać pasma mocy z reguły PWF), mapa korelacji (temperatura powietrza i procesu są silnie skorelowane - redundancja). Trzy obserwacje i dwa ograniczenia danych.

Zasady przeciwdziałania przeciekowi:

- podział train/test to pierwsza operacja po wczytaniu i kontroli jakości,
- skalowanie, kodowanie `Type` i cechy pochodne wyłącznie wewnątrz Pipeline,
- walidacja krzyżowa tylko na zbiorze treningowym,
- zbiór testowy w kamieniu M2 dotykany jeden raz, w osobnej, oznaczonej komórce.

---

## 6. Trenowanie i testowanie (etap 3)

Notebook rozdziały 5-8, logika w `ml/features.py`, `ml/train.py`, `ml/evaluate.py`.

### 6.1 Zestawy cech

- **raw:** `type`, `air_temp_k`, `process_temp_k`, `rotational_speed_rpm`, `torque_nm`, `tool_wear_min`,
- **physics:** raw + `temp_diff_k = process_temp_k - air_temp_k`, `power_w = torque_nm * rotational_speed_rpm * 2π / 60`, `strain_min_nm = tool_wear_min * torque_nm`.

Cechy pochodne powstają przez `FunctionTransformer` w Pipeline, wyłącznie z wartości dostępnych w chwili predykcji. To wiedza dziedzinowa (moc, zapas termiczny, obciążenie narzędzia), nie przeciek - w raporcie wyjaśnij to wprost.

### 6.2 Modele

Te same foldy, ten sam podział, każdy model w obu zestawach cech:

| Model | Konfiguracja |
|---|---|
| `DummyClassifier(strategy="most_frequent")` | punkt odniesienia: pokazuje pułapkę accuracy |
| `LogisticRegression` | `StandardScaler` + `OneHotEncoder(type)`, `class_weight="balanced"`, `max_iter=2000` |
| `DecisionTreeClassifier` | `class_weight="balanced"`, `max_depth` z siatki `{3, 4, 5, 6, 8}`; model interpretowalny |
| `RandomForestClassifier` | `n_estimators=300`, `class_weight="balanced_subsample"` |
| `HistGradientBoostingClassifier` | mała siatka `learning_rate` i `max_depth` |

Strojenie: `GridSearchCV` z `scoring="average_precision"` na foldach treningowych, małe siatki. Tabela wyników: średnia i odchylenie standardowe po foldach dla PR-AUC, ROC-AUC, recall, precision, F1 i accuracy.

### 6.3 Próg i wybór modelu

- Próg: `TunedThresholdClassifierCV` z własnym scorerem kosztowym (koszt = 10 · FN + 1 · FP, minimalizowany), walidacja krzyżowa na zbiorze treningowym.
- Wrażliwość: próg i koszt dla proporcji 5:1, 10:1 i 20:1.
- Tabela kryteriów wyboru: PR-AUC, recall przy progu kosztowym, liczba fałszywych alarmów na 1000 przebiegów, stabilność między foldami, interpretowalność, czas predykcji. Wybór uzasadniony słowami, nie jedną liczbą.

### 6.4 Test, analiza błędów, wyjaśnialność

- **Test, raz:** raport klasyfikacji, macierz pomyłek, krzywa precision-recall, koszt przy progu - zestawione z modelem bazowym.
- **Analiza błędów:** FN i FP rozbite według trybów awarii (flagi użyte wyłącznie tutaj) oraz 10 przykładowych błędnych rekordów z komentarzem. Spodziewane: awarie z reguł HDF, PWF i OSF (84,7%) są w zasięgu modelu; TWF jest losowe w oknie zużycia, a 9 awarii bez żadnej flagi nie wyjaśnia nic w danych. To nieusuwalna część błędu - nazwij ją wprost i porównaj recall modelu z tym pułapem.
- **Wyjaśnialność:** `permutation_importance` na zbiorze testowym (scoring `average_precision`). Dla drzewa decyzyjnego wypisz reguły (`export_text`) i porównaj progi podziałów z regułami z sekcji 4.3: czy drzewo samo odkryło granice 8,6 K, 1380 rpm, 3500 i 9000 W, 11 000-13 000 minNm? To najmocniejszy fragment projektu - model odtwarza mechanizm, którego nikt mu nie podał.

### 6.5 Zapis

- `models/model.joblib` - cały wytrenowany estymator z progiem (Pipeline w `TunedThresholdClassifierCV` albo `FixedThresholdClassifier`),
- `models/model_card.json` - `model_id` (krótki hash), data treningu, wersje Pythona i scikit-learn, SHA-256 danych, lista cech, zakresy wartości ze zbioru treningowego (dla walidacji w aplikacji), próg, proporcja kosztów, metryki z CV i testu, SHA-256 pliku `model.joblib`,
- `python -m ml.train` odtwarza model deterministycznie z `data/raw/ai4i2020.csv`.

---

## 7. Aplikacja i baza (etap 4)

### 7.1 Rdzeń `ml/predictor.py`

1. Wczytanie modelu: najpierw porównanie SHA-256 pliku z kartą modelu, potem `joblib.load`. Joblib to pickle - wczytujemy wyłącznie własny, zweryfikowany plik (materiały do ćwiczeń zwracają na to uwagę).
2. Walidacja (`ml/validation.py`, Pydantic):
   - **twarda** (odrzucenie z czytelnym komunikatem): `type` w {L, M, H}; liczby skończone; temperatury > 0 K, prędkość > 0, moment i zużycie >= 0,
   - **miękka** (predykcja z ostrzeżeniem): wartość poza zakresem ze zbioru treningowego -> `out_of_range = True`.
3. Wynik: prawdopodobieństwo awarii, próg, decyzja, cechy pochodne (różnica temperatur, moc, obciążenie - zrozumiałe dla mechanika), flaga zakresu, `model_id`.
4. Zapis do bazy przy każdym wywołaniu.

### 7.2 Baza `database.py` (SQLite)

```sql
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,        -- UTC, ISO 8601
  model_id TEXT NOT NULL,
  source TEXT NOT NULL,            -- streamlit | cli | batch
  input_json TEXT NOT NULL,
  probability REAL NOT NULL,
  threshold REAL NOT NULL,
  prediction INTEGER NOT NULL,     -- 1 = awaria
  out_of_range INTEGER NOT NULL,
  latency_ms REAL NOT NULL
);
```

Wyłącznie zapytania z placeholderami `?`. Funkcje: zapis, historia (ostatnie N, filtr po decyzji), eksport do CSV. Połączenie otwierane na operację. Baza nie przechowuje danych o użytkowniku.

### 7.3 Interfejsy

- **`app.py` (Streamlit):** formularz z zakresami z karty modelu, wynik z prawdopodobieństwem i decyzją, ostrzeżenie "poza zakresem danych treningowych", tabela historii, przycisk eksportu CSV, rozwijana informacja o modelu (`model_id`, data, metryki). Model buforowany przez `st.cache_resource`.
- **`cli.py` (konsola):** interaktywne pytania z ponawianiem przy błędnej wartości, `--history N`, `--export plik.csv`, `--batch plik.csv` (predykcja dla wielu wierszy z zapisem do bazy).

---

## 8. Dokumentacja, prezentacja, obrona (etap 5)

### 8.1 README

Cel, wymagania (Python 3.12), instalacja (`pip install -r requirements.txt`), uruchomienie (`streamlit run app.py`, `python cli.py`), odtworzenie modelu (`python -m ml.train`), testy (`pytest`), struktura plików, lista bibliotek, przypisanie autorstwa danych.

### 8.2 Raport

Źródło `reports/raport.md` według szablonu z materiałów do ćwiczeń (rozdz. 4): opis problemu, dane, analiza, przygotowanie, modele, ocena, aplikacja, baza, dokumentacja, ograniczenia i kierunki rozwoju. Wykresy z notebooka zapisane do `reports/figures/`.

Generowanie:

```
pandoc reports/raport.md -o raport.docx
pandoc reports/raport.md -o raport.pdf --pdf-engine=xelatex -V mainfont="DejaVu Serif"
```

Ograniczenia, które raport musi nazwać: dane syntetyczne z regułami prostszymi niż w realnym zakładzie; RNF i część TWF losowe z definicji; próg zależy od przyjętych kosztów; każdy przebieg traktowany niezależnie, bez historii czujników w czasie; model nie wie, który tryb awarii wystąpił.

Kierunki rozwoju: dane rzeczywiste i cechy z szeregów czasowych, klasyfikacja typu awarii, zapisywanie faktycznego wyniku procesu w bazie i monitorowanie jakości modelu.

### 8.3 `docs/obrona.md`

Kod pisze agent, a na obronie decyzje trzeba umieć wyjaśnić samemu. Plik zawiera pytania i krótkie odpowiedzi, uzupełniane na bieżąco w trakcie pracy. Minimum:

1. Dlaczego nie accuracy? (model stały ma 96,61%)
2. Dlaczego PR-AUC, a nie ROC-AUC? (przy rzadkiej klasie ROC-AUC zawyża wynik, bo prawdziwie negatywnych jest bardzo dużo)
3. Skąd próg i co znaczy 10:1? (założenie kosztów, analiza wrażliwości)
4. Dlaczego usunięto TWF, HDF, PWF, OSF, RNF? (definiują etykietę - przeciek)
5. Czy cechy pochodne to nie przeciek? (liczone z danych dostępnych przy predykcji)
6. Jak uniknięto przecieku przy skalowaniu? (podział najpierw, Pipeline, walidacja krzyżowa na treningu)
7. Dlaczego ten model? (tabela kryteriów)
8. Czego model nie wykryje i dlaczego? (część TWF i 9 awarii bez żadnej flagi)
9. Czym różnią się dane syntetyczne od rzeczywistych?
10. Dlaczego sprawdzamy sumę kontrolną przed wczytaniem modelu? (joblib = pickle)

### 8.4 Pokaz

`docs/pokaz.md` - scenariusz na 3 minuty. Wartości kandydujące policzone z reguł UCI; przed wpisaniem do scenariusza sprawdź, co faktycznie zwraca model:

| Przypadek | Type | Powietrze [K] | Proces [K] | Prędkość [rpm] | Moment [Nm] | Zużycie [min] | Oczekiwane |
|---|---|---|---|---|---|---|---|
| normalny | M | 300,0 | 310,0 | 1500 | 40 | 100 | brak awarii (moc 6283 W) |
| mała moc | M | 300,0 | 310,0 | 1400 | 15 | 100 | awaria PWF (moc 2199 W) |
| przegrzanie | M | 302,0 | 310,4 | 1300 | 45 | 50 | awaria HDF (różnica 8,4 K) |

Dalej: temperatura powietrza 310 K -> ostrzeżenie "poza zakresem"; `Type` = X albo ujemne zużycie -> odrzucenie; historia i eksport CSV.

---

## 9. Testy

| Test | Co sprawdza |
|---|---|
| `test_data.py` | SHA-256, 10 000 wierszy, nazwy kolumn |
| `test_no_leakage.py` | lista cech nie zawiera identyfikatorów ani flag trybów; preprocessing jest częścią Pipeline |
| `test_features.py` | wzory cech pochodnych na ręcznie policzonych przykładach |
| `test_validation.py` | wartości poprawne, błędne, graniczne i spoza zakresu |
| `test_model_io.py` | identyczne predykcje przed i po zapisie; odrzucenie pliku z niezgodną sumą |
| `test_predictor.py` | przypadek ewidentnej awarii mocy daje decyzję "awaria" |
| `test_database.py` | zapis, historia, eksport CSV; próba wstrzyknięcia SQL zapisana jako zwykły tekst |
| `test_app.py` | `streamlit.testing.v1.AppTest`: formularz, wynik, historia |

Notebook wykonywany od początku do końca przed każdym commitem kamienia (`jupyter nbconvert --to notebook --execute --inplace`). Wyniki zostają w pliku, żeby prowadzący widział je na GitHubie bez uruchamiania.

---

## 10. Kamienie milowe

Skala: karta przewiduje 15 h (stacjonarne) / 10 h (niestacjonarne) zajęć projektowych i 10 h / 15 h pracy własnej na prace projektowe. Zakres jest do tego dopasowany - nie dokładaj nic spoza planu.

- **M0 - szkielet:** struktura repo, `requirements.txt`, sprawdzenie środowiska (sekcja 11), `data/SOURCE.md`, `test_data.py`. Warunek: dane w repo, test zielony.
- **M1 - dane (etapy 1-2):** notebook rozdz. 1-4, `ml/data.py`, słownik kolumn, kontrola spójności etykiety, wykresy. Warunek: notebook wykonuje się w całości.
- **M2 - modele (etap 3):** rozdz. 5-8, `ml/features.py`, `ml/train.py`, `ml/evaluate.py`, zapis modelu i karty, testy cech, przecieku i zapisu modelu.
- **M3 - aplikacja (etap 4):** `ml/predictor.py`, `ml/validation.py`, `database.py`, `app.py`, `cli.py`, pozostałe testy.
- **M4 - dokumentacja (etap 5):** README, raport (`.docx` i `.pdf`), `docs/obrona.md`, `docs/pokaz.md`, przejście macierzy zgodności (sekcja 3), tag `v1.0`.

Po każdym kamieniu: zielone testy, commit, zatrzymanie i pytanie.

---

## 11. Środowisko

Sprawdzone 4.11.2026 w kontenerze chmurowym, w którym powstał ten plan (sesja, w której będziesz budować, może mieć inne ustawienia sieci):

- UCI, OpenML i Kaggle są zablokowane przez proxy - dane wyłącznie z `data/raw/`.
- PyPI działa; dostępny był scikit-learn 1.9.1 z `TunedThresholdClassifierCV`.
- pandoc 3.1.3, XeLaTeX i LibreOffice były dostępne; konwersja Markdown -> DOCX i PDF z polskimi znakami (font DejaVu Serif) przetestowana. Gdyby XeLaTeX zniknął: `soffice --headless --convert-to pdf raport.docx`.
- Interfejsu Streamlit nie zobaczysz z sesji chmurowej - weryfikacja przez `AppTest`, pokaz na własnym komputerze (`streamlit run app.py`).

---

## 12. Struktura repozytorium

Zgodna z minimalną strukturą z "Ćwiczeń i projektów" (pkt 6), rozszerzona o kod modułów, testy i dokumenty:

```
praemonitus/
  README.md  PLAN.md  CLAUDE.md  requirements.txt
  raport.pdf  raport.docx            generowane z reports/raport.md
  app.py                             interfejs Streamlit
  cli.py                             interfejs konsolowy
  database.py                        SQLite
  data/
    raw/ai4i2020.csv
    SOURCE.md                        źródło, licencja, DOI, SHA-256
  notebooks/analiza_i_model.ipynb
  ml/                                data, features, train, evaluate, validation, predictor
  models/model.joblib  models/model_card.json
  reports/raport.md  reports/figures/
  docs/obrona.md  docs/pokaz.md
  tests/
```

Plik bazy (`predictions.db`) w `.gitignore`.

---

## 13. Poza zakresem

- sieci neuronowe, LLM, AutoML,
- SHAP i inne ciężkie biblioteki wyjaśnialności (wystarczy permutation importance i reguły drzewa),
- MLflow, Docker, wdrożenie w chmurze, Flask obok Streamlit,
- klasyfikacja typu awarii (wieloklasowa) - tylko jeśli po M4 zostanie czas,
- duże siatki hiperparametrów.

---

## 14. Pułapki

1. **Flagi trybów awarii** definiują etykietę - jako cechy dałyby niemal idealny wynik, który nic nie znaczy.
2. **`Product ID`** zawiera literę wariantu i numer seryjny - usuń; wariant jest w `Type`.
3. **Accuracy** modelu stałego przekracza 95% - nigdy nie wybieraj modelu po accuracy.
4. **Podział stratyfikowany**, inaczej rzadka klasa rozkłada się nierówno między zbiory.
5. **Próg** dobieraj na foldach treningowych, nigdy na zbiorze testowym.
6. **Zbiór testowy** raz, po zamrożeniu wyboru. Jeśli po teście zmienisz model, opisz to uczciwie w raporcie.
7. **Nazwy kolumn** z nawiasami i spacjami - zamień na snake_case przy wczytaniu, w jednym miejscu (`ml/data.py`).
8. **Wersja scikit-learn** musi być ta sama przy zapisie i wczytaniu modelu - przypnij ją i zapisz w karcie modelu.
9. **Streamlit** uruchamia skrypt od nowa przy każdej interakcji - model w `st.cache_resource`, połączenie z bazą otwierane na operację.
10. **Czas** w bazie w UTC (ISO 8601); w interfejsie można pokazać czas lokalny.
11. **Polskie znaki w PDF** - XeLaTeX z fontem obsługującym polskie znaki (DejaVu Serif), nie pdfLaTeX z domyślnym fontem.
12. **Notebook** commituj po pełnym wykonaniu, bez ręcznie poprawianych wyników.
13. **BOM i CRLF w pliku danych** - wczytuj z `encoding="utf-8-sig"`; nie zapisuj pliku ponownie przez pandas ani edytor, bo zmieni się suma SHA-256.
