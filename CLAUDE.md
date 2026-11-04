# CLAUDE.md

Projekt: **Praemonitus** - zaliczenie projektu z Systemów sztucznej inteligencji. Pełna specyfikacja: `PLAN.md`. Przeczytaj ją przed pierwszą zmianą w kodzie.

## Zasady pracy

- Kamienie milowe po kolei (`PLAN.md`, sekcja 10). Kamień kończy się: zielone testy, notebook wykonany od początku do końca, commit.
- Po każdym kamieniu zatrzymaj się, pokaż, co powstało, i zapytaj przed następnym.
- Decyzji z sekcji 2 nie zmieniaj bez pytania. Zakres pilnuj według sekcji 13. Pułapki: sekcja 14.
- Dane wyłącznie z `data/raw/`. Żadnego pobierania z sieci w kodzie.
- Zbiór testowy używany raz, po zamrożeniu wyboru modelu i progu. Flagi trybów awarii nigdy nie są cechami.
- Każdą decyzję techniczną dopisuj od razu do `docs/obrona.md` jako pytanie i krótką odpowiedź.
- Liczby w raporcie i README tylko z wykonanego kodu, nigdy wpisywane z pamięci.
- Identyfikatory w kodzie po angielsku. Notebook, UI i dokumentacja po polsku. W polskich dokumentach zwykły łącznik `-`, bez półpauz i pauz.
- Przed tagiem `v1.0` przejdź macierz zgodności (sekcja 3): każdy wiersz musi mieć działający dowód.
