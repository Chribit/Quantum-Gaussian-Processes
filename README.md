# Quantum-Gaussian-Processes

Building gaussian process kernels using quantum computers to predict financial time series data. Work performed for university.

## Für Jonathan ❤️ oder andere Lehrstuhl Mitarbeiter die diesen Code anschauen dürfen oder müssen

Alle Skripte mit `main_*.py` beinhalten routinen für z.B. die Auswertungen in meinem Report. Alle anderen `*.py` Skripte sind Module die spezifische Funktionalität beinhalten.

- `data.py`: Daten-Generierung und -Formatierung
- `evaluation.py`: Fitness Funktionen
- `evolution.py`: Genetischer Algorithmus und Bestandteile
- `kernel.py`: Klassische Kernel Unterfunktionen und Kombinierte Funktionen
- `model.py`: Gaußscher Prozess Klasse
- `plot.py`: Vorgefertigte plotting Funktionen

Alle `main_*.py` Skripte sollten mit einem `run_*.sh` Skript mit gleichem Suffix ausgeführt werden um die richtigen Environment Variablen zu setzen. Alle Skripte die Datensets generieren brauchen die Environment Variable `PYTHONHASHSEED` mit einem festen Wert wie z.b. `42069`. Die Seed Strings für die Datensets produzieren sonst nicht die gleichen Ergebnisse. Alle Skripte die den genetischen Algorithmus benutzen brauchen die Environment Variable `OMP_NUM_THREADS` um die Anzahl der Verfügbaren Threads zu setzen. Mehr Threads erlaubt für schnellere Durchläufe bei der Evaluierung der Individuen einer Generation.

Nochmal eine spezifische Erklärung der Parameter für die `evolve()` Funktion:

``` python
evolve (
    model,
    gene_reader,
    gene_count,
    fitness_granularity,
    fitness_threshold,
    cycles,
    population_size,
    mutation_probability,
    crossover_probability,
    logging = False,
    plotting = False,
    filename = "evolution_timeline"
) => (best_genes, cycle_count)
```

- `model` -> Gaußscher Prozess Objekt,
- `gene_reader` -> Interpretier Funktion die Gene zu z.B. model Parameter umwandelt,
- `gene_count` -> Gen Anzahl pro Individuum z.B. Anzahl der Kernel Parameter,
- `fitness_granularity` -> Anzahl der Unterteilungen für die AUC Approximation,
- `fitness_threshold` -> Ziel Fitness die den Prozess frühzeitig abbricht,
- `cycles` -> Maximale Generations-Anzahl,
- `population_size` -> Anzahl der Individuen pro Population einer Generation,
- `mutation_probability` -> Wahrscheinlichkeit einer Mutations Events,
- `crossover_probability` -> Wahrscheinlichkeit eines Crossover Events,
- `logging` -> Ob in der Konsole der Fortschritt beschrieben werden soll,
- `plotting` -> Ob ein Evolutions Verlauf Plot erstellt werden soll,
- `filename` -> Dateiname relativ zu einem `images/` Ordner für den Plot (ohne .png)

Die Funktion gibt ein Tuple aus, welches an erster Stelle ein Array der besten Gene enthält und an zweiter Stelle die Anzahl an Zyklen / Generationen die gebraucht wurden um diese Gene zu entwickeln.