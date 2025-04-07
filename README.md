# LatentMol

Ein Deep Learning Framework für molekulare Sequenzen basierend auf Transformern.

## Beschreibung

LatentMol: Framework zur Verarbeitung und Analyse von molekularen Strukturen mittels Deep Learning. Es transformiert chemische Moleküle in spezielle Sequenzen, die für Transformer-basierte Modelle geeignet sind.

## Hauptmerkmale

- Molekül-zu-Sequenz Konvertierung
- Automatische Datenvorverarbeitung
- Molekül-Augmentierung
- Parallele Verarbeitung

## Installation

1. Klonen Sie das Repository:
```bash
git clone git@github.com:n-Baiersdorf/LatentMol.git
cd LatentMol
```

2. Erstellen Sie eine virtuelle Umgebung:
```bash
python -m venv venv
source venv/bin/activate  # Unter Linux/Mac
# oder
.\venv\Scripts\activate  # Unter Windows
```

3. Installieren Sie die Abhängigkeiten:
```bash
pip install -r requirements.txt
```

## Verwendung

### Datenvorverarbeitung

1. Laden Sie die PubChem-Daten herunter
2. Führen Sie die Vorverarbeitung durch:
```python
from main import Verarbeiter

processor = Verarbeiter()
processor.prepare_raw_data("pfad_zu_ihren_daten")
processor._make_sequence_data()
```

### Molekül-Augmentierung

```python
from main import Verarbeiter

processor = Verarbeiter()
processor.augment_molecules(
    input_path="pfad_zu_ihren_molekuelen",
    output_dir="augmented_output",
    max_variants_per_molecule=5,
    random_selection=True
)
```

## Konfiguration

Die Hauptkonfigurationsparameter befinden sich in `main.py`:

- `ATOM_DIMENSION`: Dimension der Atom-Konstanten
- `MAX_BONDS`: Maximale Anzahl von Bindungen
- `MOL_MIN_LENGTH`: Minimale Moleküllänge
- `MOL_MAX_LENGTH`: Maximale Moleküllänge
- `MAX_PERMUTATIONS`: Maximale Anzahl von Augmentierungen

## Verzeichnisstruktur

```
LatentMol/
├── PreprocessingPipeline/     # Vorverarbeitungs-Pipeline
├── misc/                      # Hilfsfunktionen
├── main.py                    # Hauptskript
├── requirements.txt           # Abhängigkeiten
└── README.md                  # Diese Datei
```

## Anforderungen

- Python 3.8+
- RDKit
- NumPy
- Pandas
- tqdm
- psutil

## Lizenz

MIT License

## Autor

Noah Baiersdorf

## Kontakt

Bei Fragen oder Problemen, öffnen Sie bitte ein Issue im GitHub Repository.

