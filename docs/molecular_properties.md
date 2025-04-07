# Molekulare Eigenschaften in LatentMol

Dieses Dokument beschreibt die Integration molekularer Eigenschaften in das LatentMol-Format.

## Übersicht

LatentMol wurde um die Berechnung und Speicherung wichtiger molekularer Eigenschaften erweitert. Diese Eigenschaften sind für das Training von Deep-Learning-Modellen und QSAR-Analysen (Quantitative Struktur-Aktivitäts-Beziehungen) nützlich.

Folgende Eigenschaften werden berechnet:

1. **Molekularmasse** - Die Masse des Moleküls in g/mol
2. **HBA (Hydrogen Bond Acceptors)** - Anzahl der Wasserstoffbrückenakzeptoren
3. **HBD (Hydrogen Bond Donors)** - Anzahl der Wasserstoffbrückendonoren
4. **Rotierende Bindungen** - Anzahl der rotierenden Bindungen im Molekül
5. **Aromatische Ringe** - Anzahl der aromatischen Ringe
6. **LogP** - Logarithmus des Oktanol-Wasser-Verteilungskoeffizienten

## Datenformat

Die molekularen Eigenschaften werden direkt in die JSON-Dateien integriert, die auch die Molekülsequenzen enthalten. Das Format wurde wie folgt angepasst:

**Altes Format:**
```json
{
  "molecule_1": [
    [0.0, 0.0, 0.0, ...],
    [0.1, 0.2, 0.3, ...],
    ...
  ],
  "molecule_2": [
    ...
  ]
}
```

**Neues Format:**
```json
{
  "molecule_1": {
    "sequence": [
      [0.0, 0.0, 0.0, ...],
      [0.1, 0.2, 0.3, ...],
      ...
    ],
    "properties": {
      "molekularmasse": 45.06,
      "hba": 2,
      "hbd": 1,
      "rotierende_bindungen": 3,
      "aromatische_ringe": 0,
      "logp": -0.75
    }
  },
  "molecule_2": {
    ...
  }
}
```

## Verwendung

### Automatische Berechnung in der Verarbeitungspipeline

Die molekularen Eigenschaften werden automatisch während der Sequenzgenerierung berechnet. Der Prozess ist vollständig in die Hauptverarbeitungspipeline integriert.

### Nachträgliche Berechnung für bestehende Daten

Für bereits generierte Sequenzdaten können die Eigenschaften mit dem Tool `scripts/add_molecular_properties.py` hinzugefügt werden:

```bash
# Eigenschaften zu einer einzelnen Datei hinzufügen
python scripts/add_molecular_properties.py --input path/to/sequences.json --output path/to/output.json

# Eigenschaften zu allen Dateien in einem Verzeichnis hinzufügen (überschreibt die Originaldateien)
python scripts/add_molecular_properties.py --input path/to/directory

# Eigenschaften zu allen Dateien in einem Verzeichnis und seinen Unterverzeichnissen hinzufügen
python scripts/add_molecular_properties.py --input path/to/directory --recursive --output path/to/output_dir
```

## Technische Details

### Berechnung der Eigenschaften

Die Eigenschaften werden mit RDKit berechnet:

- **Molekularmasse** - `Descriptors.MolWt(mol)`
- **HBA** - `Lipinski.NumHAcceptors(mol)`
- **HBD** - `Lipinski.NumHDonors(mol)`
- **Rotierende Bindungen** - `Descriptors.NumRotatableBonds(mol)`
- **Aromatische Ringe** - Benutzerdefinierte Zählung über `Chem.GetSSSR(mol)`
- **LogP** - `Descriptors.MolLogP(mol)`

### Optimierung

Die Berechnung der Eigenschaften ist für große Datensätze optimiert:

- **Parallelverarbeitung** mit mehreren CPU-Kernen
- **Batch-Verarbeitung** für effiziente Speichernutzung
- **Fortschrittsanzeige** mit tqdm
- **Robuste Fehlerbehandlung** und Logging

## Integration in eigene Skripte

Die Funktionalität kann leicht in eigene Skripte integriert werden:

```python
from PreprocessingPipeline.MolecularProperties import calculate_molecular_properties

# Konfiguration für die Eigenschaftsberechnung
config = {
    "parallel_processing": True,
    "num_processes": 4,  # Anzahl der zu verwendenden CPU-Kerne
    "batch_size": 1000   # Größe der Verarbeitungsbatches
}

# Berechne Eigenschaften für eine JSON-Datei mit Molekülsequenzen
success = calculate_molecular_properties(
    "path/to/sequences.json",  # Eingabedatei
    "path/to/output.json",     # Ausgabedatei (optional, wenn None wird die Eingabedatei überschrieben)
    config                     # Konfiguration (optional)
)
```

## Weiterverwendung der Eigenschaften

Die molekularen Eigenschaften können für verschiedene Zwecke verwendet werden:

1. **Feature Engineering** für Modelle, die sowohl die Sequenz als auch molekulare Eigenschaften nutzen
2. **Filterung von Molekülen** basierend auf ihren Eigenschaften
3. **Validierung von generierten Molekülen** in generativen Modellen
4. **Multi-Task Learning** mit verschiedenen molekularen Eigenschaften als Ziele

## Beispiele

### Filtern von Molekülen nach Eigenschaften

```python
import json

# Lade Moleküldaten
with open("path/to/sequences.json", "r") as f:
    molecules = json.load(f)

# Filtere nach Molekularmasse und LogP
filtered_molecules = {}
for mol_id, mol_data in molecules.items():
    properties = mol_data["properties"]
    
    # Nur Moleküle mit Masse < 500 und LogP < 5 (Lipinski-Regel)
    if properties["molekularmasse"] < 500 and properties["logp"] < 5:
        filtered_molecules[mol_id] = mol_data

# Speichere gefilterte Moleküle
with open("filtered_molecules.json", "w") as f:
    json.dump(filtered_molecules, f, indent=2)
```

### Statistik über Moleküleigenschaften

```python
import json
import numpy as np
import matplotlib.pyplot as plt

# Lade Moleküldaten
with open("path/to/sequences.json", "r") as f:
    molecules = json.load(f)

# Sammle Eigenschaften
masses = []
logp_values = []
hba_counts = []

for mol_id, mol_data in molecules.items():
    properties = mol_data["properties"]
    masses.append(properties["molekularmasse"])
    logp_values.append(properties["logp"])
    hba_counts.append(properties["hba"])

# Einfache Statistiken
print(f"Anzahl Moleküle: {len(molecules)}")
print(f"Durchschnittliche Masse: {np.mean(masses):.2f} g/mol")
print(f"Durchschnittlicher LogP: {np.mean(logp_values):.2f}")
print(f"Durchschnittliche HBA: {np.mean(hba_counts):.2f}")

# Erstelle Histogramm
plt.figure(figsize=(10, 6))
plt.hist(masses, bins=50)
plt.title("Verteilung der Molekularmassen")
plt.xlabel("Masse (g/mol)")
plt.ylabel("Anzahl")
plt.savefig("masse_verteilung.png")
``` 