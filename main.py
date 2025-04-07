from PreprocessingPipeline.splitters.ChemicalDataSplitter import ChemicalDataSplitter
from PreprocessingPipeline.filters.MoleculeElementFilter import MoleculeElementFilter
from PreprocessingPipeline.splitters.M_EntrySorter import M_EntrySorter
from PreprocessingPipeline.molecular.MolToSequence import molToSequenceFunction
from PreprocessingPipeline.splitters.ChemicalDataSplitter import ChemicalDataSplitter
from PreprocessingPipeline.splitters.LengthSorter import MoleculeFileSorter
from PreprocessingPipeline.utils.DirectoryAnalyzer import DirectoryAnalyzer
from PreprocessingPipeline.filters.LengthFilter import LengthFilterer
from PreprocessingPipeline.molecular.MoleculeSampleShuffler import MoleculeSampleShuffler
from PreprocessingPipeline.utils.TextFileCombiner import TextFileCombiner
from PreprocessingPipeline.filters.base_processor import ProcessingConfig
from PreprocessingPipeline.molecular.MoleculeAugmenter import augment_molecules
from PreprocessingPipeline.molecular.MolecularProperties import add_properties_to_mol_sequences
from misc.download_data import download_and_extract_pubchem_compounds

# RDKit-Importe für molekulare Eigenschaften
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

import os
import re
import logging
from pathlib import Path
import json
import glob
import time
import random
import multiprocessing
from tqdm import tqdm
from functools import partial
import psutil

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

'''This Main Script Downloads PubChem Data in MolTable format and transforms it into LatentMol's relativistic
   graph sequences for deep learning with Transformers.
   For this it works through all the steps calling external scripts.'''

ATOM_DIMENSION = 12 # Number of atom-Constants --> depends on the Dictionary 
MAX_BONDS = 4 # max number of bonds: default is 4 --> Octet rule (As each bond contains two electrons [Hydrogen gets handeled specially])
MOL_MIN_LENGTH = 1 # min length of molecules: the "length" refers to the number of atoms notated in the moltable --> many implicit Hydrogens are not counted
MOL_MAX_LENGTH = 40 # max length of molecules: the "length" refers to the number of atoms notated in the moltable --> many implicit Hydrogens are not counted
MAX_PERMUTATIONS = 5 # These are the augmented Versions --> Set it to the value that you want. Pobably something line 1000 would be appropriate. Although that would result in huge amounts of samples.

# Funktion zum Berechnen der molekularen Eigenschaften mit RDKit
def calculate_molecular_properties(mol):
    """
    Berechnet wichtige molekulare Eigenschaften mit RDKit.
    
    Args:
        mol: RDKit-Molekülobjekt
        
    Returns:
        Dictionary mit berechneten Eigenschaften
    """
    if mol is None:
        return {
            "molekularmasse": None,
            "hba": None,
            "hbd": None,
            "rotierende_bindungen": None,
            "aromatische_ringe": None,
            "logp": None
        }
    
    # Zähle aromatische Ringe
    aromatische_ringe = 0
    for ring in Chem.GetSSSR(mol):
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            aromatische_ringe += 1
    
    # Berechne die Eigenschaften
    eigenschaften = {
        "molekularmasse": round(Descriptors.MolWt(mol), 2),
        "hba": Lipinski.NumHAcceptors(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "rotierende_bindungen": Descriptors.NumRotatableBonds(mol),
        "aromatische_ringe": aromatische_ringe,
        "logp": round(Descriptors.MolLogP(mol), 2)
    }
    
    return eigenschaften

# Augmentierungskonfiguration - optimiert für High-End-Hardware
DEFAULT_AUGMENTATION_CONFIG = {
    "num_conformers": 3,             # Erhöht von 3 auf 5
    "use_geometric_perturbation": True,
    "perturbation_magnitude": 0.1,
    "use_rotation_translation": True,
    "num_rotation_samples": 2        # Erhöht von 2 auf 4
}

class Verarbeiter:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        
        # Grundkonfiguration für die Verarbeitung - optimiert für High-End-Hardware
        self.config = ProcessingConfig(
            batch_size=5000,          # Erhöht von 1000 auf 5000
            max_ram_usage=0.9,        # Erhöht von 0.75 auf 0.9 (90% RAM-Nutzung)
            checkpoint_interval=10000, # Erhöht von 5000 auf 10000
            log_level=logging.INFO
        )
        
        # Optimierte Parallelisierung: Maximale Kernnutzung für Ryzen 7 7700
        # Nutzung von 95% der verfügbaren Kerne für beste Performance
        self.num_processes = max(1, int(multiprocessing.cpu_count() * 0.95))
        self.logger.info(f"Verwende {self.num_processes} CPU-Kerne für Parallelverarbeitung")

    def prepare_raw_data(self, db_file_directory: str) -> None:
        """Optimierte Version der Datenvorverarbeitung"""
        base_dir = Path("data")
        
        self._prepare_raw_data_substep_I(db_file_directory, base_dir)
        
        analyzer = DirectoryAnalyzer("data/temp/V_original_data")
        analyzer.scan_directory()
        data = analyzer.get_data()
        
        for i in data.numbers:
            self.logger.info(f"----------------------------------------")
            self.logger.info(f"Starte mit Verarbeitung von Nummer: {i} von {MOL_MAX_LENGTH}")
            self._prepare_raw_data_substep_II(base_dir, i)

    def _prepare_raw_data_substep_I(self, db_file_directory: str, base_dir: Path) -> None:
        """Optimierte Version von Substep I mit RAM-basierter Verarbeitung"""
        
        for filename in os.listdir(db_file_directory):
            if filename.endswith(('.sdf', 'sd', 'txt')):
                file_path = Path(db_file_directory) / filename
                self.logger.info(f"Verarbeite Datei: {filename}")
                
                # 1: Aufteilen der Moleküle
                split_dir = base_dir / 'temp/split_db'
                splitter = ChemicalDataSplitter(
                    input_file=str(file_path),
                    output_directory=str(split_dir),
                    config=self.config
                )
                splitter.split_file()
                
                # 2: Längenfilterung
                filtered_split_dir = base_dir / 'temp/II_split_db_filtered_lengthwise'
                sorter = LengthFilterer(
                    min_length=MOL_MIN_LENGTH,                # Mindestlänge: mindestens 1 Atom
                    max_length=MOL_MAX_LENGTH,   # Maximallänge aus der Konstante
                    count_hydrogens=False,       # Wasserstoffatome nicht mitzählen
                    config=self.config
                )
                sorter.filter_and_copy(split_dir, filtered_split_dir)
                
                # 3: Elementfilterung
                valid_output_dir = base_dir / 'temp/II_valid_files'
                invalid_output_dir = base_dir / 'temp/obsolete/invalid_files'
                allowed_elements = ["H", "C", "O", "N", "F", "P", "S", "Cl", "Br", "I"]
                
                molecule_filter = MoleculeElementFilter(
                    input_directory=str(filtered_split_dir),
                    valid_output_dir=str(valid_output_dir),
                    invalid_output_dir=str(invalid_output_dir),
                    allowed_elements=allowed_elements,
                    config=self.config
                )
                molecule_filter.filter_molecules()
                
                # 4: M-Entry Sortierung
                output_dir_chg = base_dir / 'temp/obsolete/with_CHG'
                output_dir_iso = base_dir / 'temp/obsolete/with_ISO'
                output_dir_end = base_dir / 'temp/obsolete/with_M_END'
                significant_output = base_dir / 'temp/IV_with_Other_M'
                
                organizer = M_EntrySorter(
                    str(valid_output_dir),
                    str(output_dir_chg),
                    str(output_dir_iso),
                    str(output_dir_end),
                    str(significant_output)
                )
                organizer.organize_files()
                
                # 5: Längensortierung
                sorter = MoleculeFileSorter()
                output_dir = base_dir / 'temp/V_original_data'
                sorter.sort_files(str(significant_output), str(output_dir))

    def _prepare_raw_data_substep_II(self, base_dir: Path, number: int) -> None:
        """Optimierte Version von Substep II mit RAM-basierter Verarbeitung"""
        
        input_dir = base_dir / "temp/V_original_data" / str(number)
        
        # Debug: Überprüfe Eingabeverzeichnis
        if not input_dir.exists():
            self.logger.error(f"Eingabeverzeichnis {input_dir} existiert nicht!")
            return
            
        # Debug: Liste Dateien im Verzeichnis
        files = list(input_dir.glob('*.txt'))
        self.logger.info(f"Gefundene Dateien in {input_dir}: {[f.name for f in files]}")
        
        if not files:
            self.logger.error(f"Keine .txt Dateien in {input_dir} gefunden!")
            return
        
        # 1: Kombiniere Dateien
        output_file = f"data/temp_II/original_data/molecules_{number}.txt"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Debug: Überprüfe ob Ausgabeverzeichnis erstellt wurde
        if not os.path.exists(os.path.dirname(output_file)):
            self.logger.error(f"Konnte Ausgabeverzeichnis {os.path.dirname(output_file)} nicht erstellen!")
            return
        
        combiner = TextFileCombiner(str(input_dir), output_file)
        combiner.run()
        
        # Debug: Überprüfe ob Ausgabedatei erstellt wurde
        if not os.path.exists(output_file):
            self.logger.error(f"Ausgabedatei {output_file} wurde nicht erstellt!")
            return
            
        # Debug: Überprüfe Größe der Ausgabedatei
        file_size = os.path.getsize(output_file)
        self.logger.info(f"Größe der Ausgabedatei {output_file}: {file_size} Bytes")
        
        if file_size == 0:
            self.logger.error(f"Ausgabedatei {output_file} ist leer!")
            return
        
    def extract_numbers_from_filenames(self, directory): 
        numbers = []
        for filename in os.listdir(directory):
            # Extract numbers from the filename using regex
            found_numbers = re.findall(r'\d+', filename)
            # Convert the found numbers to integers and add to the list
            numbers.extend(map(int, found_numbers))
        return sorted(numbers)
       
 
    def _make_sequence_data(self):
        """Optimierte Version der Sequenzdatengenerierung"""
        numbers = self.extract_numbers_from_filenames("data/temp_II/original_data")
        output_dir = Path(f"data/latentmol_sequences")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Überprüfe ob das Verzeichnis existiert und beschreibbar ist
        if not output_dir.exists():
            self.logger.error(f"Ausgabeverzeichnis {output_dir} konnte nicht erstellt werden")
            return
        if not os.access(output_dir, os.W_OK):
            self.logger.error(f"Keine Schreibrechte im Verzeichnis {output_dir}")
            return

        for i in numbers:
            self.logger.info("----------------------------------------")
            self.logger.info(f"Starte MolToSequence Konvertierung für Moleküle der Länge: {i} von {MOL_MAX_LENGTH}")

            input_file = Path(f"data/temp_II/original_data/molecules_{i}.txt")
            
            # Überprüfe ob die Eingabedatei existiert
            if not input_file.exists():
                self.logger.warning(f"Eingabedatei {input_file} existiert nicht, überspringe...")
                continue
                
            # Standard-Konvertierung
            success = molToSequenceFunction(
                str(input_file),  # Direkt den Dateipfad übergeben
                str(output_dir),  # Direkt das Ausgabeverzeichnis übergeben
                i,                # Die Nummer direkt übergeben
                None             # Keine spezielle Konfiguration
            )
            
            if not success:
                self.logger.error(f"Fehler bei der Konvertierung von {input_file}")
                continue

            # Überprüfe ob die Ausgabedatei erstellt wurde
            output_file = output_dir / f"Länge_{i}.json"
            if not output_file.exists():
                self.logger.error(f"Ausgabedatei {output_file} wurde nicht erstellt")
            else:
                # Berechne molekulare Eigenschaften für die erzeugten Sequenzen
                self.logger.info(f"Berechne molekulare Eigenschaften für Sequenzen der Länge {i}...")
                
                # Bestimme das Verzeichnis mit den originalen MOL-Dateien
                mol_dir = Path(f"data/temp/V_original_data/{i}")
                
                # Überprüfe, ob das MOL-Verzeichnis existiert
                if not mol_dir.exists() or not mol_dir.is_dir():
                    self.logger.warning(f"MOL-Verzeichnis {mol_dir} existiert nicht, verwende generische Eigenschaften")
                    mol_dir = Path("data/temp/V_original_data")  # Fallback-Verzeichnis
                
                # Optimierte parallele Berechnung der Eigenschaften
                properties_config = {
                    "parallel_processing": True,
                    "num_processes": self.num_processes,
                    "batch_size": 1000
                }
                
                # Berechne und füge die Eigenschaften hinzu
                prop_success = add_properties_to_mol_sequences(
                    str(output_file),  # JSON-Datei mit Sequenzen
                    str(mol_dir),      # Verzeichnis mit MOL-Dateien
                    None,              # Überschreibe die Originaldatei
                    properties_config  # Konfiguration
                )
                
                if prop_success:
                    self.logger.info(f"Molekulare Eigenschaften erfolgreich zu {output_file} hinzugefügt")
                else:
                    self.logger.error(f"Fehler beim Hinzufügen von Eigenschaften zu {output_file}")

                # Überprüfe ob die Datei Daten enthält
                try:
                    with open(output_file, 'r') as f:
                        data = json.load(f)
                        if not data:
                            self.logger.warning(f"Ausgabedatei {output_file} ist leer")
                        else:
                            self.logger.info(f"Erfolgreich {len(data)} Moleküle in {output_file} gespeichert")
                except Exception as e:
                    self.logger.error(f"Fehler beim Lesen von {output_file}: {e}")
        
        self.logger.info("Sequenzdatengenerierung abgeschlossen")

    def _process_single_molecule(self, file_path, output_dir, augmentation_config, max_variants, random_selection):
        """
        Verarbeitet ein einzelnes Molekül und erzeugt nur die ausgewählten Varianten.
        
        Args:
            file_path: Pfad zur Moleküldatei
            output_dir: Ausgabeverzeichnis
            augmentation_config: Konfiguration für die Augmentierung
            max_variants: Maximale Anzahl Varianten
            random_selection: Ob zufällig ausgewählt werden soll
            
        Returns:
            Liste der erzeugten Dateien
        """
        try:
            # 1. Bestimme die Gesamtzahl möglicher Varianten
            num_conformers = augmentation_config.get("num_conformers", 3)
            use_perturbation = augmentation_config.get("use_geometric_perturbation", True)
            use_rotation = augmentation_config.get("use_rotation_translation", True)
            num_rotations = augmentation_config.get("num_rotation_samples", 2)
            
            # Berechne die Gesamtzahl möglicher Varianten
            total_variants = num_conformers
            if use_perturbation:
                total_variants *= 1  # Jedes Konformer wird einmal gestört
            if use_rotation:
                total_variants *= num_rotations  # Jedes gestörte Konformer wird rotiert
            
            # 2. Wähle die Varianten aus, die erzeugt werden sollen
            basename = os.path.splitext(os.path.basename(file_path))[0]
            variant_indices = list(range(total_variants))
            
            if max_variants < total_variants:
                if random_selection:
                    selected_indices = random.sample(variant_indices, max_variants)
                else:
                    selected_indices = variant_indices[:max_variants]
            else:
                selected_indices = variant_indices
            
            # 3. Erzeugen Sie nur die ausgewählten Varianten
            # Erstelle eine angepasste Konfiguration für die Auswahl
            selected_files = []
            
            # Rufe augment_molecules mit den ausgewählten Varianten auf
            # Wir übergeben die ausgewählten Indizes als zusätzlichen Parameter
            custom_config = augmentation_config.copy()
            custom_config["selected_variant_indices"] = selected_indices
            
            augmented_files = augment_molecules(file_path, output_dir, custom_config)
            selected_files.extend(augmented_files)
            
            return selected_files
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Verarbeitung von {file_path}: {str(e)}")
            return []

    def augment_molecules(self, input_path, output_dir=None, augmentation_config=None, 
                         max_variants_per_molecule=2, random_selection=True, 
                         batch_size=500, num_processes=None):
        """
        Augmentiert Moleküle aus einer Datei oder einem Verzeichnis - optimiert für Millionen von Molekülen.
        Zuerst werden die zu erzeugenden Varianten ausgewählt, dann nur diese erstellt.
        Parallelisierung und Batch-Verarbeitung für optimale Ressourcennutzung.
        
        Args:
            input_path: Pfad zur Eingabedatei oder zum Eingabeverzeichnis
            output_dir: Pfad zum Ausgabeverzeichnis (optional)
            augmentation_config: Konfiguration für die Augmentierung (optional)
            max_variants_per_molecule: Maximale Anzahl an Varianten pro Molekül
            random_selection: Ob die Varianten zufällig ausgewählt werden sollen
            batch_size: Anzahl der Moleküle, die gleichzeitig verarbeitet werden
            num_processes: Anzahl der parallelen Prozesse (None = automatisch)
        """
        start_time = time.time()
        
        # Standardwerte, wenn nicht angegeben
        if output_dir is None:
            output_dir = "augmented_output"
        
        if augmentation_config is None:
            augmentation_config = DEFAULT_AUGMENTATION_CONFIG
            
        if num_processes is None:
            num_processes = self.num_processes
            
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"Starte optimierte Augmentierung von: {input_path}")
        self.logger.info(f"Ausgabe in: {output_dir}")
        self.logger.info(f"Konfiguration: {augmentation_config}")
        self.logger.info(f"Max. Varianten pro Molekül: {max_variants_per_molecule}")
        self.logger.info(f"Zufällige Auswahl: {'Ja' if random_selection else 'Nein'}")
        self.logger.info(f"Batch-Größe: {batch_size}")
        self.logger.info(f"Prozesse: {num_processes}")
        
        total_files = 0
        all_mol_files = []
        
        # Sammle alle zu verarbeitenden Dateien
        if os.path.isfile(input_path):
            if input_path.endswith(('.mol', '.sdf')):
                all_mol_files.append(input_path)
        elif os.path.isdir(input_path):
            for root, _, files in os.walk(input_path):
                for file in files:
                    if file.endswith(('.mol', '.sdf')):
                        all_mol_files.append(os.path.join(root, file))
        else:
            self.logger.error(f"Eingabepfad {input_path} existiert nicht!")
            return []
            
        # Keine Dateien gefunden
        if not all_mol_files:
            self.logger.warning(f"Keine .mol oder .sdf Dateien in {input_path} gefunden!")
            return []
            
        self.logger.info(f"Gefunden: {len(all_mol_files)} Moleküldateien")
        
        # Definiere die Verarbeitungsfunktion mit den übrigen Parametern
        process_func = partial(
            self._process_single_molecule, 
            output_dir=output_dir, 
            augmentation_config=augmentation_config,
            max_variants=max_variants_per_molecule,
            random_selection=random_selection
        )
        
        # Verarbeite die Dateien in Batches
        all_selected_files = []
        
        # Erstelle einen Multiprocessing-Pool
        pool = multiprocessing.Pool(processes=num_processes)
        
        # Verarbeite in Batches, um den RAM zu schonen
        for i in range(0, len(all_mol_files), batch_size):
            batch = all_mol_files[i:i+batch_size]
            
            # Zeige den Fortschritt an
            self.logger.info(f"Verarbeite Batch {i//batch_size + 1}/{(len(all_mol_files) + batch_size - 1)//batch_size}: {len(batch)} Moleküle")
            
            try:
                # Verarbeite den Batch parallel
                with tqdm(total=len(batch), desc="Moleküle", unit="mol") as pbar:
                    # Verwende imap, um einen Iterator zu erhalten und den Fortschritt zu verfolgen
                    for result in pool.imap_unordered(process_func, batch):
                        all_selected_files.extend(result)
                        pbar.update(1)
                        
                # Zeige aktuelle Speichernutzung an
                ram_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                self.logger.info(f"RAM-Verbrauch: {ram_usage:.2f} MB")
                
                # Gib RAM frei, indem wir den Garbage Collector erzwingen
                import gc
                gc.collect()
                
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                self.logger.warning("Verarbeitung durch Benutzer abgebrochen")
                break
                
        # Schließe den Pool
        pool.close()
        pool.join()
        
        # Zeige Zusammenfassung
        end_time = time.time()
        total_time = end_time - start_time
        self.logger.info(f"Augmentierung abgeschlossen in {total_time:.2f} Sekunden")
        self.logger.info(f"Erstellt: {len(all_selected_files)} augmentierte Moleküle")
        
        return all_selected_files

def create_directory_structure():
    # List of directories to create
    directories = [
        'data',
        'data/temp',
        'data/temp/split_db',
        'data/temp/II_split_db_filtered_lengthwise',
        'data/temp/II_valid_files',
        'data/temp/obsolete',
        'data/temp/obsolete/invalid_files',
        'data/temp/obsolete/with_CHG',
        'data/temp/obsolete/with_ISO',
        'data/temp/obsolete/with_M_END',
        'data/temp/IV_with_Other_M',
        'data/temp/V_original_data',
        'data/temp_II',
        'data/temp_II/original_data',
        'data/latentmol_sequences',
        'augmented_output'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Beispielnutzung direkt ohne Kommandozeilenparameter
if __name__ == "__main__":
    # Verzeichnisstruktur erstellen
    create_directory_structure()
    
    # Verarbeiter initialisieren
    processor = Verarbeiter()
    
    # OPTION 1: Molekülaugmentierung (direkt ausführbar)
    # Die Standardwerte können direkt hier angepasst werden!
    
    # Beispiel 1: Einzelne Moleküldatei augmentieren
    # Ersetzen Sie 'beispiel.mol' durch den Pfad zu Ihrer .mol oder .sdf Datei
    beispiel_datei = "beispiel.mol"
    if os.path.exists(beispiel_datei):
        processor.augment_molecules(
            input_path=beispiel_datei,
            output_dir="augmented_output",
            max_variants_per_molecule=2,         # Erhöht von 2 auf 5 Varianten
            random_selection=True,               # Zufällige Auswahl
            batch_size=500,                      # Erhöht von 100 auf 500
            num_processes=None                   # Automatische Kernnutzung (nutzt jetzt 95%)
        )
    
    # Beispiel 2: Verzeichnis mit Moleküldateien augmentieren
    # Ersetzen Sie 'molekuel_verzeichnis' durch Ihr Verzeichnis mit .mol oder .sdf Dateien
    beispiel_verzeichnis = "molekuel_verzeichnis"
    if os.path.exists(beispiel_verzeichnis):
        processor.augment_molecules(
            input_path=beispiel_verzeichnis,
            output_dir="augmented_output",
            max_variants_per_molecule=3,         # Erhöht von 3 auf 8 Varianten
            random_selection=True,               # Zufällige Auswahl
            batch_size=1000,                     # Erhöht von 200 auf 1000
            num_processes=None                   # Automatische Kernnutzung (nutzt jetzt 95%)
        )
    
    # Beispiel 3: Augmentierung mit angepasster Konfiguration
    angepasste_config = {
        "num_conformers": 5,                 # Erhöht von 5 auf 8
        "use_geometric_perturbation": True,
        "perturbation_magnitude": 0.2,       # Stärkere Störungen
        "use_rotation_translation": True,
        "num_rotation_samples": 3            # Erhöht von 3 auf 5
    }
    
    # Kommentieren Sie die nächsten 8 Zeilen ein und passen Sie den Pfad an
    # processor.augment_molecules(
    #     input_path="ihre_molekuel_datei.mol",
    #     output_dir="augmented_output",
    #     augmentation_config=angepasste_config,
    #     max_variants_per_molecule=4,          # Nur 4 Varianten behalten
    #     random_selection=True,                # Zufällige Auswahl
    #     batch_size=50,                        # Kleinere Batches für komplexere Konfiguration
    #     num_processes=2                       # 2 CPU-Kerne verwenden
    # )
    
    # OPTION 2: Datenvorverarbeitung 
    # Standardpipeline für PubChem-Daten
    # Kommentieren Sie die folgenden Zeilen ein, um die Datenvorverarbeitung zu starten
    
    # PubChem-Daten herunterladen (falls noch nicht vorhanden)
    download_dir = "src"
    os.makedirs(download_dir, exist_ok=True)
    download_and_extract_pubchem_compounds(download_dir)
    #
    # # Rohdaten vorbereiten
    processor.prepare_raw_data(download_dir)
    #
    # # Sequenzdaten erstellen
    processor._make_sequence_data()
    
    # Informationsmeldung für IDE-Nutzer
    print("\nDie Verarbeitung ist abgeschlossen oder wurde nicht aktiviert.")
    print("Um bestimmte Funktionen zu nutzen, bearbeiten Sie bitte die main.py und entfernen Sie die Kommentarzeichen.")
    print("Beispiel: Entfernen Sie die '#' vor den gewünschten Funktionen.")
    print("\nVerfügbare Funktionen:")
    print("1. Molekülaugmentierung (bereits aktiviert mit Beispielen)")
    print("2. PubChem-Datenvorverarbeitung (auskommentiert)")


# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.
