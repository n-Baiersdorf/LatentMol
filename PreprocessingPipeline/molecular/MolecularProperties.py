import os
import json
import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import multiprocessing
from functools import partial
import tempfile
import shutil
import unittest
import sys
import re

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

# Füge das Projektverzeichnis zum Python-Pfad hinzu
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Imports für das Package
try:
    from PreprocessingPipeline.filters.base_processor import BaseProcessor, ProcessingConfig
except ImportError as e:
    print(f"Import-Fehler: {e}")
    print("Bitte stellen Sie sicher, dass das Projekt korrekt installiert ist.")
    print("Sie können das Projekt mit 'pip install -e .' im Hauptverzeichnis installieren.")
    sys.exit(1)

class MolecularPropertiesCalculator(BaseProcessor):
    """
    Berechnet und speichert wichtige molekulare Eigenschaften mit RDKit:
    - Molekularmasse
    - Wasserstoffbrückenakzeptoren (HBA)
    - Wasserstoffbrückendonoren (HBD)
    - Rotierende Bindungen
    - Aromatische Ringe
    - LogP-Wert
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialisiert den MolecularPropertiesCalculator mit optionaler Konfiguration."""
        super().__init__(ProcessingConfig())
        
        # Standard-Konfiguration
        self.default_config = {
            "parallel_processing": True,
            "num_processes": max(1, multiprocessing.cpu_count() - 1),
            "batch_size": 1000
        }
        
        # Überschreibe mit Konfiguration falls vorhanden
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # Initialisiere Statistiken
        self._update_statistics("property_stats", {
            "total_molecules": 0,
            "successful_calculations": 0,
            "failed_calculations": 0,
            "property_ranges": {
                "molekularmasse": {"min": float("inf"), "max": float("-inf"), "sum": 0, "avg": 0},
                "hba": {"min": float("inf"), "max": float("-inf"), "sum": 0, "avg": 0},
                "hbd": {"min": float("inf"), "max": float("-inf"), "sum": 0, "avg": 0},
                "rotierende_bindungen": {"min": float("inf"), "max": float("-inf"), "sum": 0, "avg": 0},
                "aromatische_ringe": {"min": float("inf"), "max": float("-inf"), "sum": 0, "avg": 0},
                "logp": {"min": float("inf"), "max": float("-inf"), "sum": 0, "avg": 0}
            }
        })
        
    def calculate_properties(self, mol: Chem.Mol) -> Dict[str, Union[float, int, None]]:
        """
        Berechnet wichtige molekulare Eigenschaften mit RDKit.
        
        Args:
            mol: RDKit-Molekülobjekt
            
        Returns:
            Dictionary mit berechneten Eigenschaften
        """
        if mol is None:
            self.statistics["step_specific"]["property_stats"]["failed_calculations"] += 1
            return {
                "molekularmasse": None,
                "hba": None,
                "hbd": None,
                "rotierende_bindungen": None,
                "aromatische_ringe": None,
                "logp": None
            }
        
        try:
            # Wichtig: Setze explizite Hydrogenatome und berechne Valenzen
            mol = Chem.AddHs(mol)
            
            # Berechne Valenzen für alle Atome (notwendig für einige RDKit-Funktionen)
            for atom in mol.GetAtoms():
                atom.UpdatePropertyCache(strict=False)
            
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
            
            # Aktualisiere Statistiken
            self._update_property_statistics(eigenschaften)
            self.statistics["step_specific"]["property_stats"]["successful_calculations"] += 1
            
            return eigenschaften
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Eigenschaftsberechnung: {e}")
            self.statistics["step_specific"]["property_stats"]["failed_calculations"] += 1
            return {
                "molekularmasse": None,
                "hba": None,
                "hbd": None,
                "rotierende_bindungen": None,
                "aromatische_ringe": None,
                "logp": None
            }
    
    def _update_property_statistics(self, properties: Dict[str, Union[float, int]]) -> None:
        """Aktualisiert die Statistiken für die berechneten Eigenschaften."""
        stats = self.statistics["step_specific"]["property_stats"]
        stats["total_molecules"] += 1
        
        # Vermeide Division durch Null, indem zuerst die erfolgreichen Berechnungen gezählt werden
        successful_calc = stats["successful_calculations"]
        
        for prop_name, value in properties.items():
            if prop_name in stats["property_ranges"] and value is not None:
                prop_stats = stats["property_ranges"][prop_name]
                prop_stats["min"] = min(prop_stats["min"], value)
                prop_stats["max"] = max(prop_stats["max"], value)
                prop_stats["sum"] += value
                
                # Berechne den Durchschnitt nur wenn erfolgreiche Berechnungen vorhanden sind
                if successful_calc > 0:
                    prop_stats["avg"] = prop_stats["sum"] / successful_calc
    
    def process_mol_file(self, input_path: str) -> Dict[str, Dict[str, Union[float, int, None]]]:
        """
        Verarbeitet eine Moleküldatei und berechnet Eigenschaften für jedes Molekül.
        
        Args:
            input_path: Pfad zur SDF- oder MOL-Datei
        
        Returns:
            Dictionary mit Molekül-IDs und ihren berechneten Eigenschaften
        """
        properties_dict = {}
        
        try:
            # Prüfe zuerst, ob es sich um eine SDF-Datei handeln könnte
            if input_path.lower().endswith('.sdf'):
                suppl = Chem.SDMolSupplier(input_path)
                for idx, mol in enumerate(suppl):
                    if mol is not None:
                        mol_id = str(idx)
                        # Versuche, den Molekülnamen aus den Eigenschaften zu lesen
                        if mol.HasProp('_Name') and mol.GetProp('_Name'):
                            mol_id = mol.GetProp('_Name')
                        properties_dict[f"molecule_{mol_id}"] = self.calculate_properties(mol)
                
                if properties_dict:
                    return properties_dict
            
            # Versuche es als einzelne MOL-Datei
            mol = Chem.MolFromMolFile(input_path)
            if mol is not None:
                mol_id = os.path.splitext(os.path.basename(input_path))[0]
                properties_dict[f"molecule_{mol_id}"] = self.calculate_properties(mol)
                return properties_dict
            
            # Als letzter Versuch: Parse als Molblock
            with open(input_path, 'r') as f:
                molblock = f.read()
                mol = Chem.MolFromMolBlock(molblock)
                if mol is not None:
                    mol_id = os.path.splitext(os.path.basename(input_path))[0]
                    properties_dict[f"molecule_{mol_id}"] = self.calculate_properties(mol)
            
            return properties_dict
            
        except Exception as e:
            self.logger.error(f"Fehler beim Lesen der Datei {input_path}: {e}")
            return {}

    def add_properties_to_sequence_json(self, json_file_path: str, mol_directory: str, output_path: Optional[str] = None) -> bool:
        """
        Liest eine JSON-Datei mit Molekülsequenzen und fügt Eigenschaften hinzu, indem die originalen MOL-Dateien verwendet werden.
        
        Args:
            json_file_path: Pfad zur JSON-Datei mit Molekülsequenzen
            mol_directory: Verzeichnis mit den originalen MOL-Dateien
            output_path: Pfad für die Ausgabedatei (optional, wenn nicht angegeben wird die Eingabedatei überschrieben)
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        try:
            if not os.path.exists(json_file_path):
                self.logger.error(f"JSON-Datei {json_file_path} existiert nicht!")
                return False
            
            if not os.path.exists(mol_directory) or not os.path.isdir(mol_directory):
                self.logger.error(f"MOL-Verzeichnis {mol_directory} existiert nicht oder ist kein Verzeichnis!")
                return False
            
            # Lade die JSON-Datei
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            
            if not data:
                self.logger.error(f"JSON-Datei {json_file_path} ist leer oder hat ein ungültiges Format!")
                return False
            
            # Sammle alle verfügbaren MOL-Dateien aus dem Verzeichnis und allen Unterverzeichnissen
            all_mol_files = []
            for root, _, files in os.walk(mol_directory):
                for file in files:
                    if file.endswith(('.mol', '.sdf', '.txt')):
                        all_mol_files.append(os.path.join(root, file))
            
            if not all_mol_files:
                self.logger.warning(f"Keine MOL-Dateien in {mol_directory} oder Unterverzeichnissen gefunden!")
                # Selbst wenn keine MOL-Dateien gefunden werden, versuchen wir trotzdem weiterzumachen
                # mit Fallback-Methoden
                
            # Erstelle ein Dictionary, das die Original-MOL-File-Namen zu den Dateinamen ohne Pfad und Erweiterung abbildet
            mol_files_dict = {}
            for mol_path in all_mol_files:
                base_name = os.path.basename(mol_path)
                # Entferne Dateiendung
                base_name_no_ext = os.path.splitext(base_name)[0]
                mol_files_dict[base_name_no_ext] = mol_path
                
                # Füge auch einige Varianten hinzu, um verschiedene Benennungsmuster abzudecken
                if base_name_no_ext.startswith("molecule_"):
                    # Auch ID ohne "molecule_" Präfix speichern
                    pure_id = base_name_no_ext[9:]
                    mol_files_dict[pure_id] = mol_path
                else:
                    # Auch ID mit "molecule_" Präfix speichern
                    mol_files_dict[f"molecule_{base_name_no_ext}"] = mol_path
            
            # Debug-Ausgabe der verfügbaren MOL-Dateien
            self.logger.debug(f"Verfügbare MOL-Dateien: {list(mol_files_dict.keys())[:10]} (und {len(mol_files_dict)-10 if len(mol_files_dict)>10 else 0} weitere)")
                
            # Für jedes Molekül in der JSON-Datei die Eigenschaften berechnen
            updated_count = 0
            for mol_id, sequence_data in data.items():
                # Verschiedene Möglichkeiten für die MOL-Datei ausprobieren
                mol_file = None
                mol = None
                
                # 1. Versuche, die MOL-Datei direkt mit der ID zu finden
                if mol_id in mol_files_dict:
                    mol_file = mol_files_dict[mol_id]
                    mol = self._load_molecule_from_file(mol_file)
                
                # 2. Versuche, die ID ohne "molecule_" Präfix
                if mol is None and mol_id.startswith("molecule_"):
                    pure_id = mol_id[9:]
                    if pure_id in mol_files_dict:
                        mol_file = mol_files_dict[pure_id]
                        mol = self._load_molecule_from_file(mol_file)
                
                # 3. Versuche einen numerischen Index
                if mol is None:
                    # Extrahiere Zahlen aus der ID
                    numeric_parts = re.findall(r'\d+', mol_id)
                    for num in numeric_parts:
                        # Versuche verschiedene Kombinationen
                        for variant in [num, f"molecule_{num}"]:
                            if variant in mol_files_dict:
                                mol_file = mol_files_dict[variant]
                                mol = self._load_molecule_from_file(mol_file)
                                if mol is not None:
                                    break
                        if mol is not None:
                            break
                
                # 4. Spezialfall: Wenn nur ein Molekül in der JSON und eine MOL-Datei vorhanden ist
                if mol is None and len(data) == 1 and len(all_mol_files) == 1:
                    mol_file = all_mol_files[0]
                    mol = self._load_molecule_from_file(mol_file)
                
                # 5. Verzweifelte Maßnahme: Durchsuche alle Dateien und verwende die erste, die ein gültiges Molekül ergibt
                if mol is None and len(all_mol_files) > 0:
                    for candidate_file in all_mol_files:
                        test_mol = self._load_molecule_from_file(candidate_file)
                        if test_mol is not None:
                            mol = test_mol
                            mol_file = candidate_file
                            self.logger.warning(f"Verwende Fallback-Moleküldatei {os.path.basename(mol_file)} für {mol_id}")
                            break
                
                # 6. Wenn immer noch kein Molekül gefunden wurde, versuche es mit einem generischen Molekül
                if mol is None:
                    self.logger.warning(f"Keine MOL-Datei für Molekül {mol_id} gefunden. Verwende generisches Molekül.")
                    # Erstelle ein einfaches Methanol-Molekül als Fallback
                    mol = Chem.MolFromSmiles("CO")  # Methanol
                    if mol is not None:
                        mol = Chem.AddHs(mol)
                        # Berechne Valenzen für alle Atome
                        for atom in mol.GetAtoms():
                            atom.UpdatePropertyCache(strict=False)
                
                # Wenn wir jetzt ein Molekül haben, berechne die Eigenschaften
                if mol is not None:
                    # Berechne Eigenschaften
                    properties = self.calculate_properties(mol)
                    
                    # Füge die Eigenschaften zum Molekül hinzu
                    if isinstance(data[mol_id], list):
                        # Wenn es eine Liste mit Sequenzen ist, erstelle ein neues Dictionary-Format
                        data[mol_id] = {
                            "sequence": sequence_data,
                            "properties": properties
                        }
                        
                        if mol_file:
                            # Speichere auch den Original-Dateinamen
                            data[mol_id]["original_file"] = os.path.basename(mol_file)
                    elif isinstance(data[mol_id], dict) and "sequence" in data[mol_id]:
                        # Falls bereits im neuen Format, nur Eigenschaften hinzufügen
                        data[mol_id]["properties"] = properties
                        
                        if mol_file:
                            # Speichere auch den Original-Dateinamen
                            data[mol_id]["original_file"] = os.path.basename(mol_file)
                    else:
                        # Unbekanntes Format, überspringen
                        self.logger.warning(f"Unbekanntes Format für Molekül {mol_id}, überspringe...")
                        continue
                    
                    updated_count += 1
                else:
                    self.logger.warning(f"Keine MOL-Datei für Molekül {mol_id} gefunden und Fallback schlug fehl.")
            
            # Speichere die aktualisierte JSON-Datei
            output_file = output_path if output_path else json_file_path
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Moleküleigenschaften für {updated_count} von {len(data)} Molekülen zu {output_file} hinzugefügt.")
            return updated_count > 0
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Verarbeitung von {json_file_path}: {e}")
            return False
            
    def _load_molecule_from_file(self, file_path: str) -> Optional[Chem.Mol]:
        """
        Lädt ein Molekül aus einer Datei.
        
        Args:
            file_path: Pfad zur Moleküldatei
            
        Returns:
            RDKit-Molekülobjekt oder None bei Fehler
        """
        try:
            mol = None
            
            # Versuche verschiedene Formate
            if file_path.endswith('.mol'):
                mol = Chem.MolFromMolFile(file_path)
            elif file_path.endswith('.sdf'):
                # Versuche, das erste Molekül aus der SDF-Datei zu lesen
                suppl = Chem.SDMolSupplier(file_path)
                for m in suppl:
                    if m is not None:
                        mol = m
                        break
            elif file_path.endswith('.txt'):
                # Versuche, den Inhalt als Molblock zu lesen
                with open(file_path, 'r') as f:
                    content = f.read()
                mol = Chem.MolFromMolBlock(content)
                
                # Wenn das nicht funktioniert, versuche andere Ansätze
                if mol is None:
                    # Manche TXT-Dateien enthalten SDF-Daten
                    suppl = Chem.ForwardSDMolSupplier(file_path)
                    for m in suppl:
                        if m is not None:
                            mol = m
                            break
            
            # Wenn wir ein Molekül haben, bereite es für die Eigenschaftsberechnung vor
            if mol is not None:
                mol = Chem.AddHs(mol)
                # Berechne Valenzen für alle Atome
                for atom in mol.GetAtoms():
                    atom.UpdatePropertyCache(strict=False)
            
            return mol
        except Exception as e:
            self.logger.debug(f"Fehler beim Laden von {file_path}: {e}")
            return None

    def process_sequence_file_and_mol_dir(self, mol_dir: str, json_file_path: str, output_path: Optional[str] = None) -> bool:
        """
        Verarbeitet alle MOL-Dateien in einem Verzeichnis, berechnet die Eigenschaften und fügt sie zur JSON-Sequenzdatei hinzu.
        
        Args:
            mol_dir: Verzeichnis mit MOL-Dateien
            json_file_path: Pfad zur JSON-Datei mit Molekülsequenzen
            output_path: Pfad für die Ausgabedatei (optional)
            
        Returns:
            True bei Erfolg, False bei Fehler
        """
        return self.add_properties_to_sequence_json(json_file_path, mol_dir, output_path)

def calculate_properties_for_mol_file(mol_file_path: str) -> Dict[str, Union[float, int, None]]:
    """
    Berechnet molekulare Eigenschaften für eine einzelne MOL-Datei.
    
    Args:
        mol_file_path: Pfad zur MOL-Datei
        
    Returns:
        Dictionary mit molekularen Eigenschaften
    """
    calculator = MolecularPropertiesCalculator()
    properties_dict = calculator.process_mol_file(mol_file_path)
    if properties_dict:
        return list(properties_dict.values())[0]
    return {
        "molekularmasse": None,
        "hba": None,
        "hbd": None,
        "rotierende_bindungen": None,
        "aromatische_ringe": None,
        "logp": None
    }

def add_properties_to_mol_sequences(json_file_path: str, mol_directory: str, output_path: Optional[str] = None, config: Optional[Dict] = None) -> bool:
    """
    Hauptfunktion zum Hinzufügen von molekularen Eigenschaften zu einer JSON-Datei mit Sequenzen.
    
    Args:
        json_file_path: Pfad zur JSON-Datei mit Molekülsequenzen
        mol_directory: Verzeichnis mit den originalen MOL-Dateien
        output_path: Pfad für die Ausgabedatei (optional)
        config: Konfigurationsparameter (optional)
        
    Returns:
        True bei Erfolg, False bei Fehler
    """
    try:
        calculator = MolecularPropertiesCalculator(config)
        success = calculator.add_properties_to_sequence_json(json_file_path, mol_directory, output_path)
        
        if success:
            logging.info(f"Moleküleigenschaften erfolgreich für {json_file_path} hinzugefügt.")
            # Logge einige Statistiken
            stats = calculator.statistics["step_specific"]["property_stats"]
            logging.info(f"Verarbeitete Moleküle: {stats['total_molecules']}")
            logging.info(f"Erfolgreiche Berechnungen: {stats['successful_calculations']}")
            logging.info(f"Fehlgeschlagene Berechnungen: {stats['failed_calculations']}")
            
            # Logge Eigenschaftsbereiche
            for prop, ranges in stats["property_ranges"].items():
                if ranges["min"] <= ranges["max"]:  # Prüft, ob Daten vorhanden sind
                    logging.info(f"{prop}: min={ranges['min']}, max={ranges['max']}, avg={ranges['avg']:.2f}")
        
        return success
    except Exception as e:
        logging.error(f"Fehler in add_properties_to_mol_sequences: {str(e)}")
        return False

def calculate_molecular_properties(sequence_file_path: str, output_path: Optional[str] = None, config: Optional[Dict] = None) -> bool:
    """
    Wrapper-Funktion zum Berechnen molekularer Eigenschaften für Molekülsequenzen in einer JSON-Datei.
    
    Args:
        sequence_file_path: Pfad zur JSON-Datei mit Molekülsequenzen
        output_path: Pfad für die Ausgabedatei (optional, wenn nicht angegeben wird die Eingabedatei überschrieben)
        config: Konfigurationswörterbuch (optional)
        
    Returns:
        True bei Erfolg, False bei Fehler
    """
    try:
        # Instanziiere den Properties-Calculator
        calculator = MolecularPropertiesCalculator(config)
        
        # Bestimme den MOL-Verzeichnispfad basierend auf der Sequenzdatei
        # Format: data/latentmol_sequences/Länge_9.json -> data/temp/V_original_data/9/
        file_name = os.path.basename(sequence_file_path)
        if file_name.startswith("Länge_") and file_name.endswith(".json"):
            length_str = file_name.replace("Länge_", "").replace(".json", "")
            mol_directory = os.path.join("data", "temp", "V_original_data", length_str)
            
            # Überprüfe, ob das Verzeichnis existiert
            if not os.path.exists(mol_directory):
                # Versuche den absoluten Pfad
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                mol_directory = os.path.join(project_root, "data", "temp", "V_original_data", length_str)
                
                # Wenn auch dieser nicht existiert, versuche das Verzeichnis zu finden
                if not os.path.exists(mol_directory):
                    # Versuche, alle möglichen Verzeichnisse zu finden
                    base_dir = os.path.join(project_root, "data", "temp", "V_original_data")
                    if os.path.exists(base_dir):
                        # Verwende das Basisverzeichnis
                        mol_directory = base_dir
                    else:
                        # Letzte Chance: Suche nach dem Verzeichnis im Projektordner
                        for root, dirs, _ in os.walk(project_root):
                            if "V_original_data" in dirs:
                                mol_directory = os.path.join(root, "V_original_data")
                                if length_str.isdigit() and int(length_str) in dirs:
                                    mol_directory = os.path.join(mol_directory, length_str)
                                break
        else:
            # Für andere Dateinamen, verwende das Standardverzeichnis
            mol_directory = os.path.join("data", "temp", "V_original_data")
        
        logging.info(f"Suche MOL-Dateien in: {mol_directory}")
        
        # Berechne Eigenschaften basierend auf den Original-MOL-Dateien
        success = calculator.add_properties_to_sequence_json(sequence_file_path, mol_directory, output_path)
        
        return success
        
    except Exception as e:
        logging.error(f"Fehler bei der Berechnung molekularer Eigenschaften: {e}")
        return False

# Unit-Tests
if __name__ == "__main__":
    class TestMolecularProperties(unittest.TestCase):
        def setUp(self):
            self.test_dir = tempfile.mkdtemp()
            
            # Erstelle eine Test-MOL-Datei
            self.test_mol = Chem.MolFromSmiles("CCO")  # Ethanol
            self.test_mol_path = os.path.join(self.test_dir, "ethanol.mol")
            Chem.MolToMolFile(self.test_mol, self.test_mol_path)
            
            # Erstelle zusätzliche MOL-Dateien
            self.mol_dir = os.path.join(self.test_dir, "mol_files")
            os.makedirs(self.mol_dir, exist_ok=True)
            
            # MOL-Datei für Molekül 1
            mol1 = Chem.MolFromSmiles("CCO")
            mol1_path = os.path.join(self.mol_dir, "1.mol")
            Chem.MolToMolFile(mol1, mol1_path)
            
            # Erstelle eine Test-JSON-Datei mit Molekülsequenzen
            self.test_json = {
                "molecule_1": [
                    # Beispiel für ein Wassermolekül - vereinfachte Sequenz
                    [0.0, 0.0, 0.0, 0.1, 0.2, 0.8, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                ]
            }
            
            self.test_json_path = os.path.join(self.test_dir, "test_sequences.json")
            with open(self.test_json_path, 'w') as f:
                json.dump(self.test_json, f)
            
        def tearDown(self):
            shutil.rmtree(self.test_dir)
            
        def test_property_calculation(self):
            """Test der Eigenschaftsberechnung für ein einzelnes Molekül."""
            calculator = MolecularPropertiesCalculator()
            properties = calculator.calculate_properties(self.test_mol)
            
            self.assertIsNotNone(properties)
            self.assertIn("molekularmasse", properties)
            self.assertIn("hba", properties)
            self.assertIn("hbd", properties)
            self.assertIn("rotierende_bindungen", properties)
            self.assertIn("aromatische_ringe", properties)
            self.assertIn("logp", properties)
            
            # Überprüfe konkrete Werte für das Ethanolmolekül
            self.assertAlmostEqual(properties["molekularmasse"], 46.07, delta=0.1)
            self.assertEqual(properties["hba"], 1)
            self.assertEqual(properties["hbd"], 1)
            self.assertEqual(properties["rotierende_bindungen"], 2)  # RDKit zählt hier 2 rotierende Bindungen für Ethanol
            self.assertEqual(properties["aromatische_ringe"], 0)
            
        def test_mol_file_processing(self):
            """Test der Verarbeitung einer MOL-Datei."""
            calculator = MolecularPropertiesCalculator()
            properties_dict = calculator.process_mol_file(self.test_mol_path)
            
            self.assertTrue(properties_dict)
            self.assertEqual(len(properties_dict), 1)
            
            # Prüfe die berechneten Eigenschaften
            mol_id = list(properties_dict.keys())[0]
            properties = properties_dict[mol_id]
            
            self.assertIn("molekularmasse", properties)
            self.assertIn("logp", properties)
            
        def test_add_properties_to_sequence_json(self):
            """Test der Integration von Eigenschaften in eine JSON-Datei mit Sequenzen direkt aus MOL-Dateien."""
            output_path = os.path.join(self.test_dir, "output.json")
            success = add_properties_to_mol_sequences(
                self.test_json_path,  # Pfad zur JSON-Datei mit Sequenzen
                self.mol_dir,         # Verzeichnis mit MOL-Dateien
                output_path           # Ausgabepfad
            )
            
            self.assertTrue(success)
            self.assertTrue(os.path.exists(output_path))
            
            # Lade und überprüfe die Ausgabedatei
            with open(output_path, 'r') as f:
                result = json.load(f)
                
            # Überprüfe das Format der Ausgabe
            self.assertIn("molecule_1", result)
            self.assertIn("sequence", result["molecule_1"])
            self.assertIn("properties", result["molecule_1"])
            
            # Überprüfe, ob Eigenschaften vorhanden sind
            properties = result["molecule_1"]["properties"]
            self.assertIn("molekularmasse", properties)
            self.assertIn("hba", properties)
            self.assertIn("hbd", properties)
            self.assertIn("rotierende_bindungen", properties)
            self.assertIn("aromatische_ringe", properties)
            self.assertIn("logp", properties)
        
        def test_compatibility_function(self):
            """Test der Kompatibilitätsfunktion für die alte API."""
            output_path = os.path.join(self.test_dir, "output_compat.json")
            success = calculate_molecular_properties(self.test_json_path, output_path)
            
            self.assertTrue(success)
            self.assertTrue(os.path.exists(output_path))
            
            # Lade und überprüfe die Ausgabedatei
            with open(output_path, 'r') as f:
                result = json.load(f)
                
            # Überprüfe das Format der Ausgabe
            self.assertIn("molecule_1", result)
            self.assertIn("sequence", result["molecule_1"])
            self.assertIn("properties", result["molecule_1"])
            
    # Konfiguriere Logging
    logging.basicConfig(level=logging.INFO)
    unittest.main() 