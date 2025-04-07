import os
import json
from typing import List, Dict, Tuple, Optional
from copy import deepcopy
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import unittest
import tempfile
import shutil
import sys
from pathlib import Path
import logging

# Füge das Projektverzeichnis zum Python-Pfad hinzu
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Imports für das Package
try:
    from PreprocessingPipeline.filters.base_processor import BaseProcessor, ProcessingConfig
    from misc.PSE_Data import atom_dict_normalized as atom_constants
except ImportError as e:
    print(f"Import-Fehler: {e}")
    print("Bitte stellen Sie sicher, dass das Projekt korrekt installiert ist.")
    print("Sie können das Projekt mit 'pip install -e .' im Hauptverzeichnis installieren.")
    sys.exit(1)

class MolToSequence(BaseProcessor):
    """
    Implementiert das LatentMol-Format mit 27-dimensionalen MolTokens pro Atom.
    Jeder MolToken enthält:
    - 3 Koordinaten
    - 12 Atomkonstanten
    - 4 Bindungseinträge (je 3 Werte)
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(ProcessingConfig())
        self.config = config if config else {}
        
        # Konstanten für die Normalisierung
        self.SEM_PAD = -1
        self.MAX_BONDS = 4
        
        # Standard Normalisierungskonfiguration
        self.normalization_config = {
            "coords": {
                "min": 0.01,
                "max": 0.2,
                "use_01": False
            },
            "atom_constants": {
                "min": 0.55,
                "max": 1.0,
                "use_01": False
            },
            "bonds": {
                "min": 0.0,
                "max": 0.1,
                "use_01": False
            },
            "en_diff": {
                "min": 0.1,
                "max": 0.25,
                "use_01": False
            }
        }
        
        # Überschreibe mit Konfiguration falls vorhanden
        if "normalization" in self.config:
            for key in self.normalization_config:
                if key in self.config["normalization"]:
                    self.normalization_config[key].update(self.config["normalization"][key])

        # Initialisiere Sequenz-Statistiken
        self._update_statistics("sequence_stats", {
            "total_sequences": 0,
            "total_atoms": 0,
            "total_bonds": 0,
            "avg_sequence_length": 0.0,
            "bond_type_distribution": {},
            "atom_type_distribution": {},
            "errors": {
                "parsing": 0,
                "conversion": 0,
                "validation": 0
            }
        })

    def parse_mol_file(self, input_path: str) -> Dict[str, List[List[float]]]:
        """Verarbeitet eine Mol-Datei und erstellt MolTokens für jedes Atom."""
        molecules_dict = {}
        
        try:
            # Prüfe zuerst, ob es sich um eine SDF-Datei handeln könnte
            if input_path.lower().endswith('.sdf'):
                # Verwende SDMolSupplier für SDF-Dateien
                suppl = Chem.SDMolSupplier(input_path)
                for idx, mol in enumerate(suppl):
                    if mol is not None:
                        mol_id = str(idx + 1)
                        # Versuche, den Molekülnamen aus den Eigenschaften zu lesen
                        if mol.HasProp('_Name') and mol.GetProp('_Name'):
                            mol_id = mol.GetProp('_Name')
                        self._process_single_molecule(mol, molecules_dict, mol_id)
                if molecules_dict:
                    return molecules_dict
            
            # Versuche, mit dem FileFormat von RDKit mehrere Moleküle zu lesen
            suppl = Chem.ForwardSDMolSupplier(input_path)
            has_molecules = False
            for idx, mol in enumerate(suppl):
                if mol is not None:
                    has_molecules = True
                    mol_id = str(idx + 1)
                    self._process_single_molecule(mol, molecules_dict, mol_id)
            if has_molecules:
                return molecules_dict
                
            # Wenn das nicht funktioniert, parsen wir die Datei zeilenweise
            with open(input_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # Teile den Inhalt in Moleküle auf, wenn die Datei mehrere Moleküle enthält
            # Suche nach Trennungsmarkern zwischen Molekülen
            if "$$$$" in content:
                # SDF-Format mit mehreren Molekülen
                mol_blocks = content.split("$$$$")
                for idx, block in enumerate(mol_blocks):
                    if block.strip():  # Ignoriere leere Blöcke
                        mol = Chem.MolFromMolBlock(block + "$$$$")  # Füge Trennzeichen wieder an
                        if mol is not None:
                            mol_id = str(idx + 1)
                            self._process_single_molecule(mol, molecules_dict, mol_id)
                if molecules_dict:
                    return molecules_dict
            
            # Wenn immer noch nichts gefunden, parsen wir zeilenweise und suchen nach nummerierten Molekülen
            lines = content.split('\n')
            current_mol_lines = []
            current_mol_id = None
            
            for line in lines:
                line = line.strip()
                
                if line and line.isdigit():
                    # Wenn wir bereits ein Molekül sammeln, verarbeite es
                    if current_mol_lines and current_mol_id:
                        self._process_molecule(current_mol_id, current_mol_lines, molecules_dict)
                        current_mol_lines = []
                    current_mol_id = line
                else:
                    current_mol_lines.append(line)
            
            if current_mol_lines and current_mol_id:
                self._process_molecule(current_mol_id, current_mol_lines, molecules_dict)
            
            # Wenn immer noch nichts gefunden wurde, versuche es als einzelnes Molekül
            if not molecules_dict:
                mol = Chem.MolFromMolBlock(content)
                if mol is not None:
                    self._process_single_molecule(mol, molecules_dict, "1")
            
            return molecules_dict
        except Exception as e:
            self.logger.error(f"Fehler beim Lesen der Datei {input_path}: {e}")
            self.statistics["step_specific"]["sequence_stats"]["errors"]["parsing"] += 1
            return {}

    def _process_single_molecule(self, mol: Chem.Mol, molecules_dict: Dict, mol_id: str):
        """Verarbeitet ein einzelnes RDKit-Mol-Objekt."""
        try:
            # Füge explizite Wasserstoffatome hinzu
            mol = Chem.AddHs(mol)
            
            # Generiere 3D-Koordinaten
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception as e:
                self.logger.warning(f"Konnte 3D-Konformation nicht generieren: {e}")
                # Versuche alternative Methode
                AllChem.Compute2DCoords(mol)
            
            # Extrahiere Atome und Bindungen
            atoms = []
            conf = mol.GetConformer()
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                atom = mol.GetAtomWithIdx(i)
                atoms.append({
                    'coords': [pos.x, pos.y, pos.z],
                    'element': atom.GetSymbol(),
                    'index': i + 1  # 1-basierte Indizierung
                })
            
            bonds = []
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtomIdx() + 1
                a2 = bond.GetEndAtomIdx() + 1
                bond_type = bond.GetBondType()
                bond_count = 1 if bond_type == Chem.BondType.SINGLE else \
                            2 if bond_type == Chem.BondType.DOUBLE else \
                            3 if bond_type == Chem.BondType.TRIPLE else 1
                for _ in range(bond_count):
                    bonds.append([a1, a2])
            
            # Erstelle MolTokens
            mol_tokens = self._create_mol_tokens(atoms, bonds)
            
            # Speichere mit Molekül-ID
            molecules_dict[f"molecule_{mol_id}"] = mol_tokens
            self.logger.info(f"Erfolgreich verarbeitet: {mol_id}")
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Verarbeitung von Molekül {mol_id}: {e}")
            self.statistics["step_specific"]["sequence_stats"]["errors"]["conversion"] += 1

    def _process_molecule(self, mol_id: str, lines: List[str], molecules_dict: Dict):
        """Verarbeitet ein einzelnes Molekül und erstellt MolTokens."""
        try:
            # Erstelle RDKit Molekül
            mol_block = '\n'.join(lines)
            mol = Chem.MolFromMolBlock(mol_block)
            if mol is None:
                self.logger.warning(f"Konnte Molekül {mol_id} nicht mit RDKit parsen, versuche mit bereinigten Daten.")
                # Versuche mit bereinigten Daten
                cleaned_lines = []
                for line in lines:
                    if len(line.split()) >= 4:  # Könnte eine Atomzeile sein
                        parts = line.split()
                        try:
                            # Versuche die Teile als Float zu interpretieren
                            float(parts[0]), float(parts[1]), float(parts[2])
                            # Wenn erfolgreich, behalte die Zeile bei
                            cleaned_lines.append(line)
                        except ValueError:
                            # Keine Atomzeile, behalte unverändert
                            cleaned_lines.append(line)
                    else:
                        cleaned_lines.append(line)
                        
                mol = Chem.MolFromMolBlock('\n'.join(cleaned_lines))
                if mol is None:
                    raise ValueError(f"Konnte Molekül {mol_id} nicht parsen")
            
            # Verarbeite das Molekül mit dem gemeinsamen Code
            self._process_single_molecule(mol, molecules_dict, mol_id)
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Verarbeitung von Molekül {mol_id}: {e}")
            self.statistics["step_specific"]["sequence_stats"]["errors"]["conversion"] += 1

    def _create_mol_tokens(self, atoms: List[Dict], bonds: List[List[int]]) -> List[List[float]]:
        """Erstellt MolTokens für jedes Atom."""
        mol_tokens = []
        
        for atom in atoms:
            # 1. Koordinaten (3 Werte)
            coords = self._normalize_coordinates([atom['coords']])[0]
            
            # 2. Atomkonstanten (12 Werte)
            atom_consts = self._get_normalized_atom_constants(atom['element'])
            
            # 3. Bindungseinträge (4 * 3 = 12 Werte)
            bond_entries = self._create_bond_entries(atom, atoms, bonds)
            
            # Kombiniere alle Werte zu einem MolToken
            mol_token = coords + atom_consts + bond_entries
            mol_tokens.append(mol_token)
        
        return mol_tokens

    def _create_bond_entries(self, atom: Dict, all_atoms: List[Dict], all_bonds: List[List[int]]) -> List[float]:
        """Erstellt die vier Bindungseinträge für ein Atom."""
        atom_idx = atom['index']
        element = atom['element']
        
        # Finde alle Bindungen für dieses Atom
        atom_bonds = [bond for bond in all_bonds if atom_idx in bond]
        
        # Verarbeite die Bindungen
        processed_bonds = []
        for bond in atom_bonds:
            partner_idx = bond[0] if bond[0] != atom_idx else bond[1]
            partner_atom = next((a for a in all_atoms if a['index'] == partner_idx), None)
            if partner_atom is None:
                self.logger.warning(f"Partneratom mit Index {partner_idx} nicht gefunden")
                continue
                
            # Berechne Elektronegativitätsdifferenz
            en_diff = self._calculate_en_diff(element, partner_atom['element'])
            
            # Normalisiere Indizes
            norm_idx = self._normalize_bond_index(atom_idx)
            norm_partner = self._normalize_bond_index(partner_idx)
            
            processed_bonds.append([norm_idx, norm_partner, en_diff])
        
        # Fülle mit freien Elektronenpaaren auf
        while len(processed_bonds) < self.MAX_BONDS:
            if element == 'H':
                # Wasserstoff: Fülle mit Platzhaltern
                processed_bonds.append([self.SEM_PAD, self.SEM_PAD, self.SEM_PAD])
            elif element == 'C':
                # Kohlenstoff: Fülle mit impliziten C-H Bindungen
                norm_idx = self._normalize_bond_index(atom_idx)
                ch_en_diff = self._calculate_en_diff('C', 'H')
                processed_bonds.append([norm_idx, self.SEM_PAD, ch_en_diff])
            else:
                # Andere Elemente: Fülle mit freien Elektronenpaaren
                norm_idx = self._normalize_bond_index(atom_idx)
                processed_bonds.append([norm_idx, norm_idx, self.SEM_PAD])
        
        # Flatten die Liste
        flattened = []
        for bond in processed_bonds[:self.MAX_BONDS]:  # Begrenze auf MAX_BONDS
            flattened.extend(bond)
        return flattened

    def _normalize_coordinates(self, atom_coords: List[List[float]]) -> List[List[float]]:
        """Normalisiert Koordinaten entsprechend der Konfiguration."""
        if not atom_coords:
            return []
            
        # Konvertiere zu numpy-Array für einfachere Berechnung
        coords = np.array(atom_coords)
        
        # Finde min/max für jede Dimension
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        
        # Vermeide Division durch Null
        ranges = max_coords - min_coords
        ranges[ranges == 0] = 1.0
        
        if self.normalization_config["coords"]["use_01"]:
            # [0,1] Normalisierung
            normalized = (coords - min_coords) / ranges
        else:
            # Intervallbasierte Normalisierung
            target_min = self.normalization_config["coords"]["min"]
            target_max = self.normalization_config["coords"]["max"]
            normalized = ((coords - min_coords) / ranges) * (target_max - target_min) + target_min
        
        return normalized.tolist()

    def _normalize_bond_index(self, index: int) -> float:
        """Normalisiert Bindungsindizes entsprechend der Konfiguration."""
        if self.normalization_config["bonds"]["use_01"]:
            return index / 100  # Annahme: max 100 Atome
        else:
            target_min = self.normalization_config["bonds"]["min"]
            target_max = self.normalization_config["bonds"]["max"]
            return index * (target_max - target_min) / 100 + target_min

    def _calculate_en_diff(self, element1: str, element2: str) -> float:
        """Berechnet die normalisierte Elektronegativitätsdifferenz."""
        # Überprüfe, ob die Elemente in den Atomkonstanten vorhanden sind
        if element1 not in atom_constants:
            self.logger.warning(f"Element {element1} nicht in Atomkonstanten gefunden, verwende Standardwerte")
            return 0.2  # Rückfall-Wert
        if element2 not in atom_constants:
            self.logger.warning(f"Element {element2} nicht in Atomkonstanten gefunden, verwende Standardwerte")
            return 0.2  # Rückfall-Wert
            
        en1 = atom_constants[element1][5]  # Elektronegativität ist der 6. Wert
        en2 = atom_constants[element2][5]
        raw_diff = abs(en1 - en2)
        
        if self.normalization_config["en_diff"]["use_01"]:
            return raw_diff / 3.20  # Maximal mögliche Differenz
        else:
            target_min = self.normalization_config["en_diff"]["min"]
            target_max = self.normalization_config["en_diff"]["max"]
            return (raw_diff / 3.20) * (target_max - target_min) + target_min

    def _get_normalized_atom_constants(self, element: str) -> List[float]:
        """Normalisiert Atomkonstanten entsprechend der Konfiguration."""
        if element not in atom_constants:
            self.logger.warning(f"Element {element} nicht in Konstanten gefunden, verwende Standardwerte")
            # Rückfallwerte für unbekannte Elemente
            return [0.75] * 12
            
        values = atom_constants[element]
        normalized = []
        
        for i, val in enumerate(values):
            all_vals_for_i = [atom_constants[e][i] for e in atom_constants]
            p_min = min(all_vals_for_i)
            p_max = max(all_vals_for_i)
            
            if self.normalization_config["atom_constants"]["use_01"]:
                # Einfache [0,1] Normalisierung
                if p_max - p_min == 0.0:
                    scaled = 0.5
                else:
                    scaled = (val - p_min) / (p_max - p_min)
            else:
                # Intervallbasierte Normalisierung
                target_min = self.normalization_config["atom_constants"]["min"]
                target_max = self.normalization_config["atom_constants"]["max"]
                if p_max - p_min == 0.0:
                    scaled = (target_min + target_max) / 2
                else:
                    scaled = ((val - p_min) / (p_max - p_min)) * (target_max - target_min) + target_min
                    
            normalized.append(scaled)
        return normalized

def molToSequenceFunction(input_file: str, output_dir: str, number: int, config: Optional[Dict] = None) -> bool:
    """Hauptfunktion für die Sequenzgenerierung"""
    try:
        if not os.path.exists(input_file):
            logging.error(f"Eingabedatei {input_file} existiert nicht!")
            return False
            
        if not os.path.exists(output_dir):
            logging.error(f"Ausgabeverzeichnis {output_dir} existiert nicht!")
            return False
            
        if not os.access(output_dir, os.W_OK):
            logging.error(f"Keine Schreibrechte im Verzeichnis {output_dir}")
            return False
        
        processor = MolToSequence(config)
        molecules_dict = processor.parse_mol_file(input_file)
        
        if not molecules_dict:
            logging.error("Keine Moleküle gefunden oder verarbeitet!")
            return False
            
        output_file = os.path.join(output_dir, f"Länge_{number}.json")
        with open(output_file, 'w') as f:
            json.dump(molecules_dict, f)
            
        return True
        
    except Exception as e:
        logging.error(f"Fehler in molToSequenceFunction: {str(e)}")
        return False

if __name__ == "__main__":
    class TestMolToSequence(unittest.TestCase):
        def setUp(self):
            self.test_dir = tempfile.mkdtemp()
            self.test_mol = """H2O
     RDKit          3D

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    1.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    1.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END"""
            
            self.test_file = os.path.join(self.test_dir, "test.mol")
            with open(self.test_file, 'w') as f:
                f.write(self.test_mol)
                
        def tearDown(self):
            shutil.rmtree(self.test_dir)
            
        def test_sequence_generation(self):
            config = {
                "normalization": {
                    "coords": {"use_01": True},
                    "atom_constants": {"use_01": False},
                    "bonds": {"use_01": False},
                    "en_diff": {"use_01": False}
                }
            }
            
            success = molToSequenceFunction(
                self.test_file,
                self.test_dir,
                1,
                config=config
            )
            
            self.assertTrue(success, "Sequenzgenerierung fehlgeschlagen")
            
            output_file = os.path.join(self.test_dir, "Länge_1.json")
            self.assertTrue(os.path.exists(output_file), "Ausgabedatei existiert nicht")
            
            with open(output_file, 'r') as f:
                data = json.load(f)
                
            self.assertTrue(data, "Ausgabedatei ist leer")
            
            # Überprüfe MolToken-Struktur
            for key, sequences in data.items():
                self.assertTrue(sequences, f"Keine Sequenzen für {key}")
                for seq in sequences:
                    self.assertEqual(len(seq), 27, "MolToken hat nicht die richtige Länge")
                    # Überprüfe Koordinaten
                    for coord in seq[:3]:
                        self.assertGreaterEqual(coord, 0)
                        self.assertLessEqual(coord, 1)
                    # Überprüfe Atomkonstanten
                    for const in seq[3:15]:
                        self.assertGreaterEqual(const, 0.55)
                        self.assertLessEqual(const, 1.0)
                    # Überprüfe Bindungseinträge
                    for i in range(4):
                        bond_start = 15 + i * 3
                        bond_end = bond_start + 3
                        bond = seq[bond_start:bond_end]
                        self.assertEqual(len(bond), 3, "Bindungseintrag hat nicht die richtige Länge")
    
    # Konfiguriere Logging
    logging.basicConfig(level=logging.INFO)
    unittest.main()

# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.
