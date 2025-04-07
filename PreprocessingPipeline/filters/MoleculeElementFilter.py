import os
from pathlib import Path
from typing import List, Dict, Set, Optional
import re
from tqdm import tqdm
from .base_processor import BaseProcessor, ProcessingConfig


"""The MoleculeElementFilter Script is receiving a directory filled with files containing one Moltable each.
   It checks all elements present and based on the specified selection filteres all the moltables out that contain 
   elements not part of the specified list. This is because some elements might be present in such a small amount,
   and removing them may make the dataset more robust."""


class MoleculeElementFilter(BaseProcessor):
    """RAM-optimierte Version des MoleculeElementFilter"""
    
    def __init__(self, input_directory: str, valid_output_dir: str,
                 invalid_output_dir: str, allowed_elements: List[str],
                 config: Optional[ProcessingConfig] = None):
        super().__init__(config)
        self.input_directory = Path(input_directory)
        self.valid_output_dir = Path(valid_output_dir)
        self.invalid_output_dir = Path(invalid_output_dir)
        self.allowed_elements = set(allowed_elements)
        
        # Erstelle Ausgabeverzeichnisse
        self.valid_output_dir.mkdir(parents=True, exist_ok=True)
        self.invalid_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Kompiliere Regex für bessere Performance
        self.element_pattern = re.compile(r'^\s*\d+\s+(\w+)')
        
        # Initialisiere Element-Statistiken
        self._update_statistics("element_stats", {
            "allowed_elements": list(allowed_elements),
            "found_elements": {},
            "valid_molecules": 0,
            "invalid_molecules": 0,
            "element_violations": {}
        })
    
    def _extract_elements(self, molecule_content: str) -> Set[str]:
        """Extrahiert alle chemischen Elemente aus einem Molekül"""
        elements = set()
        lines = molecule_content.split('\n')
        
        # Überspringe Header (3 Zeilen) und finde die Anzahl der Atome
        if len(lines) < 4:
            return elements
            
        counts_line = lines[3].strip()
        try:
            num_atoms = int(counts_line[:3])
        except ValueError:
            return elements
        
        # Analysiere die Atomzeilen
        for line_num in range(4, min(4 + num_atoms, len(lines))):
            if match := self.element_pattern.match(lines[line_num]):
                element = match.group(1)
                elements.add(element)
                # Aktualisiere Element-Statistiken
                stats = self.statistics["step_specific"]["element_stats"]
                if element not in stats["found_elements"]:
                    stats["found_elements"][element] = 0
                stats["found_elements"][element] += 1
        
        return elements
    
    def _process_batch(self) -> None:
        """Verarbeitet einen Batch von Molekülen"""
        valid_molecules = []
        invalid_molecules = []
        
        for mol in self.molecules:
            elements = self._extract_elements(mol["content"])
            invalid_elements = [elem for elem in elements if elem not in self.allowed_elements]
            
            if not invalid_elements:
                valid_molecules.append(mol)
                self.statistics["step_specific"]["element_stats"]["valid_molecules"] += 1
            else:
                invalid_molecules.append(mol)
                self.statistics["step_specific"]["element_stats"]["invalid_molecules"] += 1
                # Erfasse Verstöße gegen die Element-Regeln
                for elem in invalid_elements:
                    if elem not in self.statistics["step_specific"]["element_stats"]["element_violations"]:
                        self.statistics["step_specific"]["element_stats"]["element_violations"][elem] = 0
                    self.statistics["step_specific"]["element_stats"]["element_violations"][elem] += 1
        
        # Speichere valide Moleküle
        for mol in valid_molecules:
            output_path = self.valid_output_dir / mol["filename"]
            output_path.write_text(mol["content"])
        
        # Speichere invalide Moleküle
        for mol in invalid_molecules:
            output_path = self.invalid_output_dir / mol["filename"]
            output_path.write_text(mol["content"])
        
        self.logger.info(f"Batch verarbeitet: {len(valid_molecules)} valide, "
                        f"{len(invalid_molecules)} invalide Moleküle")
    
    def _save_checkpoint(self) -> None:
        """Implementierung optional - nicht benötigt für diese Klasse"""
        pass
    
    def _load_checkpoint(self) -> None:
        """Implementierung optional - nicht benötigt für diese Klasse"""
        pass
    
    def filter_molecules(self) -> None:
        """Hauptmethode zum Filtern der Moleküle"""
        try:
            # Sammle alle Moleküldateien
            mol_files = list(self.input_directory.glob("*.txt"))
            
            with tqdm(total=len(mol_files), desc="Filtere Moleküle") as pbar:
                for mol_file in mol_files:
                    try:
                        content = mol_file.read_text()
                        self.add_molecule({
                            "filename": mol_file.name,
                            "content": content
                        })
                        pbar.update(1)
                        
                    except Exception as e:
                        self.logger.error(f"Fehler beim Lesen von {mol_file}: {e}")
                        continue
            
            self.finalize()
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Molekülfilterung: {e}")
            raise

if __name__ == "__main__":
    # Beispielverwendung
    config = ProcessingConfig(
        batch_size=1000,
        max_ram_usage=0.8
    )
    
    allowed_elements = ["H", "C", "O", "N", "F", "P", "S", "Cl", "Br", "I"]
    
    filter_processor = MoleculeElementFilter(
        input_directory='data/temp/split_db',
        valid_output_dir='data/temp/valid_files',
        invalid_output_dir='data/temp/invalid_files',
        allowed_elements=allowed_elements,
        config=config
    )
    
    filter_processor.filter_molecules()

# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.