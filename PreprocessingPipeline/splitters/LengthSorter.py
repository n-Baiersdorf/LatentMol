import os
import shutil
from pathlib import Path
from typing import Union, Dict, List, Optional
from tqdm import tqdm
from collections import defaultdict
import re
from ..filters.base_processor import BaseProcessor, ProcessingConfig

class MoleculeFileSorter(BaseProcessor):
    """RAM-optimierte Version des MoleculeFileSorter"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        super().__init__(config)
        self.molecule_groups: Dict[int, List[Dict]] = defaultdict(list)
        self.atom_pattern = re.compile(r'V2000$', re.MULTILINE)
    
    def _extract_atom_count(self, content: str) -> int:
        """Extrahiert die Anzahl der Atome aus dem Molekülinhalt"""
        try:
            match = self.atom_pattern.search(content)
            if match:
                line_start = content.rfind('\n', 0, match.start()) + 1
                atom_count_str = content[line_start:line_start+3].strip()
                return int(atom_count_str)
            raise ValueError("Keine V2000-Zeile gefunden")
        except Exception as e:
            raise ValueError(f"Fehler beim Extrahieren der Atomanzahl: {str(e)}")
    
    def _process_batch(self) -> None:
        """Gruppiert Moleküle nach ihrer Atomanzahl"""
        for mol in self.molecules:
            try:
                atom_count = self._extract_atom_count(mol["content"])
                self.molecule_groups[atom_count].append(mol)
            except Exception as e:
                self.logger.error(f"Fehler bei {mol['filename']}: {e}")
    
    def _save_checkpoint(self) -> None:
        """Implementierung optional - nicht benötigt für diese Klasse"""
        pass
    
    def _load_checkpoint(self) -> None:
        """Implementierung optional - nicht benötigt für diese Klasse"""
        pass
    
    def _write_molecule_groups(self, output_base_path: Path) -> None:
        """Schreibt die gruppierten Moleküle in ihre jeweiligen Verzeichnisse"""
        total_molecules = sum(len(mols) for mols in self.molecule_groups.values())
        
        with tqdm(total=total_molecules, desc="Schreibe sortierte Moleküle") as pbar:
            for atom_count, molecules in self.molecule_groups.items():
                if not molecules:
                    continue
                
                # Erstelle Verzeichnis für diese Atomanzahl
                output_dir = output_base_path / str(atom_count)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Schreibe alle Moleküle dieser Gruppe
                for mol in molecules:
                    output_file = output_dir / mol["filename"]
                    output_file.write_text(mol["content"])
                    pbar.update(1)
    
    def sort_files(self, input_directory: Union[str, Path], 
                   output_base_directory: Union[str, Path]) -> None:
        """Sortiert Moleküldateien nach ihrer Atomanzahl"""
        input_path = Path(input_directory)
        output_base_path = Path(output_base_directory)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Eingabeverzeichnis nicht gefunden: {input_directory}")
        
        # Sammle alle Moleküldateien
        mol_files = list(input_path.glob("*.txt"))
        self.logger.info(f"Verarbeite {len(mol_files)} Moleküldateien...")
        
        with tqdm(total=len(mol_files), desc="Lese Moleküle") as pbar:
            for mol_file in mol_files:
                try:
                    content = mol_file.read_text()
                    self.add_molecule({
                        "filename": mol_file.name,
                        "content": content
                    })
                    pbar.update(1)
                    
                    if self._check_memory_usage():
                        self.logger.warning("Hohe RAM-Nutzung - Batch wird vorzeitig verarbeitet")
                        self._process_batch()
                        self.molecules = []
                        self.batch_counter = 0
                        
                except Exception as e:
                    self.logger.error(f"Fehler beim Lesen von {mol_file}: {e}")
                    continue
        
        # Verarbeite verbleibende Moleküle
        self.finalize()
        
        # Schreibe alle gruppierten Moleküle
        self._write_molecule_groups(output_base_path)
        
        # Statistiken ausgeben
        total_sorted = sum(len(mols) for mols in self.molecule_groups.values())
        self.logger.info(f"Sortierung abgeschlossen: {total_sorted} Moleküle in "
                        f"{len(self.molecule_groups)} Gruppen sortiert")

if __name__ == "__main__":
    # Beispielverwendung
    config = ProcessingConfig(
        batch_size=1000,
        max_ram_usage=0.8
    )
    
    sorter = MoleculeFileSorter(config=config)
    sorter.sort_files(
        input_directory="data/temp/with_Other_M",
        output_base_directory="data/temp/V_original_data"
    )

# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.