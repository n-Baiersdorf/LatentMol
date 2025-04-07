import os
import shutil
from pathlib import Path
from typing import Union, Optional
from tqdm import tqdm
from rdkit import Chem
from .base_processor import BaseProcessor, ProcessingConfig

class LengthFilterer(BaseProcessor):
    """RAM-optimierte Version des LengthFilterer"""
    
    def __init__(self, max_length: int, min_length: int = 1, count_hydrogens: bool = False, config: Optional[ProcessingConfig] = None):
        """
        Initialisiert den LengthFilterer.
        
        Args:
            max_length: Maximale Anzahl an Atomen (exklusive oder inklusive Wasserstoff, je nach count_hydrogens)
            min_length: Minimale Anzahl an Atomen (exklusive oder inklusive Wasserstoff, je nach count_hydrogens)
            count_hydrogens: Ob Wasserstoffatome bei der Zählung berücksichtigt werden sollen
            config: Optionale Konfiguration für den Basisverarbeiter
        """
        super().__init__(config)
        self.max_length = max_length
        self.min_length = min_length
        self.count_hydrogens = count_hydrogens
        self.valid_count = 0
        self.invalid_count = 0
        
        # Initialisiere Längen-Statistiken
        self._update_statistics("length_stats", {
            "min_length": min_length,
            "max_length": max_length,
            "count_hydrogens": count_hydrogens,
            "length_distribution": {},
            "valid_molecules": 0,
            "invalid_molecules": 0,
            "min_length_found": float('inf'),
            "max_length_found": 0,
            "avg_length": 0.0,
            "total_molecules": 0,
            "too_short": 0,
            "too_long": 0
        })
    
    def _extract_atom_count(self, content: str) -> int:
        """
        Extrahiert die Anzahl der Atome aus einer MDL-formatierten Datei.
        
        Berücksichtigt die count_hydrogens-Einstellung und versucht, mit RDKit zu parsen,
        falls das direkte Extrahieren fehlschlägt.
        """
        try:
            # Zuerst versuchen, die Anzahl direkt aus dem Header zu extrahieren
            lines = content.split('\n')
            for line in lines:
                if line.strip().endswith('V2000'):
                    atom_count_str = line.strip()[:3]
                    try:
                        atom_count = int(atom_count_str)
                        # Wenn wir Wasserstoffatome zählen sollen, parsen wir mit RDKit
                        if self.count_hydrogens:
                            try:
                                mol = Chem.MolFromMolBlock(content)
                                if mol is not None:
                                    # AddHs fügt explizite Wasserstoffatome hinzu
                                    mol_with_h = Chem.AddHs(mol)
                                    return mol_with_h.GetNumAtoms()
                                # Fallback zur direkten Zählung, wenn RDKit fehlschlägt
                                return atom_count
                            except:
                                self.logger.warning("RDKit konnte Molekül nicht parsen, verwende direkte Zählung")
                                return atom_count
                        else:
                            return atom_count
                    except ValueError:
                        raise ValueError("Ungültiges Format der Counts-Zeile")
            
            # Wenn wir hier ankommen, versuchen wir es mit RDKit
            mol = Chem.MolFromMolBlock(content)
            if mol is not None:
                if self.count_hydrogens:
                    mol = Chem.AddHs(mol)
                return mol.GetNumAtoms()
                
            raise ValueError("Keine gültige MDL-Tabelle gefunden")
        except Exception as e:
            raise ValueError(f"Fehler beim Extrahieren der Atomanzahl: {str(e)}")
    
    def _update_length_statistics(self, atom_count: int) -> None:
        """Aktualisiert die Längenstatistiken"""
        stats = self.statistics["step_specific"]["length_stats"]
        
        # Aktualisiere Längenverteilung
        if atom_count not in stats["length_distribution"]:
            stats["length_distribution"][atom_count] = 0
        stats["length_distribution"][atom_count] += 1
        
        # Aktualisiere Min/Max/Durchschnitt
        stats["min_length_found"] = min(stats["min_length_found"], atom_count)
        stats["max_length_found"] = max(stats["max_length_found"], atom_count)
        stats["total_molecules"] += 1
        
        # Berechne laufenden Durchschnitt
        old_avg = stats["avg_length"]
        stats["avg_length"] = old_avg + (atom_count - old_avg) / stats["total_molecules"]
        
        # Aktualisiere Valid/Invalid Zähler
        if atom_count <= self.max_length and atom_count >= self.min_length:
            stats["valid_molecules"] += 1
        else:
            stats["invalid_molecules"] += 1
            if atom_count < self.min_length:
                stats["too_short"] += 1
            else:
                stats["too_long"] += 1
    
    def _process_batch(self) -> None:
        """Verarbeitet einen Batch von Molekülen"""
        for mol in self.molecules:
            try:
                atom_count = self._extract_atom_count(mol["content"])
                self._update_length_statistics(atom_count)
                
                # Überprüfe sowohl min_length als auch max_length
                if atom_count <= self.max_length and atom_count >= self.min_length:
                    output_path = Path(mol["output_dir"]) / mol["filename"]
                    output_path.write_text(mol["content"])
                    self.valid_count += 1
                else:
                    self.invalid_count += 1
                    if atom_count < self.min_length:
                        self.logger.debug(f"Molekül {mol['filename']} hat zu wenig Atome: {atom_count} < {self.min_length}")
                    else:
                        self.logger.debug(f"Molekül {mol['filename']} hat zu viele Atome: {atom_count} > {self.max_length}")
            except Exception as e:
                self.logger.error(f"Fehler bei Molekül {mol['filename']}: {e}")
                self.invalid_count += 1
                self.statistics["errors"] += 1
    
    def _save_checkpoint(self) -> None:
        """Implementierung optional - nicht benötigt für diese Klasse"""
        pass
    
    def _load_checkpoint(self) -> None:
        """Implementierung optional - nicht benötigt für diese Klasse"""
        pass
    
    def filter_and_copy(self, input_directory: Union[str, Path], 
                       output_directory: Union[str, Path]) -> None:
        """Filtert Dateien basierend auf ihrer Atomanzahl"""
        input_path = Path(input_directory)
        output_path = Path(output_directory)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Eingabeverzeichnis nicht gefunden: {input_directory}")
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        mol_files = list(input_path.glob("*.txt"))
        self.logger.info(f"Verarbeite {len(mol_files)} Moleküldateien...")
        
        with tqdm(total=len(mol_files), desc="Längenfilterung") as pbar:
            for mol_file in mol_files:
                try:
                    content = mol_file.read_text()
                    self.add_molecule({
                        "filename": mol_file.name,
                        "content": content,
                        "output_dir": output_directory
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
        
        self.finalize()
        
        # Gib zusätzlich Informationen aus, warum Moleküle ausgeschlossen wurden
        stats = self.statistics["step_specific"]["length_stats"]
        self.logger.info(f"Filterung abgeschlossen: {self.valid_count} valide, "
                        f"{self.invalid_count} invalide Moleküle")
        self.logger.info(f"Moleküle mit zu wenigen Atomen: {stats['too_short']}, "
                        f"Moleküle mit zu vielen Atomen: {stats['too_long']}")
        self.logger.info(f"Längenbereich: Min. gefunden {stats['min_length_found']}, "
                        f"Max. gefunden {stats['max_length_found']}")

if __name__ == "__main__":
    # Beispielverwendung
    config = ProcessingConfig(
        batch_size=1000,
        max_ram_usage=0.8
    )
    
    # Verwendet jetzt min_length und count_hydrogens Parameter
    filterer = LengthFilterer(
        min_length=5,      # Minimum 5 Atome
        max_length=50,     # Maximum 50 Atome
        count_hydrogens=False,  # Wasserstoffatome nicht mitzählen
        config=config
    )
    
    filterer.filter_and_copy(
        input_directory="data/temp/split_db",
        output_directory="data/temp/filtered_lengthwise"
    )

# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.