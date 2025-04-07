import os
import re
from pathlib import Path
from typing import Optional, List, Dict
from tqdm import tqdm
from PreprocessingPipeline.filters.base_processor import BaseProcessor, ProcessingConfig
import gc


"""This Script takes in a directory with many small text files each containing but one moltable. It then combines
   them into one quite long text file.
   It uses '###ID###' as a divider where ID is the ID of the molecule in the database."""


class TextFileCombiner(BaseProcessor):
    """RAM-optimierte Version des TextFileCombiner"""
    
    def __init__(self, input_directory: str, output_file: str, 
                 config: Optional[ProcessingConfig] = None):
        super().__init__(config)
        self.input_directory = Path(input_directory)
        self.output_file = Path(output_file)
        self.combined_content: List[str] = []
        self.total_size = 0
        self.divider_template = "###{}###"
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _process_batch(self) -> None:
        """Verarbeitet einen Batch von Molekülen"""
        for mol in self.molecules:
            self.combined_content.append(mol["content"])
            self.total_size += len(mol["content"])
            
            # Schreibe den Batch, wenn der Speicher zu voll wird
            if self._check_memory_usage():
                self._write_batch()
    
    def _write_batch(self) -> None:
        """Schreibt den aktuellen Batch in die Ausgabedatei"""
        if not self.combined_content:
            return
            
        mode = 'a' if self.output_file.exists() else 'w'
        with open(self.output_file, mode) as f:
            f.write(''.join(self.combined_content))
        
        self.combined_content = []
        gc.collect()  # Explizite Garbage Collection
    
    def _save_checkpoint(self) -> None:
        """Implementierung optional - nicht benötigt für diese Klasse"""
        pass
    
    def _load_checkpoint(self) -> None:
        """Implementierung optional - nicht benötigt für diese Klasse"""
        pass
    
    def run(self) -> None:
        """Hauptmethode zum Kombinieren der Dateien"""
        if not self.input_directory.exists():
            raise FileNotFoundError(f"Eingabeverzeichnis nicht gefunden: {self.input_directory}")
        
        # Erstelle Ausgabeverzeichnis falls nötig
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Sammle alle Textdateien
        text_files = list(self.input_directory.glob("*.txt"))
        self.logger.info(f"Verarbeite {len(text_files)} Textdateien...")
        
        # Verarbeite die Dateien
        with tqdm(total=len(text_files), desc="Kombiniere Dateien") as pbar:
            for text_file in text_files:
                try:
                    content = text_file.read_text()
                    self.add_molecule({
                        "filename": text_file.name,
                        "content": content
                    })
                    pbar.update(1)
                    
                except Exception as e:
                    self.logger.error(f"Fehler beim Lesen von {text_file}: {e}")
                    continue
        
        # Verarbeite verbleibende Moleküle
        self.finalize()
        
        # Schreibe verbleibenden Content
        self._write_batch()
        
        self.logger.info(f"Kombination abgeschlossen: {self.total_processed} Dateien, "
                        f"Gesamtgröße: {self.total_size/1024/1024:.2f} MB")

if __name__ == "__main__":
    # Beispielverwendung
    config = ProcessingConfig(
        batch_size=1000,
        max_ram_usage=0.8
    )
    
    combiner = TextFileCombiner(
        input_directory="data/temp/original_data/10",
        output_file="data/temp/combined/output.txt",
        config=config
    )
    combiner.run()


# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.