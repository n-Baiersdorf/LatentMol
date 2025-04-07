import os
import re
from tqdm import tqdm
from typing import Dict, List, Optional, Generator, Tuple
import numpy as np
from pathlib import Path
from ..filters.base_processor import BaseProcessor, ProcessingConfig
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import io


'''The ChemicalDataSplitter class takes in a long input text file - i.e. from PubChem - and splits all the entries 
   into single text files in the output directory.'''

class ChemicalDataSplitter(BaseProcessor):
    """Optimierte Version des ChemicalDataSplitter für maximale CPU- und RAM-Nutzung"""

    def __init__(self, input_file: str, output_directory: str, config: Optional[ProcessingConfig] = None):
        super().__init__(config)
        self.input_file = Path(input_file)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.output_directory / "checkpoint.json"
        
        # Optimierte Ressourcennutzung
        self.num_cpus = mp.cpu_count()
        self.total_ram = psutil.virtual_memory().total
        
        # Dynamische Chunk-Größe basierend auf verfügbarem RAM
        if self.total_ram > 32 * 1024 * 1024 * 1024:  # Mehr als 32GB RAM
            self.target_ram_usage = int(self.total_ram * 0.8)  # 80% der verfügbaren RAM
            self.chunk_size = max(100 * 1024 * 1024, self.target_ram_usage // (self.num_cpus * 2))  # Min. 100MB pro Chunk
        else:
            self.target_ram_usage = int(self.total_ram * 0.6)  # 60% der verfügbaren RAM
            self.chunk_size = max(50 * 1024 * 1024, self.target_ram_usage // (self.num_cpus * 2))  # Min. 50MB pro Chunk
        
        self._load_checkpoint()

    def extract_db_id(self, entry: str) -> Optional[str]:
        """Extrahiert die ID aus verschiedenen Formaten"""
        patterns = [
            r'> \s*(\d+)',              # Format: '> <number>'
            r'nmrshiftdb2\s+(\d+)',     # Format: 'nmrshiftdb2 <number>'
            r'^\s*(\d+)\s*$',           # Format: numerische Zeilen
            r'-OEChem-(\d+)'            # Format: '-OEChem-...'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, entry, re.MULTILINE)
            if match:
                return match.group(1)
        return None

    def _process_chunk(self, chunk_data: Tuple[bytes, int]) -> List[Dict]:
        """Optimierte Chunk-Verarbeitung mit effizienter Molekül-Erkennung"""
        chunk, position = chunk_data
        molecules = []
        
        # Suche nach Molekül-Trennern
        separators = [m.start() for m in re.finditer(b'\$\$\$\$', chunk)]
        
        if not separators:
            return molecules
            
        # Verarbeite Moleküle zwischen den Trennern
        for i in range(len(separators)):
            start = separators[i] + 4 if i > 0 else 0
            end = separators[i]
            
            if start >= end:
                continue
                
            molecule_data = chunk[start:end].decode('utf-8', errors='ignore')
            
            # Extrahiere Molekül-ID
            mol_id = f"molecule_{position + start}"
            
            molecules.append({
                'id': mol_id,
                'content': molecule_data,
                'position': position + start
            })
            
        return molecules

    def _parallel_process_chunks(self, chunks: List[Tuple[bytes, int]]) -> None:
        """Optimierte parallele Chunk-Verarbeitung mit ThreadPoolExecutor"""
        # Dynamische Thread-Anzahl basierend auf CPU-Kernen
        num_threads = min(self.num_cpus * 2, 32)  # Maximal 32 Threads
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Verarbeite Chunks in Batches
            batch_size = max(1, len(chunks) // num_threads)
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i+batch_size]
                futures = [executor.submit(self._process_chunk, chunk) for chunk in batch]
                
                for future in as_completed(futures):
                    try:
                        molecules = future.result()
                        for mol in molecules:
                            output_file = self.output_directory / f"molecule_{mol['id']}.txt"
                            output_file.write_text(mol['content'])
                            self.total_processed += 1
                            
                            if self.total_processed % self.config.checkpoint_interval == 0:
                                self.last_position = mol['position']
                                self._save_checkpoint()
                                
                            # Überwache Speichernutzung
                            if psutil.Process().memory_percent() > 80:
                                self.logger.warning("Hohe Speichernutzung - reduziere Batch-Größe")
                                batch_size = max(1, batch_size // 2)
                                
                    except Exception as e:
                        self.logger.error(f"Fehler bei der Chunk-Verarbeitung: {e}")

    def _save_checkpoint(self) -> None:
        """Speichert den aktuellen Verarbeitungsfortschritt"""
        checkpoint_data = {
            "total_processed": self.total_processed,
            "last_position": self.last_position
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        self.logger.info(f"Checkpoint gespeichert: {self.total_processed} Moleküle verarbeitet")

    def _load_checkpoint(self) -> None:
        """Lädt den letzten Checkpoint, falls vorhanden"""
        self.last_position = 0
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    checkpoint_data = json.load(f)
                self.total_processed = checkpoint_data["total_processed"]
                self.last_position = checkpoint_data["last_position"]
                self.logger.info(f"Checkpoint geladen: Fortsetzen ab Position {self.last_position}")
            except Exception as e:
                self.logger.warning(f"Fehler beim Laden des Checkpoints: {e}")

    def split_file(self) -> None:
        """Optimierte Dateiaufteilung mit effizienter Ressourcennutzung"""
        try:
            file_size = os.path.getsize(self.input_file)
            current_position = self.last_position if hasattr(self, 'last_position') else 0
            
            chunks = []
            
            with open(self.input_file, 'rb') as file:
                with tqdm(total=file_size, initial=current_position,
                         desc="Lese und verarbeite Chunks", unit="B", unit_scale=True) as pbar:
                    
                    file.seek(current_position)
                    while current_position < file_size:
                        chunk = file.read(self.chunk_size)
                        if not chunk:
                            break
                            
                        # Finde das letzte vollständige Molekül im Chunk
                        last_separator = chunk.rfind(b"$$$$")
                        if last_separator != -1:
                            process_chunk = chunk[:last_separator + 4]
                            file.seek(current_position + last_separator + 4)
                        else:
                            process_chunk = chunk
                            
                        chunks.append((process_chunk, current_position))
                        current_position = file.tell()
                        pbar.update(len(process_chunk))
                        
                        # Verarbeite Chunks wenn genug RAM verfügbar
                        if len(chunks) >= self.num_cpus or psutil.virtual_memory().percent > 80:
                            self._parallel_process_chunks(chunks)
                            chunks = []
                            
                    # Verarbeite verbleibende Chunks
                    if chunks:
                        self._parallel_process_chunks(chunks)
            
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                
        except Exception as e:
            self.logger.error(f"Fehler bei der Verarbeitung: {e}")
            self._save_checkpoint()
            raise

if __name__ == "__main__":
    # Beispielverwendung
    config = ProcessingConfig(
        batch_size=500,
        max_ram_usage=0.8,
        checkpoint_interval=1000
    )
    
    splitter = ChemicalDataSplitter(
        input_file='data/input.sdf',
        output_directory='data/temp/split_db',
        config=config
    )
    splitter.split_file()

# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.