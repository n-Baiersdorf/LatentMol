from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import gc
import os
import json
from datetime import datetime

@dataclass
class ProcessingConfig:
    """Basiskonfiguration für die Verarbeitung"""
    batch_size: int = 1000  # Anzahl der Moleküle pro Batch
    max_ram_usage: float = 0.8  # Maximale RAM-Nutzung (80% des verfügbaren RAMs)
    checkpoint_interval: int = 1000  # Anzahl der Moleküle zwischen Checkpoints
    log_level: int = logging.INFO
    report_dir: str = "processing_reports"

class BaseProcessor:
    """Basisklasse für RAM-basierte Molekülverarbeitung"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.logger = self._setup_logger()
        self.molecules: List[Dict[str, Any]] = []
        self.batch_counter = 0
        self.total_processed = 0
        self.statistics: Dict[str, Any] = {
            "processor_name": self.__class__.__name__,
            "start_time": datetime.now().isoformat(),
            "total_processed": 0,
            "errors": 0,
            "warnings": 0,
            "step_specific": {}
        }
        
        # Erstelle Report-Verzeichnis
        os.makedirs(self.config.report_dir, exist_ok=True)
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(self.config.log_level)
        return logger
    
    def _check_memory_usage(self) -> bool:
        """Überprüft die aktuelle RAM-Nutzung"""
        try:
            # Versuche zuerst psutil
            try:
                import psutil
                process = psutil.Process()
                memory_percent = process.memory_percent()
                if memory_percent > self.config.max_ram_usage * 100:
                    self.logger.warning(f"Hohe RAM-Nutzung (psutil): {memory_percent:.1f}%")
                    return True
                return False
            except ImportError:
                pass
            
            # Fallback: Lies direkt aus /proc/meminfo (Linux)
            if os.path.exists('/proc/meminfo'):
                with open('/proc/meminfo', 'r') as f:
                    meminfo = {}
                    for line in f:
                        key, value = line.split(':')
                        value = value.strip()
                        if value.endswith('kB'):
                            value = int(value.rstrip('kB')) * 1024
                        meminfo[key] = value
                
                total = int(meminfo['MemTotal'])
                available = int(meminfo['MemAvailable'])
                used_percent = (total - available) / total * 100
                
                if used_percent > self.config.max_ram_usage * 100:
                    self.logger.warning(f"Hohe RAM-Nutzung (meminfo): {used_percent:.1f}%")
                    return True
                return False
            
            # Wenn keine Methode verfügbar ist
            self.logger.warning("Keine RAM-Überwachung möglich - fahre mit reduzierter Batch-Größe fort")
            self.config.batch_size = max(100, self.config.batch_size // 2)
            return False
            
        except Exception as e:
            self.logger.warning(f"Fehler bei RAM-Überwachung: {e}")
            return False
    
    def _process_batch(self) -> None:
        """Verarbeitet den aktuellen Batch von Molekülen"""
        raise NotImplementedError("Muss von der Kindklasse implementiert werden")
    
    def _save_checkpoint(self) -> None:
        """Speichert einen Checkpoint der aktuellen Verarbeitung"""
        raise NotImplementedError("Muss von der Kindklasse implementiert werden")
    
    def _load_checkpoint(self) -> None:
        """Lädt einen gespeicherten Checkpoint"""
        raise NotImplementedError("Muss von der Kindklasse implementiert werden")
    
    def add_molecule(self, molecule: Dict[str, Any]) -> None:
        """Fügt ein Molekül zum aktuellen Batch hinzu"""
        self.molecules.append(molecule)
        self.batch_counter += 1
        
        if self.batch_counter >= self.config.batch_size:
            self._process_batch()
            self.molecules = []
            self.batch_counter = 0
            gc.collect()  # Explizite Garbage Collection nach Batch-Verarbeitung
        
        self.total_processed += 1
        if self.total_processed % self.config.checkpoint_interval == 0:
            self._save_checkpoint()
    
    def _update_statistics(self, key: str, value: Any) -> None:
        """Aktualisiert die Statistiken für den aktuellen Verarbeitungsschritt"""
        if isinstance(value, dict):
            if key not in self.statistics["step_specific"]:
                self.statistics["step_specific"][key] = {}
            self.statistics["step_specific"][key].update(value)
        else:
            self.statistics["step_specific"][key] = value

    def _save_report(self) -> None:
        """Speichert den Verarbeitungsbericht als JSON-Datei"""
        self.statistics["end_time"] = datetime.now().isoformat()
        self.statistics["total_processed"] = self.total_processed
        
        report_file = os.path.join(
            self.config.report_dir,
            f"{self.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.statistics, f, indent=2)
        
        self.logger.info(f"Verarbeitungsbericht gespeichert in: {report_file}")

    def finalize(self) -> None:
        """Verarbeitet verbleibende Moleküle und speichert den Bericht"""
        if self.molecules:
            self._process_batch()
        self._save_report()
        self.logger.info(f"Verarbeitung abgeschlossen. Gesamt verarbeitet: {self.total_processed}") 