import os
import json
import logging
from typing import List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from tqdm import tqdm

@dataclass
class ProcessingResult:
    filename: str
    success: bool
    error_message: str = None
    processing_time: float = None

class JSONListMerger:
    def __init__(self, input_dir: str, output_dir: str, max_workers: int = 4):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_workers = max_workers
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'processing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def validate_json_structure(self, data: List[List[Any]]) -> bool:
        if not isinstance(data, list):
            raise ValueError("Hauptstruktur muss eine Liste sein")
        if len(data) % 2 != 0:
            raise ValueError(f"Die Liste enthält {len(data)} Elemente. Es muss eine gerade Anzahl sein.")
        embedding_length = len(data[0])
        for i in range(0, len(data), 2):
            if not isinstance(data[i], list):
                raise ValueError(f"Element {i} ist keine Liste")
            if len(data[i]) != embedding_length:
                raise ValueError(f"Embedding {i} hat abweichende Länge: {len(data[i])} statt {embedding_length}")
        positional_length = len(data[1])
        for i in range(1, len(data), 2):
            if not isinstance(data[i], list):
                raise ValueError(f"Element {i} ist keine Liste")
            if len(data[i]) != positional_length:
                raise ValueError(f"Positional Embedding {i} hat abweichende Länge: {len(data[i])} statt {positional_length}")
        return True

    def merge_sublists(self, data: List[List[Any]]) -> List[List[Any]]:
        if len(data) % 2 != 0:
            raise ValueError("Die Liste muss eine gerade Anzahl von Sublisten enthalten!")
        merged_list = []
        for i in range(0, len(data), 2):
            combined = data[i] + data[i + 1]
            merged_list.append(combined)
        return merged_list
        
    def process_file(self, filename: str) -> ProcessingResult:
        start_time = datetime.now()
        input_path = os.path.join(self.input_dir, filename)
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    return ProcessingResult(filename, False, f"Ungültiges JSON-Format in {filename}: {str(e)}")
            
            try:
                self.validate_json_structure(data)
            except ValueError as e:
                return ProcessingResult(filename, False, f"Ungültige Struktur in {filename}: {str(e)}")
            
            processed_data = self.merge_sublists(data)
            
            expected_length = len(processed_data[0])
            for i, merged_list in enumerate(processed_data):
                if len(merged_list) != expected_length:
                    return ProcessingResult(filename, False, f"Kombinierte Liste {i} hat unerwartete Länge: {len(merged_list)} statt {expected_length}")
            
            os.makedirs(self.output_dir, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            return ProcessingResult(filename, True, processing_time=processing_time)
            
        except Exception as e:
            return ProcessingResult(filename, False, f"Unerwarteter Fehler bei {filename}: {str(e)}")
    
    def process_directory(self) -> List[ProcessingResult]:
        if not os.path.exists(self.input_dir):
            raise ValueError(f"Eingabeverzeichnis {self.input_dir} existiert nicht!")
        
        json_files = [f for f in os.listdir(self.input_dir) if f.endswith('.json')]
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {executor.submit(self.process_file, filename): filename for filename in json_files}
            
            for future in tqdm(as_completed(future_to_file), total=len(json_files), desc="Processing files"):
                result = future.result()
                results.append(result)
        
        successful = len([r for r in results if r.success])
        failed = len(results) - successful
        
        logging.info(f"Verarbeitung abgeschlossen: Erfolgreich: {successful}, Fehlgeschlagen: {failed}, Gesamt: {len(results)}")
        
        return results
