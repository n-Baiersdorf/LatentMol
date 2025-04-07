#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dieses Skript berechnet molekulare Eigenschaften für bereits vorhandene LatentMol-Sequenzdateien.
Es kann auf einzelne Dateien oder ganze Verzeichnisse angewendet werden.

Molekulare Eigenschaften umfassen:
- Molekularmasse
- Wasserstoffbrückenakzeptoren (HBA)
- Wasserstoffbrückendonoren (HBD)
- Rotierende Bindungen
- Aromatische Ringe
- LogP-Wert

Beispielverwendung:
    python add_molecular_properties.py --input path/to/sequences.json --output path/to/output.json
    python add_molecular_properties.py --input path/to/directory --recursive
"""

import os
import sys
import argparse
import logging
import json
import glob
import multiprocessing
from pathlib import Path
from tqdm import tqdm

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Imports für das Package
try:
    from PreprocessingPipeline.MolecularProperties import calculate_molecular_properties
except ImportError as e:
    print(f"Import-Fehler: {e}")
    print("Bitte stellen Sie sicher, dass das Projekt korrekt installiert ist.")
    print("Sie können das Projekt mit 'pip install -e .' im Hauptverzeichnis installieren.")
    sys.exit(1)

def setup_logging(verbose=False):
    """Konfiguriert das Logging"""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('property_calculation.log'),
            logging.StreamHandler()
        ]
    )

def process_file(file_path, output_dir=None, num_processes=None):
    """
    Verarbeitet eine einzelne JSON-Datei mit Molekülsequenzen.
    
    Args:
        file_path: Pfad zur JSON-Datei
        output_dir: Ausgabeverzeichnis (optional)
        num_processes: Anzahl der zu verwendenden Prozesse (optional)
        
    Returns:
        True bei Erfolg, False bei Fehler
    """
    try:
        logging.info(f"Verarbeite Datei: {file_path}")
        
        # Bestimme Ausgabepfad
        if output_dir:
            # Erstelle Ausgabeverzeichnis, falls es nicht existiert
            os.makedirs(output_dir, exist_ok=True)
            
            # Behalte die Dateistruktur relativ zum Eingangsverzeichnis bei
            rel_path = os.path.basename(file_path)
            output_path = os.path.join(output_dir, rel_path)
        else:
            # Überschreibe die Originaldatei
            output_path = None
        
        # Konfiguration für die Eigenschaftsberechnung
        config = {
            "parallel_processing": True,
            "num_processes": num_processes if num_processes else max(1, multiprocessing.cpu_count() - 1),
            "batch_size": 1000
        }
        
        # Berechne Eigenschaften
        success = calculate_molecular_properties(file_path, output_path, config)
        
        if success:
            logging.info(f"Eigenschaften erfolgreich hinzugefügt: {file_path}")
        else:
            logging.error(f"Fehler bei der Verarbeitung von: {file_path}")
        
        return success
        
    except Exception as e:
        logging.error(f"Unerwarteter Fehler bei der Verarbeitung von {file_path}: {str(e)}")
        return False

def process_directory(input_dir, output_dir=None, recursive=False, num_processes=None):
    """
    Verarbeitet alle JSON-Dateien in einem Verzeichnis.
    
    Args:
        input_dir: Eingabeverzeichnis
        output_dir: Ausgabeverzeichnis (optional)
        recursive: Auch Unterverzeichnisse durchsuchen
        num_processes: Anzahl der zu verwendenden Prozesse (optional)
        
    Returns:
        Anzahl erfolgreich verarbeiteter Dateien
    """
    # Finde alle JSON-Dateien
    if recursive:
        search_pattern = os.path.join(input_dir, "**", "*.json")
        json_files = glob.glob(search_pattern, recursive=True)
    else:
        search_pattern = os.path.join(input_dir, "*.json")
        json_files = glob.glob(search_pattern)
    
    if not json_files:
        logging.warning(f"Keine JSON-Dateien in {input_dir} gefunden.")
        return 0
    
    logging.info(f"Gefunden: {len(json_files)} JSON-Dateien")
    
    # Verarbeite alle Dateien
    successful = 0
    for file_path in tqdm(json_files, desc="Dateien", unit="file"):
        # Bestimme Ausgabepfad wenn ein Ausgabeverzeichnis angegeben wurde
        if output_dir:
            # Behalte die Verzeichnisstruktur relativ zum Eingangsverzeichnis bei
            rel_path = os.path.relpath(file_path, input_dir)
            file_output_dir = os.path.dirname(os.path.join(output_dir, rel_path))
            
            # Erstelle Verzeichnis falls notwendig
            if file_output_dir:
                os.makedirs(file_output_dir, exist_ok=True)
                
            if process_file(file_path, file_output_dir, num_processes):
                successful += 1
        else:
            # Überschreibe die Originaldateien
            if process_file(file_path, None, num_processes):
                successful += 1
    
    return successful

def main():
    """Hauptfunktion"""
    # Parse Kommandozeilenargumente
    parser = argparse.ArgumentParser(description="Berechnet molekulare Eigenschaften für LatentMol-Sequenzdateien")
    parser.add_argument("--input", required=True, help="Eingabedatei oder -verzeichnis")
    parser.add_argument("--output", help="Ausgabedatei oder -verzeichnis (optional)")
    parser.add_argument("--recursive", action="store_true", help="Unterverzeichnisse rekursiv durchsuchen")
    parser.add_argument("--processes", type=int, help="Anzahl der zu verwendenden Prozesse")
    parser.add_argument("--verbose", action="store_true", help="Ausführlichere Ausgabe")
    
    args = parser.parse_args()
    
    # Logging einrichten
    setup_logging(args.verbose)
    
    # Prüfe ob der Eingabepfad existiert
    if not os.path.exists(args.input):
        logging.error(f"Eingabepfad existiert nicht: {args.input}")
        return 1
    
    # Anzahl der zu verwendenden Prozesse
    num_processes = args.processes
    if not num_processes:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        logging.info(f"Verwende {num_processes} Prozesse")
    
    # Verarbeite je nach Eingabetyp
    if os.path.isfile(args.input):
        # Verarbeite einzelne Datei
        if not args.input.endswith('.json'):
            logging.error(f"Eingabedatei muss eine JSON-Datei sein: {args.input}")
            return 1
            
        output_dir = os.path.dirname(args.output) if args.output else None
        success = process_file(args.input, output_dir, num_processes)
        
        return 0 if success else 1
    else:
        # Verarbeite Verzeichnis
        successful = process_directory(args.input, args.output, args.recursive, num_processes)
        
        logging.info(f"Verarbeitung abgeschlossen. {successful} Dateien erfolgreich verarbeitet.")
        return 0 if successful > 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 