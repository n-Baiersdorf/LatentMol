#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Molekülaugmentierung und -verarbeitung
-------------------------------------

Dieses Skript zeigt, wie der MoleculeAugmenter zusammen mit MolToSequence
zur Erstellung von augmentierten Moleküldatensätzen verwendet werden kann.
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import multiprocessing
import logging
from typing import List, Dict, Optional

# Füge das Projektverzeichnis zum Python-Pfad hinzu
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from PreprocessingPipeline.MoleculeAugmenter import augment_molecules
    from PreprocessingPipeline.MolToSequence import molToSequenceFunction
except ImportError as e:
    print(f"Import-Fehler: {e}")
    print("Bitte stellen Sie sicher, dass das Projekt korrekt installiert ist.")
    sys.exit(1)

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AugmentAndProcess")


def augment_and_process_file(input_file: str, output_dir: str, augment_config: Optional[Dict] = None, sequence_config: Optional[Dict] = None) -> bool:
    """
    Augmentiert ein einzelnes Molekül und verarbeitet die augmentierten Moleküle mit MolToSequence.
    
    Args:
        input_file: Pfad zur Mol- oder SDF-Datei
        output_dir: Ausgabeverzeichnis für die Sequenzen
        augment_config: Konfiguration für die Augmentierung
        sequence_config: Konfiguration für MolToSequence
        
    Returns:
        True, wenn die Verarbeitung erfolgreich war, sonst False
    """
    try:
        # Verzeichnisstruktur erstellen
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        augment_dir = os.path.join(output_dir, f"augmented_{base_name}")
        sequence_dir = os.path.join(output_dir, f"sequences_{base_name}")
        
        os.makedirs(augment_dir, exist_ok=True)
        os.makedirs(sequence_dir, exist_ok=True)
        
        # Augmentierung
        logger.info(f"Augmentiere {input_file}...")
        augmented_files = augment_molecules(input_file, augment_dir, augment_config)
        
        if not augmented_files:
            logger.warning(f"Keine augmentierten Moleküle für {input_file} erzeugt!")
            return False
            
        logger.info(f"Erzeugte {len(augmented_files)} augmentierte Moleküle")
        
        # Verarbeite jedes augmentierte Molekül mit MolToSequence
        success_count = 0
        for i, aug_file in enumerate(augmented_files):
            aug_name = os.path.splitext(os.path.basename(aug_file))[0]
            logger.info(f"Verarbeite augmentiertes Molekül {i+1}/{len(augmented_files)}: {aug_name}")
            
            success = molToSequenceFunction(aug_file, sequence_dir, i+1, sequence_config)
            if success:
                success_count += 1
                
        logger.info(f"Erfolgreich verarbeitet: {success_count}/{len(augmented_files)} augmentierte Moleküle")
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Fehler bei Augmentierung/Verarbeitung von {input_file}: {e}")
        return False


def process_directory(input_dir: str, output_dir: str, file_pattern: str = "*.mol",
                     augment_config: Optional[Dict] = None, sequence_config: Optional[Dict] = None,
                     max_workers: int = 0) -> int:
    """
    Verarbeitet alle Dateien in einem Verzeichnis.
    
    Args:
        input_dir: Eingabeverzeichnis
        output_dir: Ausgabeverzeichnis
        file_pattern: Glob-Muster für zu verarbeitende Dateien
        augment_config: Konfiguration für die Augmentierung
        sequence_config: Konfiguration für MolToSequence
        max_workers: Max. Anzahl paralleler Prozesse (0 = CPU-Zahl)
        
    Returns:
        Anzahl erfolgreich verarbeiteter Dateien
    """
    if not os.path.exists(input_dir):
        logger.error(f"Eingabeverzeichnis {input_dir} existiert nicht!")
        return 0
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Finde alle Dateien
    file_pattern_path = os.path.join(input_dir, file_pattern)
    input_files = glob.glob(file_pattern_path)
    
    if not input_files:
        logger.warning(f"Keine Dateien gefunden, die auf {file_pattern_path} passen!")
        return 0
        
    logger.info(f"Gefunden: {len(input_files)} Dateien zum Verarbeiten")
    
    # Mehrere Dateien parallel verarbeiten
    if max_workers <= 0:
        max_workers = multiprocessing.cpu_count()
        
    logger.info(f"Starte Verarbeitung mit {max_workers} parallelen Prozessen")
    
    # Bei nur einer CPU, verarbeite sequentiell
    if max_workers == 1:
        success_count = 0
        for i, input_file in enumerate(input_files):
            logger.info(f"Verarbeite Datei {i+1}/{len(input_files)}: {os.path.basename(input_file)}")
            if augment_and_process_file(input_file, output_dir, augment_config, sequence_config):
                success_count += 1
        return success_count
    
    # Mit mehreren CPUs, nutze ProcessPoolExecutor
    try:
        from concurrent.futures import ProcessPoolExecutor
        
        success_count = 0
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Bereite Aufgaben vor
            futures = []
            for input_file in input_files:
                future = executor.submit(
                    augment_and_process_file,
                    input_file,
                    output_dir,
                    augment_config,
                    sequence_config
                )
                futures.append((input_file, future))
            
            # Sammle Ergebnisse
            for i, (input_file, future) in enumerate(futures):
                try:
                    result = future.result()
                    logger.info(f"Datei {i+1}/{len(input_files)}: {os.path.basename(input_file)} - {'Erfolgreich' if result else 'Fehlgeschlagen'}")
                    if result:
                        success_count += 1
                except Exception as e:
                    logger.error(f"Fehler bei Datei {os.path.basename(input_file)}: {e}")
                    
        return success_count
    
    except ImportError:
        # Fallback: Sequentielle Verarbeitung
        logger.warning("concurrent.futures nicht verfügbar, verwende sequentielle Verarbeitung")
        success_count = 0
        for i, input_file in enumerate(input_files):
            logger.info(f"Verarbeite Datei {i+1}/{len(input_files)}: {os.path.basename(input_file)}")
            if augment_and_process_file(input_file, output_dir, augment_config, sequence_config):
                success_count += 1
        return success_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Molekülaugmentierung und -verarbeitung")
    parser.add_argument("input", help="Eingabedatei oder -verzeichnis")
    parser.add_argument("output_dir", help="Ausgabeverzeichnis")
    parser.add_argument("--pattern", default="*.mol", help="Dateimuster für Verzeichnisverarbeitung")
    
    # Augmentierungsparameter
    aug_group = parser.add_argument_group("Augmentierung")
    aug_group.add_argument("--num_conformers", type=int, default=3, help="Anzahl zu generierender Konformere pro Molekül")
    aug_group.add_argument("--perturbation", type=float, default=0.1, help="Stärke der geometrischen Störungen")
    aug_group.add_argument("--num_rotations", type=int, default=2, help="Anzahl der Rotationsvarianten pro Molekül")
    aug_group.add_argument("--no_perturbation", action="store_true", help="Keine geometrischen Störungen anwenden")
    aug_group.add_argument("--no_rotation", action="store_true", help="Keine Rotationen anwenden")
    
    # MolToSequence-Parameter
    seq_group = parser.add_argument_group("MolToSequence")
    seq_group.add_argument("--coords_min", type=float, default=0.01, help="Minimaler Wert für Koordinatennormalisierung")
    seq_group.add_argument("--coords_max", type=float, default=0.2, help="Maximaler Wert für Koordinatennormalisierung")
    seq_group.add_argument("--use_01_coords", action="store_true", help="Koordinaten auf [0,1] normalisieren")
    
    # Allgemeine Parameter
    parser.add_argument("--workers", type=int, default=0, help="Anzahl paralleler Prozesse (0 = CPU-Anzahl)")
    parser.add_argument("--debug", action="store_true", help="Debug-Modus mit detaillierter Ausgabe")
    
    args = parser.parse_args()
    
    # Konfiguriere Logging basierend auf Debug-Modus
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Augmentierungskonfiguration erstellen
    augment_config = {
        "num_conformers": args.num_conformers,
        "use_geometric_perturbation": not args.no_perturbation,
        "perturbation_magnitude": args.perturbation,
        "use_rotation_translation": not args.no_rotation,
        "num_rotation_samples": args.num_rotations
    }
    
    # MolToSequence-Konfiguration erstellen
    sequence_config = {
        "normalization": {
            "coords": {
                "min": args.coords_min,
                "max": args.coords_max,
                "use_01": args.use_01_coords
            }
        }
    }
    
    # Überprüfe, ob Eingabe eine Datei oder ein Verzeichnis ist
    if os.path.isfile(args.input):
        logger.info(f"Verarbeite einzelne Datei: {args.input}")
        success = augment_and_process_file(args.input, args.output_dir, augment_config, sequence_config)
        sys.exit(0 if success else 1)
    elif os.path.isdir(args.input):
        logger.info(f"Verarbeite Verzeichnis: {args.input}")
        success_count = process_directory(
            args.input, 
            args.output_dir, 
            args.pattern,
            augment_config, 
            sequence_config,
            args.workers
        )
        logger.info(f"Verarbeitung abgeschlossen. {success_count} Dateien erfolgreich verarbeitet.")
        sys.exit(0 if success_count > 0 else 1)
    else:
        logger.error(f"Eingabe {args.input} ist weder eine Datei noch ein Verzeichnis!")
        sys.exit(1) 