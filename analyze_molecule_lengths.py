#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dieses Skript analysiert die Längenverteilung von Molekülen in einem Verzeichnis.
Es zählt die Anzahl der Atome mit und ohne Wasserstoff und erstellt Statistiken.
"""

import os
import sys
import glob
import logging
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
import matplotlib.pyplot as plt
import numpy as np

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('molecule_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def extract_atom_count(content, count_hydrogens=False):
    """
    Extrahiert die Anzahl der Atome aus einer MDL-formatierten Datei.
    
    Args:
        content: Der Inhalt der Moleküldatei
        count_hydrogens: Ob Wasserstoffatome mitgezählt werden sollen
        
    Returns:
        Anzahl der Atome oder None bei Fehler
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
                    if count_hydrogens:
                        try:
                            mol = Chem.MolFromMolBlock(content)
                            if mol is not None:
                                # AddHs fügt explizite Wasserstoffatome hinzu
                                mol_with_h = Chem.AddHs(mol)
                                return mol_with_h.GetNumAtoms()
                            # Fallback zur direkten Zählung, wenn RDKit fehlschlägt
                            return atom_count
                        except:
                            logger.warning("RDKit konnte Molekül nicht parsen, verwende direkte Zählung")
                            return atom_count
                    else:
                        return atom_count
                except ValueError:
                    logger.error("Ungültiges Format der Counts-Zeile")
                    return None
        
        # Wenn wir hier ankommen, versuchen wir es mit RDKit
        mol = Chem.MolFromMolBlock(content)
        if mol is not None:
            if count_hydrogens:
                mol = Chem.AddHs(mol)
            return mol.GetNumAtoms()
            
        logger.error("Keine gültige MDL-Tabelle gefunden")
        return None
    except Exception as e:
        logger.error(f"Fehler beim Extrahieren der Atomanzahl: {str(e)}")
        return None

def analyze_directory(directory, target_length=None, count_hydrogens=False):
    """
    Analysiert alle Moleküldateien in einem Verzeichnis.
    
    Args:
        directory: Das zu analysierende Verzeichnis
        target_length: Spezifische Länge, die gesucht wird (optional)
        count_hydrogens: Ob Wasserstoffatome mitgezählt werden sollen
        
    Returns:
        Dictionary mit Statistiken
    """
    stats = {
        "total_files": 0,
        "valid_files": 0,
        "invalid_files": 0,
        "length_distribution": {},
        "min_length": float('inf'),
        "max_length": 0,
        "avg_length": 0,
        "matching_target": 0 if target_length else None
    }
    
    # Finde alle Moleküldateien
    mol_files = []
    for ext in ["*.txt", "*.mol", "*.sdf"]:
        mol_files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    
    if not mol_files:
        logger.warning(f"Keine Moleküldateien in {directory} gefunden.")
        return stats
    
    stats["total_files"] = len(mol_files)
    logger.info(f"Analysiere {len(mol_files)} Moleküldateien in {directory}...")
    
    # Analysiere die Dateien
    sum_lengths = 0
    for mol_file in tqdm(mol_files, desc="Analyse"):
        try:
            with open(mol_file, 'r') as f:
                content = f.read()
            
            atom_count = extract_atom_count(content, count_hydrogens)
            if atom_count is not None:
                stats["valid_files"] += 1
                
                # Aktualisiere Längenverteilung
                if atom_count not in stats["length_distribution"]:
                    stats["length_distribution"][atom_count] = 0
                stats["length_distribution"][atom_count] += 1
                
                # Aktualisiere Min/Max/Durchschnitt
                stats["min_length"] = min(stats["min_length"], atom_count)
                stats["max_length"] = max(stats["max_length"], atom_count)
                sum_lengths += atom_count
                
                # Prüfe auf Ziellänge
                if target_length is not None and atom_count == target_length:
                    stats["matching_target"] += 1
            else:
                stats["invalid_files"] += 1
        except Exception as e:
            logger.error(f"Fehler bei {mol_file}: {e}")
            stats["invalid_files"] += 1
    
    # Berechne Durchschnitt
    if stats["valid_files"] > 0:
        stats["avg_length"] = sum_lengths / stats["valid_files"]
    else:
        stats["min_length"] = 0  # Reset min_length wenn keine gültigen Dateien
    
    return stats

def plot_distribution(stats, title, output_file):
    """
    Erstellt ein Histogramm der Längenverteilung und speichert es.
    
    Args:
        stats: Statistik-Dictionary mit length_distribution
        title: Titel des Diagramms
        output_file: Pfad, unter dem das Diagramm gespeichert werden soll
    """
    # Extrahiere Längen und Häufigkeiten
    lengths = list(stats["length_distribution"].keys())
    counts = list(stats["length_distribution"].values())
    
    # Sortiere die Daten
    sorted_data = sorted(zip(lengths, counts))
    lengths, counts = zip(*sorted_data) if sorted_data else ([], [])
    
    # Erstelle das Histogramm
    plt.figure(figsize=(12, 6))
    plt.bar(lengths, counts, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel("Anzahl der Atome")
    plt.ylabel("Anzahl der Moleküle")
    
    # Begrenze den X-Achsenbereich auf einen sinnvollen Bereich
    max_plot_length = min(max(lengths) + 5, 100)  # Beschränke auf max. 100 für bessere Lesbarkeit
    plt.xlim(-1, max_plot_length)
    
    # Füge Gitterlinien hinzu
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Speichere das Diagramm
    plt.tight_layout()
    plt.savefig(output_file)
    logger.info(f"Diagramm gespeichert unter {output_file}")

def main():
    """Hauptfunktion"""
    if len(sys.argv) < 2:
        print("Verwendung: python analyze_molecule_lengths.py <Verzeichnis> [Ziellänge]")
        return 1
    
    directory = sys.argv[1]
    target_length = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Überprüfe, ob das Verzeichnis existiert
    if not os.path.exists(directory):
        logger.error(f"Verzeichnis {directory} existiert nicht.")
        return 1
    
    # Erstelle Ausgabeverzeichnis für Diagramme
    output_dir = "molecule_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Analysiere Moleküle ohne Wasserstoffatome
    logger.info("Analysiere Moleküle ohne Wasserstoffatome...")
    stats_no_h = analyze_directory(directory, target_length, count_hydrogens=False)
    
    # Analysiere Moleküle mit Wasserstoffatomen
    logger.info("Analysiere Moleküle mit Wasserstoffatomen...")
    stats_with_h = analyze_directory(directory, target_length, count_hydrogens=True)
    
    # Zeige die Ergebnisse an
    logger.info("\n--- Analyse ohne Wasserstoffatome ---")
    logger.info(f"Anzahl der Dateien: {stats_no_h['total_files']}")
    logger.info(f"Gültige Moleküle: {stats_no_h['valid_files']}")
    logger.info(f"Ungültige Moleküle: {stats_no_h['invalid_files']}")
    logger.info(f"Längenbereich: {stats_no_h['min_length']} bis {stats_no_h['max_length']} Atome")
    logger.info(f"Durchschnittliche Länge: {stats_no_h['avg_length']:.2f} Atome")
    
    if target_length is not None:
        logger.info(f"Moleküle mit genau {target_length} Atomen: {stats_no_h['matching_target']}")
    
    logger.info("\n--- Analyse mit Wasserstoffatomen ---")
    logger.info(f"Anzahl der Dateien: {stats_with_h['total_files']}")
    logger.info(f"Gültige Moleküle: {stats_with_h['valid_files']}")
    logger.info(f"Ungültige Moleküle: {stats_with_h['invalid_files']}")
    logger.info(f"Längenbereich: {stats_with_h['min_length']} bis {stats_with_h['max_length']} Atome")
    logger.info(f"Durchschnittliche Länge: {stats_with_h['avg_length']:.2f} Atome")
    
    if target_length is not None:
        logger.info(f"Moleküle mit genau {target_length} Atomen (inkl. H): {stats_with_h['matching_target']}")
    
    # Erstelle Verteilungsdiagramme
    plot_distribution(
        stats_no_h, 
        f"Längenverteilung der Moleküle (ohne H-Atome)", 
        os.path.join(output_dir, "length_distribution_no_h.png")
    )
    
    plot_distribution(
        stats_with_h, 
        f"Längenverteilung der Moleküle (mit H-Atomen)", 
        os.path.join(output_dir, "length_distribution_with_h.png")
    )
    
    # Erstelle die Top 10 häufigsten Längen
    logger.info("\n--- Top 10 häufigste Längen (ohne H) ---")
    top_lengths = sorted(stats_no_h["length_distribution"].items(), key=lambda x: x[1], reverse=True)[:10]
    for length, count in top_lengths:
        logger.info(f"Länge {length}: {count} Moleküle")
    
    logger.info("\n--- Top 10 häufigste Längen (mit H) ---")
    top_lengths = sorted(stats_with_h["length_distribution"].items(), key=lambda x: x[1], reverse=True)[:10]
    for length, count in top_lengths:
        logger.info(f"Länge {length}: {count} Moleküle")
    
    logger.info("Analyse abgeschlossen.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 