#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Schnellere Analyse eines Samples der Moleküle.
"""

import os
import sys
import random
import logging
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
import matplotlib.pyplot as plt

# Konfiguriere Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_atom_count(file_path, count_hydrogens=False):
    """Extrahiert die Anzahl der Atome aus einer Moleküldatei."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Versuche zuerst aus dem Header zu extrahieren
        lines = content.split('\n')
        for line in lines:
            if line.strip().endswith('V2000'):
                atom_count_str = line.strip()[:3]
                try:
                    atom_count = int(atom_count_str)
                    if count_hydrogens:
                        # Versuche mit RDKit zu parsen und Wasserstoffatome hinzuzufügen
                        mol = Chem.MolFromMolBlock(content)
                        if mol is not None:
                            mol_with_h = Chem.AddHs(mol)
                            return mol_with_h.GetNumAtoms()
                    return atom_count
                except ValueError:
                    pass
        
        # Fallback: Versuche mit RDKit
        mol = Chem.MolFromMolBlock(content)
        if mol is not None:
            if count_hydrogens:
                mol = Chem.AddHs(mol)
            return mol.GetNumAtoms()
            
        logger.warning(f"Konnte Atomanzahl für {file_path} nicht ermitteln")
        return None
    except Exception as e:
        logger.error(f"Fehler bei {file_path}: {e}")
        return None

def main():
    """Hauptfunktion"""
    input_dir = "data/temp/split_db"
    sample_size = 1000  # Anzahl der zu analysierenden Moleküle
    
    # Prüfe, ob das Verzeichnis existiert
    if not os.path.exists(input_dir):
        logger.error(f"Verzeichnis {input_dir} existiert nicht")
        return 1
    
    # Finde alle Moleküldateien
    all_files = list(Path(input_dir).glob("*.txt"))
    if not all_files:
        logger.error(f"Keine Moleküldateien in {input_dir} gefunden")
        return 1
    
    # Nimm eine zufällige Stichprobe
    sample_size = min(sample_size, len(all_files))
    sample_files = random.sample(all_files, sample_size)
    
    logger.info(f"Analysiere {sample_size} von {len(all_files)} Molekülen...")
    
    # Statistiken ohne Wasserstoffatome
    lengths_no_h = {}
    # Statistiken mit Wasserstoffatomen
    lengths_with_h = {}
    
    # Verarbeite die Stichprobe
    for file_path in tqdm(sample_files, desc="Analyse"):
        # Atomanzahl ohne Wasserstoffatome
        atom_count = extract_atom_count(file_path, count_hydrogens=False)
        if atom_count is not None:
            lengths_no_h[atom_count] = lengths_no_h.get(atom_count, 0) + 1
        
        # Atomanzahl mit Wasserstoffatomen
        atom_count_h = extract_atom_count(file_path, count_hydrogens=True)
        if atom_count_h is not None:
            lengths_with_h[atom_count_h] = lengths_with_h.get(atom_count_h, 0) + 1
    
    # Erstelle Histogramme
    os.makedirs("molecule_analysis", exist_ok=True)
    
    # Ohne Wasserstoffatome
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    lengths = sorted(lengths_no_h.keys())
    counts = [lengths_no_h[length] for length in lengths]
    ax1.bar(lengths, counts, color='blue', alpha=0.7)
    ax1.set_title("Längenverteilung (ohne H-Atome)")
    ax1.set_xlabel("Anzahl der Atome")
    ax1.set_ylabel("Anzahl der Moleküle")
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("molecule_analysis/quick_no_h.png")
    
    # Mit Wasserstoffatomen
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    lengths = sorted(lengths_with_h.keys())
    counts = [lengths_with_h[length] for length in lengths]
    ax2.bar(lengths, counts, color='green', alpha=0.7)
    ax2.set_title("Längenverteilung (mit H-Atomen)")
    ax2.set_xlabel("Anzahl der Atome")
    ax2.set_ylabel("Anzahl der Moleküle")
    ax2.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("molecule_analysis/quick_with_h.png")
    
    # Statistiken ausgeben
    logger.info("--- Statistik ohne Wasserstoffatome ---")
    logger.info(f"Anzahl unterschiedlicher Längen: {len(lengths_no_h)}")
    logger.info(f"Minimale Länge: {min(lengths_no_h.keys())}")
    logger.info(f"Maximale Länge: {max(lengths_no_h.keys())}")
    logger.info(f"Länge 9: {lengths_no_h.get(9, 0)} Moleküle")
    
    logger.info("\n--- Statistik mit Wasserstoffatomen ---")
    logger.info(f"Anzahl unterschiedlicher Längen: {len(lengths_with_h)}")
    logger.info(f"Minimale Länge: {min(lengths_with_h.keys())}")
    logger.info(f"Maximale Länge: {max(lengths_with_h.keys())}")
    logger.info(f"Länge 9: {lengths_with_h.get(9, 0)} Moleküle")
    
    # Top 10 häufigste Längen
    logger.info("\n--- Top 10 häufigste Längen (ohne H) ---")
    top_lengths = sorted(lengths_no_h.items(), key=lambda x: x[1], reverse=True)[:10]
    for length, count in top_lengths:
        logger.info(f"Länge {length}: {count} Moleküle ({count/sample_size*100:.1f}%)")
    
    logger.info("\n--- Top 10 häufigste Längen (mit H) ---")
    top_lengths = sorted(lengths_with_h.items(), key=lambda x: x[1], reverse=True)[:10]
    for length, count in top_lengths:
        logger.info(f"Länge {length}: {count} Moleküle ({count/sample_size*100:.1f}%)")
    
    # Für MOL_MAX_LENGTH=10, wie viele Moleküle würden passen?
    max_length = 10
    count_no_h = sum(lengths_no_h.get(length, 0) for length in range(1, max_length + 1))
    percent_no_h = count_no_h / sample_size * 100
    
    logger.info(f"\nMoleküle mit Länge <= {max_length} (ohne H): {count_no_h}/{sample_size} ({percent_no_h:.1f}%)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 