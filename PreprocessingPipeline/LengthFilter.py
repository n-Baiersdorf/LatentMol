import os
import shutil
from pathlib import Path
from typing import Union
from tqdm import tqdm

class LengthFilterer:
    def __init__(self, max_length: int):
        """
        Initialisiert den LengthSorter.

        Args:
            max_length (int): Maximale Anzahl der Atome für die zu kopierenden Dateien.
        """
        self.max_length = max_length

    def filter_and_copy(self, input_directory: Union[str, Path], output_directory: Union[str, Path]) -> None:
        """
        Filtert Dateien aus dem Eingabeverzeichnis und kopiert sie in das Ausgabeverzeichnis,
        wenn ihre Atomanzahl den Schwellenwert nicht überschreitet.

        Args:
            input_directory: Pfad zum Verzeichnis mit den zu filternden Dateien
            output_directory: Pfad zum Verzeichnis für die gefilterten Dateien

        Raises:
            FileNotFoundError: Wenn das Eingabeverzeichnis nicht existiert
            PermissionError: Wenn keine Schreibrechte für das Ausgabeverzeichnis vorliegen
        """
        input_path = Path(input_directory)
        output_path = Path(output_directory)

        if not input_path.exists():
            raise FileNotFoundError(f"Eingabeverzeichnis nicht gefunden: {input_directory}")

        self.create_directory(output_path)

        total_files = sum(1 for _ in input_path.glob("*.txt"))

        for file_path in tqdm(input_path.glob("*.txt"), total=total_files, desc="Verarbeite Dateien"):
            try:
                atom_count = self.extract_atom_count(file_path)
                if atom_count <= self.max_length:
                    shutil.copy2(file_path, output_path / file_path.name)
            except Exception as e:
                tqdm.write(f"Fehler bei der Verarbeitung von {file_path}: {str(e)}")
                continue

    def extract_atom_count(self, file_path: Union[str, Path]) -> int:
        """
        Extrahiert die Anzahl der Atome aus einer MDL-formatierten Datei.

        Args:
            file_path: Pfad zur zu analysierenden Datei

        Returns:
            int: Die Anzahl der Atome im Molekül

        Raises:
            ValueError: Wenn keine gültige MDL-Tabelle gefunden wird
            FileNotFoundError: Wenn die Datei nicht existiert
        """
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    if line.strip().endswith('V2000'):
                        atom_count_str = line.strip()[:3]
                        try:
                            return int(atom_count_str)
                        except ValueError:
                            raise ValueError(f"Ungültiges Format der Counts-Zeile in {file_path}")
                raise ValueError(f"Keine gültige MDL-Tabelle in {file_path} gefunden")
        except FileNotFoundError:
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
        except Exception as e:
            raise Exception(f"Fehler beim Lesen der Datei {file_path}: {str(e)}")

    def create_directory(self, directory_path: Union[str, Path]) -> None:
        """
        Erstellt ein Verzeichnis, falls es noch nicht existiert.

        Args:
            directory_path: Pfad zum zu erstellenden Verzeichnis

        Raises:
            PermissionError: Wenn keine ausreichenden Rechte zum Erstellen des Verzeichnisses vorliegen
        """
        try:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise PermissionError(f"Keine Berechtigung zum Erstellen des Verzeichnisses: {directory_path}")



# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.