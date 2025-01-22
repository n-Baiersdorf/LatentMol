import os
import shutil
from pathlib import Path
from typing import Union, Optional
from tqdm import tqdm

class MoleculeFileSorter:
    """
    Eine Klasse zum Sortieren von MDL-formatierten Moleküldateien nach ihrer Atomanzahl.
    
    Die Klasse verarbeitet .txt Dateien, die MDL V2000 formatierte Molekültabellen enthalten,
    und sortiert sie in Verzeichnisse basierend auf der Anzahl ihrer Atome.
    """
    
    def __init__(self):
        """Initialisiert den MoleculeFileSorter."""
        pass

    def sort_files(self, input_directory: Union[str, Path], output_base_directory: Union[str, Path]) -> None:
        """
        Sortiert alle .txt Dateien aus dem Eingabeverzeichnis in Unterverzeichnisse nach Atomanzahl.

        Args:
            input_directory: Pfad zum Verzeichnis mit den zu sortierenden Dateien
            output_base_directory: Pfad zum Basisverzeichnis für die sortierten Dateien

        Raises:
            FileNotFoundError: Wenn das Eingabeverzeichnis nicht existiert
            PermissionError: Wenn keine Schreibrechte für das Ausgabeverzeichnis vorliegen
        """
        input_path = Path(input_directory)
        output_base_path = Path(output_base_directory)

        if not input_path.exists():
            raise FileNotFoundError(f"Eingabeverzeichnis nicht gefunden: {input_directory}")

        # Zähle die Anzahl der zu verarbeitenden Dateien
        total_files = sum(1 for _ in input_path.glob("*.txt"))

        # Durchsuche alle .txt Dateien im Eingabeverzeichnis mit tqdm-Fortschrittsanzeige
        for file_path in tqdm(input_path.glob("*.txt"), total=total_files, desc="Sorting Molecules by length"):
            try:
                # Extrahiere die Atomanzahl aus der Datei
                atom_count = self.extract_atom_count(file_path)
                
                # Erstelle das Zielverzeichnis basierend auf der Atomanzahl
                target_directory = output_base_path / str(atom_count)
                self.create_directory(target_directory)
                
                # Kopiere die Datei in das entsprechende Verzeichnis
                shutil.copy2(file_path, target_directory / file_path.name)
                
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
                    # Suche nach der Counts-Zeile (endet mit V2000)
                    if line.strip().endswith('V2000'):
                        # Extrahiere die Atomanzahl aus den ersten drei Zeichen
                        atom_count_str = line.strip()[:3]
                        try:
                            return int(atom_count_str)
                        except ValueError:
                            raise ValueError(f"Ungültiges Format der Counts-Zeile in {file_path}")
                
                # Wenn keine V2000-Zeile gefunden wurde
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
        

if __name__ == "__main__":
    sorter = MoleculeFileSorter()
    try:
        sorter.sort_files("data/temp/with_Other_M", "data/test_sort")
    except Exception as e:
        print(f"Fehler bei der Verarbeitung: {str(e)}")

# Copyright (c) 2025 Noah Baiersdorf
# This software is released under the MIT License.