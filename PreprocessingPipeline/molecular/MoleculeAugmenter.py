import os
import json
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import Geometry  # Korrekter Import für Geometry
import logging
from pathlib import Path
import sys
from scipy.spatial.transform import Rotation
import copy
import tempfile

# Füge das Projektverzeichnis zum Python-Pfad hinzu
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from PreprocessingPipeline.filters.base_processor import BaseProcessor, ProcessingConfig
except ImportError as e:
    print(f"Import-Fehler: {e}")
    print("Bitte stellen Sie sicher, dass das Projekt korrekt installiert ist.")
    sys.exit(1)


class MoleculeAugmentationConfig(ProcessingConfig):
    """Konfiguration für die Molekülaugmentierung"""
    def __init__(self, 
                num_conformers: int = 3,
                use_geometric_perturbation: bool = True,
                perturbation_magnitude: float = 0.1,
                use_rotation_translation: bool = True,
                num_rotation_samples: int = 2,
                **kwargs):
        """
        Initialisiert die Konfiguration für die Molekülaugmentierung.
        
        Args:
            num_conformers: Anzahl der zu generierenden Konformere pro Molekül
            use_geometric_perturbation: Ob geometrische Störungen angewendet werden sollen
            perturbation_magnitude: Stärke der geometrischen Störungen
            use_rotation_translation: Ob Rotationen und Translationen angewendet werden sollen
            num_rotation_samples: Anzahl der Rotationsvarianten pro Molekül
            **kwargs: Zusätzliche Parameter für die Basiskonfiguration
        """
        super().__init__(**kwargs)
        self.num_conformers = num_conformers
        self.use_geometric_perturbation = use_geometric_perturbation
        self.perturbation_magnitude = perturbation_magnitude
        self.use_rotation_translation = use_rotation_translation
        self.num_rotation_samples = num_rotation_samples


class MoleculeAugmenter(BaseProcessor):
    """
    Klasse zur Augmentierung von Molekülen mit verschiedenen Methoden:
    1. 3D-Konformergeneration
    2. Geometrische Störungen (kleine Koordinatenänderungen)
    3. Rotation und Translation
    
    Diese Klasse nimmt .mol oder .sdf Dateien als Eingabe und erzeugt 
    augmentierte Versionen dieser Moleküle, die dann mit MolToSequence
    weiterverarbeitet werden können.
    """
    
    def __init__(self, config: Optional[Union[MoleculeAugmentationConfig, Dict]] = None):
        """Initialisiert den Augmentierer mit der gegebenen Konfiguration."""
        # Extrahiere selected_variant_indices vor der Übergabe an den Basisklassenkonstruktor
        self.selected_variant_indices = None
        if isinstance(config, dict) and 'selected_variant_indices' in config:
            self.selected_variant_indices = config['selected_variant_indices']
            # Entferne den Parameter aus der Konfiguration
            config_copy = config.copy()
            if 'selected_variant_indices' in config_copy:
                del config_copy['selected_variant_indices']
            config = config_copy
            
        if isinstance(config, dict):
            config = MoleculeAugmentationConfig(**config)
        elif config is None:
            config = MoleculeAugmentationConfig()
            
        super().__init__(config)
        self.config = config
        self.temp_dir = tempfile.mkdtemp(prefix="mol_augmentation_")
        self.logger.info(f"Temporäres Verzeichnis für Augmentierung: {self.temp_dir}")
        
        # Statistiken für die Augmentierung
        self._update_statistics("augmentation_stats", {
            "total_input_molecules": 0,
            "total_generated_molecules": 0,
            "conformers_generated": 0,
            "perturbations_applied": 0,
            "rotations_applied": 0,
            "errors": {
                "conformer_generation": 0,
                "perturbation": 0,
                "rotation": 0
            }
        })
        
    def augment_mol_file(self, input_file: str, output_dir: str) -> List[str]:
        """
        Augmentiert Moleküle aus einer .mol oder .sdf Datei und speichert
        die resultierenden Moleküle im output_dir.
        
        Args:
            input_file: Pfad zur Eingabedatei (.mol oder .sdf)
            output_dir: Verzeichnis zum Speichern der augmentierten Moleküle
            
        Returns:
            Liste der Pfade zu den augmentierten Moleküldateien
        """
        if not os.path.exists(input_file):
            self.logger.error(f"Eingabedatei {input_file} existiert nicht!")
            return []
            
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        output_files = []
        
        try:
            # Lade Moleküle aus der Eingabedatei
            mols = self._load_molecules(input_file)
            if not mols:
                self.logger.warning(f"Keine Moleküle in {input_file} gefunden!")
                return []
                
            self._update_statistics("augmentation_stats", {
                "total_input_molecules": len(mols)
            })
            
            # Augmentiere jedes Molekül
            for i, (mol_id, mol) in enumerate(mols):
                self.logger.info(f"Augmentiere Molekül {mol_id} ({i+1}/{len(mols)})")
                augmented_mols = self._augment_molecule(mol, self.selected_variant_indices)
                
                # Speichere augmentierte Moleküle
                for j, aug_mol in enumerate(augmented_mols):
                    if aug_mol is None:
                        continue
                        
                    # Stelle sicher, dass alle Wasserstoffatome explizit sind
                    aug_mol = Chem.AddHs(aug_mol)
                        
                    # Erstelle Ausgabedatei
                    output_file = os.path.join(output_dir, f"{mol_id}_aug_{j}.mol")
                    
                    # Speichere Molekül mit allen Wasserstoffatomen
                    mol_block = Chem.MolToMolBlock(aug_mol, kekulize=True)
                    with open(output_file, 'w') as f:
                        f.write(mol_block)
                        
                    output_files.append(output_file)
                    
            self._update_statistics("augmentation_stats", {
                "total_generated_molecules": len(output_files)
            })
                    
            return output_files
            
        except Exception as e:
            self.logger.error(f"Fehler bei der Augmentierung von {input_file}: {str(e)}")
            return []
    
    def _load_molecules(self, input_file: str) -> List[Tuple[str, Chem.Mol]]:
        """Lädt Moleküle aus einer Datei und gibt sie als Liste zurück."""
        mols = []
        
        try:
            # Prüfe, ob es sich um eine SDF-Datei handelt
            if input_file.lower().endswith('.sdf'):
                supplier = Chem.SDMolSupplier(input_file, removeHs=False)  # Keine Wasserstoffe entfernen
                for i, mol in enumerate(supplier):
                    if mol is not None:
                        # Stelle sicher, dass alle Wasserstoffatome vorhanden sind
                        if not mol.GetNumAtoms() >= 3:  # Minimale Größe für Tests
                            mol = Chem.AddHs(mol)
                        mol_id = f"mol_{i+1}"
                        if mol.HasProp('_Name') and mol.GetProp('_Name'):
                            mol_id = mol.GetProp('_Name')
                        mols.append((mol_id, mol))
            else:
                # Versuche als einfache .mol-Datei zu laden ohne Wasserstoffe zu entfernen
                mol = Chem.MolFromMolFile(input_file, removeHs=False)
                if mol is not None:
                    # Stelle sicher, dass alle Wasserstoffatome vorhanden sind
                    if not mol.GetNumAtoms() >= 3:  # Minimale Größe für Tests
                        mol = Chem.AddHs(mol)
                    mol_id = os.path.splitext(os.path.basename(input_file))[0]
                    mols.append((mol_id, mol))
        except Exception as e:
            self.logger.error(f"Fehler beim Laden von {input_file}: {str(e)}")
            
        return mols
    
    def _augment_molecule(self, mol: Chem.Mol, selected_variant_indices=None) -> List[Chem.Mol]:
        """
        Wendet Augmentierungstechniken auf ein Molekül an.
        
        Reihenfolge:
        1. Erzeuge Konformere
        2. Wende geometrische Störungen auf jedes Konformer an
        3. Rotiere und verschiebe jedes gestörte Konformer
        
        Args:
            mol: Eingabemolekül
            selected_variant_indices: Liste von Indizes der zu erzeugenden Varianten
                                     (None = alle erzeugen)
        """
        augmented_mols = []
        all_variants = []  # Liste aller möglichen Varianten
        variant_index = 0   # Zähler für die aktuellen Varianten
        
        try:
            # Stelle sicher, dass Wasserstoffatome explizit sind
            mol = Chem.AddHs(mol)
            
            # 1. Konformergeneration
            conformers = self._generate_conformers(mol)
            
            for conf_mol in conformers:
                if conf_mol is None:
                    continue
                    
                # 2. Geometrische Störungen anwenden
                if self.config.use_geometric_perturbation:
                    perturbed_mols = self._apply_perturbations(conf_mol)
                else:
                    perturbed_mols = [conf_mol]
                    
                # 3. Rotationen und Translationen anwenden
                for pert_mol in perturbed_mols:
                    if pert_mol is None:
                        continue
                        
                    if self.config.use_rotation_translation:
                        rotated_mols = self._apply_rotations(pert_mol)
                        
                        # Füge jede rotierte Variante zur Liste aller Varianten hinzu
                        for rot_mol in rotated_mols:
                            # Wenn keine Auswahl angegeben oder diese Variante ausgewählt wurde
                            if selected_variant_indices is None or variant_index in selected_variant_indices:
                                augmented_mols.append(rot_mol)
                            variant_index += 1
                    else:
                        # Wenn keine Auswahl angegeben oder diese Variante ausgewählt wurde
                        if selected_variant_indices is None or variant_index in selected_variant_indices:
                            augmented_mols.append(pert_mol)
                        variant_index += 1
        
        except Exception as e:
            self.logger.error(f"Fehler bei der Augmentierung eines Moleküls: {str(e)}")
        
        return augmented_mols
    
    def _generate_conformers(self, mol: Chem.Mol) -> List[Chem.Mol]:
        """Generiert 3D-Konformere für ein Molekül."""
        conformers = []
        
        try:
            # Tiefe Kopie des Moleküls
            mol_copy = Chem.Mol(mol)
            
            # Stelle sicher, dass das Molekül 3D-Koordinaten hat
            if mol_copy.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol_copy, randomSeed=42)
            
            # Generiere mehrere Konformere mit stark unterschiedlichen Parametern
            for i in range(self.config.num_conformers):
                try:
                    # Erstelle ein neues Molekül für jedes Konformer
                    conf_mol = Chem.Mol(mol_copy)
                    # Entferne alle vorhandenen Konformere, um neu zu beginnen
                    conf_mol.RemoveAllConformers()
                    
                    # Verwende verschiedene Seeds für unterschiedliche Ergebnisse
                    seed = 42 + i*100  # Größerer Unterschied zwischen Seeds
                    
                    # Setze Parameter für die Konformergenerierung
                    params = AllChem.ETKDGv3()  # Neue Algorithmusversion
                    params.randomSeed = seed
                    params.useRandomCoords = True  # Beginne mit zufälligen Koordinaten
                    params.numThreads = 1  # Single-Thread für Reproduzierbarkeit
                    
                    # Generiere ein Konformer mit diesen Parametern
                    conf_id = AllChem.EmbedMolecule(conf_mol, params)
                    
                    if conf_id >= 0:  # Erfolgreiche Generierung
                        # Optimiere das Konformer
                        AllChem.MMFFOptimizeMolecule(conf_mol, confId=0, maxIters=1000)
                        
                        # Zusätzliche Störung, um Unterschiede zu verstärken
                        conf = conf_mol.GetConformer()
                        for atom_idx in range(conf_mol.GetNumAtoms()):
                            pos = conf.GetAtomPosition(atom_idx)
                            # Zufällige Störung, die vom Seed abhängt
                            np.random.seed(seed + atom_idx)
                            noise = np.random.normal(0, 0.2, 3)
                            new_pos = Geometry.Point3D(
                                pos.x + noise[0],
                                pos.y + noise[1],
                                pos.z + noise[2]
                            )
                            conf.SetAtomPosition(atom_idx, new_pos)
                        
                        conformers.append(conf_mol)
                    
                except Exception as e:
                    self.logger.warning(f"Fehler bei Konformeroptimierung: {str(e)}")
                    self.statistics["step_specific"]["augmentation_stats"]["errors"]["conformer_generation"] += 1
            
            self._update_statistics("augmentation_stats", {
                "conformers_generated": len(conformers)
            })
            
            # Falls keine Konformere generiert werden konnten
            if not conformers:
                # Fallback: Verwende das Originalmolekül mit einer neuen 3D-Konformation
                mol_copy = Chem.Mol(mol)
                params = AllChem.ETKDGv3()
                params.randomSeed = 42
                AllChem.EmbedMolecule(mol_copy, params)
                AllChem.MMFFOptimizeMolecule(mol_copy)
                conformers = [mol_copy]
                
        except Exception as e:
            self.logger.error(f"Fehler bei Konformergenerierung: {str(e)}")
            self.statistics["step_specific"]["augmentation_stats"]["errors"]["conformer_generation"] += 1
            
            # Fallback: Verwende das Originalmolekül
            try:
                mol_copy = Chem.Mol(mol)
                AllChem.EmbedMolecule(mol_copy, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol_copy)
                conformers = [mol_copy]
            except:
                # Wenn alles fehlschlägt, verwende das Originalmolekül ohne Änderungen
                conformers = [mol]
            
        return conformers
    
    def _apply_perturbations(self, mol: Chem.Mol) -> List[Chem.Mol]:
        """Wendet kleine zufällige Störungen auf die Atomkoordinaten an."""
        perturbed_mols = []
        
        try:
            # Tiefe Kopie des Moleküls
            mol_copy = Chem.Mol(mol)
            
            # Stelle sicher, dass das Molekül einen Konformer hat
            if mol_copy.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol_copy, randomSeed=42)
            
            conf = mol_copy.GetConformer()
            
            # Erzeuge ein perturbiertes Molekül
            perturbed_mol = Chem.Mol(mol_copy)
            perturbed_conf = perturbed_mol.GetConformer()
            
            # Wende zufällige Störungen auf jedes Atom an
            for i in range(perturbed_mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                
                # Generiere zufällige Störungen
                noise = np.random.normal(0, self.config.perturbation_magnitude, 3)
                
                # Setze neue Position
                new_pos = Geometry.Point3D(
                    pos.x + noise[0],
                    pos.y + noise[1],
                    pos.z + noise[2]
                )
                perturbed_conf.SetAtomPosition(i, new_pos)
            
            perturbed_mols.append(perturbed_mol)
            
            self._update_statistics("augmentation_stats", {
                "perturbations_applied": len(perturbed_mols)
            })
            
        except Exception as e:
            self.logger.error(f"Fehler bei geometrischer Störung: {str(e)}")
            self.statistics["step_specific"]["augmentation_stats"]["errors"]["perturbation"] += 1
            
            # Fallback: Verwende das Originalmolekül
            perturbed_mols = [mol]
            
        return perturbed_mols
    
    def _apply_rotations(self, mol: Chem.Mol) -> List[Chem.Mol]:
        """Wendet zufällige Rotationen und Translationen auf das Molekül an."""
        rotated_mols = []
        
        try:
            # Stelle sicher, dass das Molekül einen Konformer hat
            mol_copy = Chem.Mol(mol)
            if mol_copy.GetNumConformers() == 0:
                AllChem.EmbedMolecule(mol_copy, randomSeed=42)
            
            # Generiere mehrere rotierte Varianten
            for i in range(self.config.num_rotation_samples):
                # Tiefe Kopie des Moleküls für jede Rotation
                rot_mol = Chem.Mol(mol_copy)
                conf = rot_mol.GetConformer()
                
                # Generiere zufällige Rotationsmatrix
                rot = Rotation.random(random_state=i)  # Verschiedene Seeds
                rot_matrix = rot.as_matrix()
                
                # Generiere zufällige Translation
                translation = np.random.normal(0, 2.0, 3)
                
                # Wende Rotation und Translation auf jedes Atom an
                for atom_idx in range(rot_mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(atom_idx)
                    
                    # Konvertiere zu NumPy-Array
                    coord = np.array([pos.x, pos.y, pos.z])
                    
                    # Rotiere
                    rotated_coord = np.dot(rot_matrix, coord)
                    
                    # Verschiebe
                    final_coord = rotated_coord + translation
                    
                    # Setze neue Position
                    new_pos = Geometry.Point3D(
                        final_coord[0],
                        final_coord[1],
                        final_coord[2]
                    )
                    conf.SetAtomPosition(atom_idx, new_pos)
                
                rotated_mols.append(rot_mol)
            
            self._update_statistics("augmentation_stats", {
                "rotations_applied": len(rotated_mols)
            })
            
        except Exception as e:
            self.logger.error(f"Fehler bei Rotation/Translation: {str(e)}")
            self.statistics["step_specific"]["augmentation_stats"]["errors"]["rotation"] += 1
            
            # Fallback: Verwende das Originalmolekül
            rotated_mols = [mol]
            
        return rotated_mols
    
    def cleanup(self):
        """Bereinigt temporäre Dateien und Verzeichnisse."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir)
            self.logger.info(f"Temporäres Verzeichnis {self.temp_dir} gelöscht")
        except Exception as e:
            self.logger.warning(f"Fehler beim Löschen des temporären Verzeichnisses: {str(e)}")
    
    def __del__(self):
        """Bereinigt beim Löschen des Objekts."""
        self.cleanup()


def augment_molecules(input_file: str, output_dir: str, config: Optional[Dict] = None) -> List[str]:
    """
    Hauptfunktion zur Augmentierung von Molekülen aus einer Datei.
    
    Args:
        input_file: Pfad zur Eingabedatei (.mol oder .sdf)
        output_dir: Verzeichnis zum Speichern der augmentierten Moleküle
        config: Konfiguration für die Augmentierung, kann auch ausgewählte Variantenindizes enthalten
        
    Returns:
        Liste der Pfade zu den augmentierten Moleküldateien
    """
    try:
        augmenter = MoleculeAugmenter(config)
        augmented_files = augmenter.augment_mol_file(input_file, output_dir)
        augmenter.finalize()
        return augmented_files
    except Exception as e:
        logging.error(f"Fehler bei der Molekülaugmentierung: {str(e)}")
        return []


if __name__ == "__main__":
    import argparse
    import unittest
    import tempfile
    import shutil
    
    class TestMoleculeAugmenter(unittest.TestCase):
        """Unit-Tests für den MoleculeAugmenter"""
        
        def setUp(self):
            """Erstellt ein temporäres Verzeichnis und eine Test-Moleküldatei"""
            self.test_dir = tempfile.mkdtemp(prefix="augmenter_test_")
            self.output_dir = os.path.join(self.test_dir, "output")
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Erstelle eine Test-Moleküldatei (Wasser) mit korrigiertem Format
            self.test_mol = """H2O
     RDKit          3D

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.9511    0.3090    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.9511    0.3090    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
M  END"""
            
            self.test_file = os.path.join(self.test_dir, "test_water.mol")
            with open(self.test_file, 'w') as f:
                f.write(self.test_mol)
        
        def tearDown(self):
            """Löscht das temporäre Verzeichnis"""
            try:
                shutil.rmtree(self.test_dir)
            except Exception as e:
                print(f"Fehler beim Löschen des temporären Verzeichnisses: {e}")
        
        def test_basic_augmentation(self):
            """Testet die grundlegende Augmentierungsfunktionalität"""
            config = {
                "num_conformers": 2,
                "use_geometric_perturbation": True,
                "perturbation_magnitude": 0.1,
                "use_rotation_translation": True,
                "num_rotation_samples": 2
            }
            
            augmented_files = augment_molecules(self.test_file, self.output_dir, config)
            
            # Überprüfe, ob Dateien erstellt wurden
            self.assertTrue(len(augmented_files) > 0, "Keine augmentierten Moleküle erzeugt")
            
            # Überprüfe, ob die Dateien existieren
            for aug_file in augmented_files:
                self.assertTrue(os.path.exists(aug_file), f"Augmentierte Datei {aug_file} existiert nicht")
                
                # Überprüfe, ob die Datei ein gültiges Molekül enthält
                # Wichtig: removeHs=False verwenden, um die Wasserstoffatome nicht zu entfernen!
                mol = Chem.MolFromMolFile(aug_file, removeHs=False)
                self.assertIsNotNone(mol, f"Datei {aug_file} enthält kein gültiges Molekül")
                
                # Überprüfe, ob die Atom- und Bindungszahl erhalten bleibt
                self.assertEqual(mol.GetNumAtoms(), 3, "Falsche Anzahl an Atomen")
                self.assertEqual(mol.GetNumBonds(), 2, "Falsche Anzahl an Bindungen")
        
        def test_conformer_generation(self):
            """Testet die Generierung von Konformeren"""
            config = {
                "num_conformers": 3,
                "use_geometric_perturbation": False,
                "use_rotation_translation": False
            }
            
            augmenter = MoleculeAugmenter(config)
            mol = Chem.MolFromMolBlock(self.test_mol)
            conformers = augmenter._generate_conformers(mol)
            
            # Überprüfe, ob Konformere erzeugt wurden
            self.assertGreaterEqual(len(conformers), 1, "Keine Konformere erzeugt")
            
            # Überprüfe, ob die Konformere unterschiedlich sind
            if len(conformers) >= 2:
                conf1 = conformers[0].GetConformer()
                conf2 = conformers[1].GetConformer()
                
                pos1 = conf1.GetAtomPosition(0)  # Position des ersten Atoms
                pos2 = conf2.GetAtomPosition(0)
                
                # Überprüfe, ob die Positionen unterschiedlich sind
                self.assertFalse(
                    (pos1.x == pos2.x and pos1.y == pos2.y and pos1.z == pos2.z),
                    "Konformere haben identische Atompositionen"
                )
        
        def test_geometric_perturbation(self):
            """Testet die geometrische Störung"""
            config = {
                "perturbation_magnitude": 0.5
            }
            
            augmenter = MoleculeAugmenter(config)
            mol = Chem.MolFromMolBlock(self.test_mol)
            
            # Stelle sicher, dass 3D-Koordinaten vorhanden sind
            AllChem.EmbedMolecule(mol)
            
            perturbed_mols = augmenter._apply_perturbations(mol)
            
            # Überprüfe, ob gestörte Moleküle erzeugt wurden
            self.assertGreaterEqual(len(perturbed_mols), 1, "Keine gestörten Moleküle erzeugt")
            
            # Überprüfe, ob die Störung angewendet wurde
            orig_conf = mol.GetConformer()
            pert_conf = perturbed_mols[0].GetConformer()
            
            # Überprüfe für jedes Atom, ob die Position geändert wurde
            any_change = False
            for i in range(mol.GetNumAtoms()):
                orig_pos = orig_conf.GetAtomPosition(i)
                pert_pos = pert_conf.GetAtomPosition(i)
                
                if (orig_pos.x != pert_pos.x or orig_pos.y != pert_pos.y or orig_pos.z != pert_pos.z):
                    any_change = True
                    break
            
            self.assertTrue(any_change, "Keine geometrische Störung angewendet")
        
        def test_rotation_translation(self):
            """Testet Rotation und Translation"""
            config = {
                "num_rotation_samples": 2
            }
            
            augmenter = MoleculeAugmenter(config)
            mol = Chem.MolFromMolBlock(self.test_mol)
            
            # Stelle sicher, dass 3D-Koordinaten vorhanden sind
            AllChem.EmbedMolecule(mol)
            
            rotated_mols = augmenter._apply_rotations(mol)
            
            # Überprüfe, ob rotierte Moleküle erzeugt wurden
            self.assertEqual(len(rotated_mols), 2, "Falsche Anzahl rotierter Moleküle")
            
            # Überprüfe, ob die Rotation angewendet wurde
            orig_conf = mol.GetConformer()
            rot_conf = rotated_mols[0].GetConformer()
            
            # Überprüfe für jedes Atom, ob die Position geändert wurde
            any_change = False
            for i in range(mol.GetNumAtoms()):
                orig_pos = orig_conf.GetAtomPosition(i)
                rot_pos = rot_conf.GetAtomPosition(i)
                
                if (orig_pos.x != rot_pos.x or orig_pos.y != rot_pos.y or orig_pos.z != rot_pos.z):
                    any_change = True
                    break
            
            self.assertTrue(any_change, "Keine Rotation/Translation angewendet")
        
        def test_integration_with_mol_to_sequence(self):
            """Testet die Integration mit MolToSequence"""
            try:
                from PreprocessingPipeline.MolToSequence import molToSequenceFunction
                
                config = {
                    "num_conformers": 1,
                    "use_geometric_perturbation": True,
                    "perturbation_magnitude": 0.1,
                    "use_rotation_translation": True,
                    "num_rotation_samples": 1
                }
                
                # Generiere augmentierte Moleküle
                augmented_files = augment_molecules(self.test_file, self.output_dir, config)
                self.assertTrue(len(augmented_files) > 0, "Keine augmentierten Moleküle erzeugt")
                
                # Verarbeite mit MolToSequence
                sequence_dir = os.path.join(self.test_dir, "sequences")
                os.makedirs(sequence_dir, exist_ok=True)
                
                for i, aug_file in enumerate(augmented_files):
                    success = molToSequenceFunction(aug_file, sequence_dir, i+1)
                    self.assertTrue(success, f"MolToSequence fehlgeschlagen für {aug_file}")
                    
                    # Überprüfe, ob die Ausgabedatei existiert
                    output_file = os.path.join(sequence_dir, f"Länge_{i+1}.json")
                    self.assertTrue(os.path.exists(output_file), f"Ausgabedatei {output_file} existiert nicht")
                    
                    # Überprüfe, ob die Ausgabedatei gültige Daten enthält
                    with open(output_file, 'r') as f:
                        import json
                        data = json.load(f)
                        self.assertTrue(len(data) > 0, "Keine Sequenzen in der Ausgabedatei")
                
            except ImportError:
                self.skipTest("MolToSequence nicht verfügbar")
                
    # Führe die Tests durch und starte dann ggf. die Kommandozeilenverarbeitung
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Nur Tests ausführen
        unittest.main(argv=['first-arg-is-ignored'])
    else:
        # Normale Kommandozeilenverarbeitung
        parser = argparse.ArgumentParser(description="Molekülaugmentierung")
        parser.add_argument("input_file", help="Pfad zur Eingabedatei (.mol oder .sdf)")
        parser.add_argument("output_dir", help="Verzeichnis zum Speichern der augmentierten Moleküle")
        parser.add_argument("--num_conformers", type=int, default=3, help="Anzahl der zu generierenden Konformere")
        parser.add_argument("--perturbation", type=float, default=0.1, help="Stärke der geometrischen Störungen")
        parser.add_argument("--num_rotations", type=int, default=2, help="Anzahl der Rotationsvarianten")
        parser.add_argument("--no_perturbation", action="store_true", help="Keine geometrischen Störungen anwenden")
        parser.add_argument("--no_rotation", action="store_true", help="Keine Rotationen anwenden")
        parser.add_argument("--test", action="store_true", help="Führt die Unit-Tests aus")
        
        args = parser.parse_args()
        
        if args.test:
            # Wenn --test angegeben wurde, führe die Tests aus
            unittest.main(argv=['first-arg-is-ignored'])
        else:
            config = {
                "num_conformers": args.num_conformers,
                "use_geometric_perturbation": not args.no_perturbation,
                "perturbation_magnitude": args.perturbation,
                "use_rotation_translation": not args.no_rotation,
                "num_rotation_samples": args.num_rotations
            }
            
            augmented_files = augment_molecules(args.input_file, args.output_dir, config)
            
            print(f"Insgesamt {len(augmented_files)} augmentierte Moleküle erzeugt.")
            
            # Beispiel für eine Weiterverarbeitung mit MolToSequence
            try:
                from PreprocessingPipeline.MolToSequence import molToSequenceFunction
                
                print("\nWeiterverarbeitung mit MolToSequence:")
                for i, aug_file in enumerate(augmented_files[:5]):  # Nur die ersten 5 zur Demonstration
                    output_dir = os.path.join(args.output_dir, "sequences")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    success = molToSequenceFunction(aug_file, output_dir, i+1)
                    print(f"  Molekül {i+1}: {'Erfolgreich' if success else 'Fehlgeschlagen'}")
                    
                if len(augmented_files) > 5:
                    print(f"  ... und {len(augmented_files) - 5} weitere")
            except ImportError:
                print("\nMolToSequence nicht verfügbar für Weiterverarbeitung.") 