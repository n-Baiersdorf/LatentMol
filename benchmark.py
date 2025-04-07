import time
import psutil
import multiprocessing as mp
import logging
from pathlib import Path
from PreprocessingPipeline.molecular.MolecularProperties import MolecularPropertiesCalculator
from PreprocessingPipeline.splitters.ChemicalDataSplitter import ChemicalDataSplitter
from PreprocessingPipeline.filters.base_processor import ProcessingConfig

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def get_system_info():
    """Gibt Systeminformationen aus"""
    cpu_count = mp.cpu_count()
    total_ram = psutil.virtual_memory().total / (1024**3)  # in GB
    return {
        "cpu_cores": cpu_count,
        "total_ram_gb": total_ram,
        "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else "N/A"
    }

def benchmark_molecular_properties():
    """Testet die Performance der MolecularProperties-Klasse"""
    logger.info("Starte Benchmark für MolecularProperties...")
    
    # Erstelle Test-Moleküle
    from rdkit import Chem
    test_molecules = []
    for i in range(1000):
        smiles = f"C{'C' * i}O"  # Einfache Alkohole
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            test_molecules.append(mol)
    
    # Initialisiere Calculator
    calculator = MolecularPropertiesCalculator()
    
    # Führe Benchmark durch
    start_time = time.time()
    results = calculator.process_molecules(test_molecules)
    end_time = time.time()
    
    duration = end_time - start_time
    logger.info(f"MolecularProperties Benchmark abgeschlossen in {duration:.2f} Sekunden")
    logger.info(f"Verarbeitete Moleküle: {len(results)}")
    logger.info(f"Pro Molekül: {duration/len(results)*1000:.2f} ms")
    
    return duration

def benchmark_data_splitter():
    """Testet die Performance des ChemicalDataSplitter"""
    logger.info("Starte Benchmark für ChemicalDataSplitter...")
    
    # Erstelle Testdatei
    test_file = Path("test_data.sdf")
    with open(test_file, "w") as f:
        for i in range(1000):
            f.write(f"$$$$\nTest Molecule {i}\n\n\n$$$$\n")
    
    # Initialisiere Splitter
    config = ProcessingConfig()
    splitter = ChemicalDataSplitter(
        input_file=str(test_file),
        output_directory="test_output",
        config=config
    )
    
    # Führe Benchmark durch
    start_time = time.time()
    splitter.split_file()
    end_time = time.time()
    
    duration = end_time - start_time
    logger.info(f"ChemicalDataSplitter Benchmark abgeschlossen in {duration:.2f} Sekunden")
    
    # Aufräumen
    test_file.unlink()
    import shutil
    shutil.rmtree("test_output")
    
    return duration

def main():
    """Hauptfunktion für den Benchmark"""
    logger.info("Starte Performance-Benchmark...")
    
    # Systeminformationen
    system_info = get_system_info()
    logger.info(f"Systeminformationen:")
    logger.info(f"CPU-Kerne: {system_info['cpu_cores']}")
    logger.info(f"RAM: {system_info['total_ram_gb']:.1f} GB")
    logger.info(f"CPU-Frequenz: {system_info['cpu_freq']} MHz")
    
    # Führe Benchmarks durch
    mol_prop_time = benchmark_molecular_properties()
    splitter_time = benchmark_data_splitter()
    
    # Zusammenfassung
    logger.info("\nBenchmark-Zusammenfassung:")
    logger.info(f"MolecularProperties: {mol_prop_time:.2f} Sekunden")
    logger.info(f"ChemicalDataSplitter: {splitter_time:.2f} Sekunden")
    logger.info(f"Gesamtzeit: {mol_prop_time + splitter_time:.2f} Sekunden")

if __name__ == "__main__":
    main() 