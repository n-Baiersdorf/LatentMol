from setuptools import setup, find_packages

setup(
    name="latentmol",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "rdkit",
    ],
    python_requires=">=3.7",
) 