# LatentMol: Preprocessing Pipeline for Deep Molecular Representation Learning

Welcome to the **LatentMol** repository! This project focuses on the development of a holistic molecular input format designed to enhance data-driven modeling in chemistry. The repository currently contains the unfinished but roughly functional **preprocessing pipeline**, which serves as the foundation for preparing large datasets for future model training.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Current Status](#current-status)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Future Plans](#future-plans)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

LatentMol aims to teach artificial neural networks the "language" of molecules via deep molecular representation learning. 
For this the input embedding shall integrate key chemical concepts such as:
- Mesomeric effects (not implemented yet)
- Protonation states (not implemented yet)
- Intermolecular interactions (not implemented yet)

The long-term goal is to enable applications such as reaction modeling, spectroscopic data analysis, and molecular property prediction. While the project envisions a comprehensive framework, this repository currently focuses on the **preprocessing pipeline**.

---

## Current Status

At this stage, the repository includes:
1. **Preprocessing Pipeline**: Converts molecular data (e.g., MolTables) into a custom sequence format for training Transformer models.
   - It filters molecules based on length, elements, and isotopes for consistency.
   - Normalizes appropriatly for tanh as activation function (for the input layers of the Transformer model)
2. **Basic Sequence Transformation**: Implements the foundation of the LatentMol input format.

---

## Features

- **Data Filtering**: Removes outliers such as non-standard isotopes or overly complex molecules.
- **Sequence Transformation**: Converts molecular data into LatentMol's input format.
- **Length-Based Sorting**: Organizes molecules by size for better handling during preprocessing.
- **Scalability**: Partly designed to handle large datasets efficiently.

---

## Installation

To set up the preprocessing pipeline, follow these steps:

1. Clone this repository:


```git clone https://github.com/n-Baiersdorf/LatentMol.git```
```cd latentmol```
>

2. Ensure you have Python 3.8+ installed


3. Create virtual environment:


```python -m venv myenv```


4. Activate virtual environment:


   Linux & MacOS: ```source .venv/bin/activate```


   Windows: ```.venv/bin/activate```


6. Install dependencies:


```pip install -r requirements.txt```



---

## Usage

Run the ```main.py``` script

In it you can configure some parameters, though the only significant ones are:
1. The Number of Permutations (these are augmented versions of molecules)
2. The Number of Molecules to download from PubChem (the script does it automatically) --> you can define it in steps of 500.000

---

## Future Plans

The following features are planned for future releases:
1. **Model Training**: Implementation of LatentMol-BERT using PyTorch. Auxiliary training tasks are supposed to be:
   1. NSP(Next Sentence Prediction)-like prediction of multiple molecules for preperation of reaction modelling
   2. MSM(Masked Sequence Modelling) so basically Diffusion
   3. Reconstruction via a Decoder-Head
   4. experimental exercises involving image-data such as structure drawings and spectral data 
3. **Evaluation Metrics**: Incorporation of benchmarks to compare LatentMol with existing methods.
4. **Reaction Modeling**: Expansion to support intermolecular interactions and dynamic processes.
5. **Integration with Lab Tools**: Deployment as a laboratory companion program.

---

## Contributing

Contributions are welcome! If you'd like to contribute:
1. Fork this repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to your fork (`git push origin feature-name`).
5. Open a pull request.

For major changes, please open an issue first to discuss your ideas.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Thank you for exploring LatentMol! If you have any questions or suggestions, feel free to open an issue or contact us directly.

