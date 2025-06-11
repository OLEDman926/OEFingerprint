# OrganicElectronicFingerprint Documentation

## Overview
This module generates OEFP (Organic Electronic Fingerprint) vectors based on substructure matching using RDKit. It supports binary or count-based encoding and offers visualization and DataFrame integration utilities.

## Installation

### Create a new environment (recommended)
```bash
conda create -n oefp_env python=3.8 -y
conda activate oefp_env
```

### Install RDKit (required)
```bash
conda install -c conda-forge rdkit
```

### Install other required packages
```bash
pip install pandas numpy
```

### Optional: For ML & SHAP analysis
```bash
pip install scikit-learn xgboost shap seaborn
```

## Requirements

### Core requirements
* rdkit >= 2022.09.1
* pandas >= 2.0.0
* numpy >= 1.24.0

### For visualization
* matplotlib >= 3.7.0
* cairosvg >= 2.7.0
* cairocffi >= 1.6.0
* pillow >= 10.0.0

### Optional (for fingerprint similarity search, SHAP, and ML)
* scikit-learn >= 1.3.0
* xgboost >= 2.1.1
* shap >= 0.44.0
* seaborn >= 0.13.0

---

## Variables

**fragment_dict (*dictionary*)**  
Dictionary of OEFP fingerprint substructures from JSON file (keys = bit names, values = SMARTS or SMILES)

**OEFPDictMol (*dictionary*)**  
Dictionary of RDKit Mol objects corresponding to OEFP fingerprint substructures

**OEFPKeys (*list*)**  
List of all OEFP bit identifiers (e.g., "OLEDFP101", "OPDFP22")

**bit_size (*int*)**  
Total number of bits in the OEFP fingerprint

---

## Functions

### GenerateOEFPFingerprints(structures, count=False, output_type='list', verbose=True)
Generate list, dictionary, or DataFrame of OEFP fingerprints for given molecules  
* **Parameters**
  * structures (*list, string, or rdkit.Chem.Mol*) – input molecule(s)
  * count (*bool*) – if True, counts all matches instead of binary presence
  * output_type (*string*) – output format: 'list', 'dictionary', or 'dataframe'
  * verbose (*bool*) – show progress bar
* **Return**
  * list, dictionary, or DataFrame of OEFP fingerprints

---

### GenerateOEFPFingerprintsToDataFrame(data, structures_column, count=False, verbose=True)
Append OEFP fingerprint bits to a given DataFrame of molecular structures  
* **Parameters**
  * data (*DataFrame*) – DataFrame containing SMILES strings
  * structures_column (*string*) – name of the SMILES column
  * count (*bool*) – whether to count or use binary match
  * verbose (*bool*) – show progress
* **Return**
  * DataFrame with appended OEFP bit columns

---

### ListToDictionary(oefp_list)
Convert a list of OEFP vectors into a dictionary  
* **Parameters**
  * oefp_list (*list[list[int]]*) – list of OEFP bit vectors
* **Return**
  * dictionary – key: bit name, value: list of bit values

---
