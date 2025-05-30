import os
import json
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem import MolStandardize
from rdkit.Chem.Fingerprints import FingerprintMols
import time
import numbers

# 경로 설정
json_path = os.path.join("input", "OEFP.json")

class OEFPFingerprints:

    with open(json_path, "r", encoding="utf-8") as f:
        fragment_dict = json.load(f)

    OEFPKeys = list(fragment_dict.keys())
    bit_size = len(OEFPKeys)

    def build_mol_dict(fragment_dict):
        mol_dict = {}
        for key in fragment_dict:
            pattern = fragment_dict[key]["pattern"]
            ftype = fragment_dict[key]["type"].upper()
            try:
                mol = Chem.MolFromSmarts(pattern) if ftype == "SMARTS" else Chem.MolFromSmiles(pattern)
            except:
                mol = None
            mol_dict[key] = mol
        return mol_dict

    OEFPDictMol = build_mol_dict(fragment_dict)

    def GenerateOEFPFingerprints(structures, count=False, output_type='list', verbose=True):
        if isinstance(structures, str) or isinstance(structures, Chem.rdchem.Mol):
            structures = [structures]

        oefp_list = []

        RDLogger.DisableLog('rdApp.info')
        for i, ligand in enumerate(structures):
            start = time.time()
            if isinstance(ligand, str):
                ligand = Chem.MolFromSmiles(ligand)
            elif isinstance(ligand, Chem.rdchem.Mol):
                ligand = ligand
            ligand = MolStandardize.rdMolStandardize.Cleanup(ligand) 
            ligand = MolStandardize.rdMolStandardize.FragmentParent(ligand)
            uncharger = MolStandardize.rdMolStandardize.Uncharger()
            ligand = uncharger.uncharge(ligand)

            if count is False:
                oefp = [1 if ligand.HasSubstructMatch(OEFPFingerprints.OEFPDictMol[fp]) else 0 for fp in OEFPFingerprints.OEFPKeys]
            elif count:
                oefp = [len(ligand.GetSubstructMatches(OEFPFingerprints.OEFPDictMol[fp])) for fp in OEFPFingerprints.OEFPKeys]
            oefp_list.append(oefp)

            end = time.time()
            if verbose:
                estimated_time = (end - start) * (len(structures) - i + 1)
                estimated_time_str = time.strftime("%H:%M:%S", time.gmtime(estimated_time))
                print(str(i+1) + "/" + str(len(structures)) + " structures, time remaining: " + str(estimated_time_str))
                percentage = (i+1) / len(structures)
                for hashes in range(int(percentage * 20)): print('#', end='')
                for spaces in range(20 - int(percentage * 20)): print(' ', end='')
                print(' ' + str(int(percentage * 100)) + '%')
                from IPython.display import clear_output
                clear_output(wait=True)

        if output_type == 'list':
            return oefp_list
        elif output_type == 'dictionary':
            oefp_list = OEFPFingerprints.ListToDictionary(oefp_list)
        elif output_type == 'dataframe':
            oefp_list = pd.DataFrame(OEFPFingerprints.ListToDictionary(oefp_list))

        return oefp_list

    def GenerateOEFPFingerprintsToDataFrame(data, structures_column, count=False, verbose=True):
        oefp_list = OEFPFingerprints.GenerateOEFPFingerprints(data[structures_column], count, 'dictionary', verbose=verbose)
        data = pd.concat([data, pd.DataFrame(oefp_list, index=data.index, columns=OEFPFingerprints.OEFPKeys)], axis=1)
        return data

    def ListToDictionary(oefp_list):
        descriptors = {}
        for i, fp_vector in enumerate(oefp_list):
            for j, key in enumerate(OEFPFingerprints.OEFPKeys):
                if i == 0:
                    descriptors[key] = [fp_vector[j]]
                else:
                    descriptors[key].append(fp_vector[j])
        return descriptors

    def DrawFingerprint(fp_key):
        mol = OEFPFingerprints.OEFPDictMol[fp_key]
        img = Draw.MolToImage(mol, legend=fp_key)
        return img

    def DrawFingerprints(fp_keys, molsPerRow=3):
        if isinstance(fp_keys, str):
            fp_keys = [fp_keys]
        mols = [OEFPFingerprints.OEFPDictMol[k] for k in fp_keys]
        return Draw.MolsToGridImage(mols, molsPerRow=molsPerRow, legends=fp_keys, subImgSize=(300, 300))
