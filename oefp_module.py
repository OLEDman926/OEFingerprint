import os
import json
import time
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem import MolStandardize

# ----------------------------
# Public API constants
# ----------------------------
DEFAULT_JSON_PATH = os.path.join("input", "OEFP.json")
OutputType = Literal["list", "dictionary", "dataframe"]
OnError = Literal["zeros", "skip"]

RDLogger.DisableLog("rdApp.info")


class OEFPEncoder:
    """
    OEFP (Organic Electronic Fingerprint) encoder.
    - Loads SMARTS/SMILES fragment definitions from JSON.
    - Generates binary or count fingerprints via substructure matching.
    """

    def __init__(self, json_path: str = DEFAULT_JSON_PATH) -> None:
        self.json_path: str = json_path
        self.fragment_spec: Dict[str, Dict[str, str]] = self._load_fragment_spec(json_path)
        self.bit_keys: List[str] = list(self.fragment_spec.keys())
        self.bit_size: int = len(self.bit_keys)
        self.compiled_patterns: Dict[str, Optional[Chem.Mol]] = self._compile_patterns(self.fragment_spec)

    # ----------------------------
    # Loading / compilation
    # ----------------------------
    @staticmethod
    def _load_fragment_spec(json_path: str) -> Dict[str, Dict[str, str]]:
        with open(json_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
        # minimal schema check
        for k, v in spec.items():
            if "pattern" not in v or "type" not in v:
                raise ValueError(f"Invalid fragment entry for key '{k}': requires 'pattern' and 'type'.")
        return spec

    @staticmethod
    def _compile_patterns(fragment_spec: Dict[str, Dict[str, str]]) -> Dict[str, Optional[Chem.Mol]]:
        compiled: Dict[str, Optional[Chem.Mol]] = {}
        for key, item in fragment_spec.items():
            pattern = item["pattern"]
            ftype = item["type"].strip().upper()
            mol: Optional[Chem.Mol]
            try:
                if ftype == "SMARTS":
                    mol = Chem.MolFromSmarts(pattern)
                elif ftype == "SMILES":
                    mol = Chem.MolFromSmiles(pattern)
                else:
                    mol = None
            except Exception:
                mol = None
            compiled[key] = mol
        return compiled

    # ----------------------------
    # Public API
    # ----------------------------
    def generate(
        self,
        structures: Union[str, Chem.Mol, Sequence[Union[str, Chem.Mol]]],
        count: bool = False,
        output: OutputType = "list",
        verbose: bool = False,
        on_error: OnError = "zeros",
    ) -> Union[List[List[int]], Dict[str, List[int]], pd.DataFrame]:
        """
        Generate OEFP vectors.

        Parameters
        ----------
        structures : SMILES or RDKit Mol or sequence of them
        count      : False -> binary(0/1), True -> match counts
        output     : 'list' | 'dictionary' | 'dataframe'
        verbose    : print ETA progress (slows down for very large batches)
        on_error   : 'zeros' -> replace invalid SMILES with zero-vector,
                     'skip'  -> drop invalid items from output

        Returns
        -------
        List[List[int]] or Dict[str, List[int]] or pandas.DataFrame
        """
        items: List[Union[str, Chem.Mol]]
        if isinstance(structures, (str, Chem.Mol)):
            items = [structures]
        else:
            items = list(structures)

        vectors: List[List[int]] = []
        kept_indices: List[int] = []

        t0 = time.perf_counter()
        n = len(items)

        for i, obj in enumerate(items):
            mol = self._to_mol(obj)
            if mol is None:
                if on_error == "zeros":
                    vectors.append([0] * self.bit_size)
                    kept_indices.append(i)
                # if skip: simply do not append
                if verbose:
                    self._print_progress(i + 1, n, t0)
                continue

            mol_std = self._standardize_mol(mol)
            vec = self._match_vector(mol_std, count)
            vectors.append(vec)
            kept_indices.append(i)

            if verbose:
                self._print_progress(i + 1, n, t0)

        if output == "list":
            return vectors
        elif output == "dictionary":
            return self.list_to_dict(vectors)
        elif output == "dataframe":
            df = pd.DataFrame(self.list_to_dict(vectors), index=kept_indices)
            df.index.name = "input_index"
            return df
        else:
            raise ValueError("output must be one of {'list','dictionary','dataframe'}.")

    def append_to_dataframe(
        self,
        data: pd.DataFrame,
        smiles_column: str,
        count: bool = False,
        verbose: bool = False,
        on_error: OnError = "zeros",
    ) -> pd.DataFrame:
        """
        Append OEFP columns to a DataFrame containing SMILES strings.
        """
        if smiles_column not in data.columns:
            raise KeyError(f"Column '{smiles_column}' not found in DataFrame.")
        fp_df = self.generate(
            data[smiles_column].tolist(),
            count=count,
            output="dataframe",
            verbose=verbose,
            on_error=on_error,
        )
        # align by positional index
        fp_df_aligned = fp_df.reindex(range(len(data))).fillna(0).astype(int)
        fp_df_aligned.columns = self.bit_keys
        return pd.concat([data.reset_index(drop=True), fp_df_aligned.reset_index(drop=True)], axis=1)

    def draw_bit(self, bit_key: str, legend: Optional[str] = None, size: Tuple[int, int] = (300, 300)):
        """
        Draw a single fragment associated with bit_key.
        """
        if bit_key not in self.compiled_patterns:
            raise KeyError(f"Unknown bit key: {bit_key}")
        mol = self.compiled_patterns[bit_key]
        if mol is None:
            raise ValueError(f"Pattern for '{bit_key}' failed to parse; cannot draw.")
        return Draw.MolToImage(mol, legend=(legend or bit_key), size=size)

    def draw_bits(
        self,
        bit_keys: Union[str, Sequence[str]],
        mols_per_row: int = 3,
        sub_img_size: Tuple[int, int] = (300, 300),
    ):
        """
        Draw a grid image of multiple fragments.
        """
        keys = [bit_keys] if isinstance(bit_keys, str) else list(bit_keys)
        mols: List[Chem.Mol] = []
        legends: List[str] = []
        for k in keys:
            mol = self.compiled_patterns.get(k, None)
            if mol is not None:
                mols.append(mol)
                legends.append(k)
        if not mols:
            raise ValueError("No drawable fragments in provided keys.")
        return Draw.MolsToGridImage(mols, molsPerRow=mols_per_row, legends=legends, subImgSize=sub_img_size)

    # ----------------------------
    # Helpers
    # ----------------------------
    @staticmethod
    def list_to_dict(vectors: List[List[int]]) -> Dict[str, List[int]]:
        if not vectors:
            return {}
        bit_len = len(vectors[0])
        out: Dict[str, List[int]] = {f"OEFP{i}": [] for i in range(bit_len)}
        for vec in vectors:
            if len(vec) != bit_len:
                raise ValueError("Inconsistent vector length detected.")
            for j, val in enumerate(vec):
                out[f"OEFP{j}"].append(int(val))
        return out

    def _match_vector(self, mol: Chem.Mol, count: bool) -> List[int]:
        vec: List[int] = []
        for key in self.bit_keys:
            patt = self.compiled_patterns[key]
            if patt is None:
                vec.append(0)
                continue
            if count:
                vec.append(len(mol.GetSubstructMatches(patt)))
            else:
                vec.append(1 if mol.HasSubstructMatch(patt) else 0)
        return vec

    @staticmethod
    def _to_mol(obj: Union[str, Chem.Mol]) -> Optional[Chem.Mol]:
        if isinstance(obj, Chem.Mol):
            return obj
        if isinstance(obj, str):
            try:
                return Chem.MolFromSmiles(obj)
            except Exception:
                return None
        return None

    @staticmethod
    def _standardize_mol(mol: Chem.Mol) -> Chem.Mol:
        mol = MolStandardize.rdMolStandardize.Cleanup(mol)
        mol = MolStandardize.rdMolStandardize.FragmentParent(mol)
        uncharger = MolStandardize.rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        return mol

    @staticmethod
    def _print_progress(done: int, total: int, t0: float) -> None:
        elapsed = time.perf_counter() - t0
        remaining = (elapsed / max(done, 1)) * (total - done)
        bar_len = 20
        ratio = done / total
        hashes = int(ratio * bar_len)
        bar = "#" * hashes + " " * (bar_len - hashes)
        print(f"{done}/{total}  [{bar}] {int(ratio*100)}%  ETA {int(remaining)}s", end="\r")
        if done == total:
            print()
