"""Preprocessing: build drugtopathway effect matrix from DGIdb + Reactome.

Inputs
------
- data/raw/dgidb_interactions.tsv
- data/raw/Ensembl2Reactome.txt

Outputs
-------
- data/processed/drug_pathway_effects.npz
- data/processed/metadata.json

Notes
-----
DGIdb interactions are typically keyed by gene symbols. Reactome mapping here is
Ensembl-based, so we map symbols to Ensembl using mygene.info (cached locally).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import mygene


@dataclass(frozen=True)
class EffectMatrix:
    drug_names: List[str]
    pathway_ids: List[str]
    pathway_names: List[str]
    effects: np.ndarray  # shape (N_drugs, N_pathways)


def _normalize_symbol(sym: str) -> str:
    if sym is None:
        return ""
    return re.sub(r"\s+", "", str(sym)).upper()


def load_dgidb_interactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    # Try to locate drug and gene columns robustly
    cols = {c.lower(): c for c in df.columns}

    # Common DGIdb column patterns (historically):
    #   - drug_name, gene_name
    #   - drug_claim_name, gene_claim_name
    drug_col = None
    gene_col = None
    for cand in ["drug_name", "drug", "drugclaim", "drug_claim_name"]:
        for k, v in cols.items():
            if cand in k:
                drug_col = v
                break
        if drug_col:
            break
    for cand in ["gene_name", "gene", "geneclaim", "gene_claim_name", "gene_symbol", "genesymbol"]:
        for k, v in cols.items():
            if cand in k:
                gene_col = v
                break
        if gene_col:
            break

    if drug_col is None or gene_col is None:
        raise ValueError(
            f"Could not detect drug/gene columns in DGIdb file. Columns={list(df.columns)}"
        )

    out = df[[drug_col, gene_col]].rename(columns={drug_col: "drug", gene_col: "gene_symbol"})
    out["drug"] = out["drug"].astype(str).str.strip()
    out["gene_symbol"] = out["gene_symbol"].astype(str).map(_normalize_symbol)
    out = out[(out["drug"] != "") & (out["gene_symbol"] != "")].drop_duplicates()
    return out


def load_reactome_ensembl2reactome(path: Path) -> pd.DataFrame:
    # Reactome file is tab-delimited:
    # EnsemblGeneID  ReactomePathwayID  URL  PathwayName  EvidenceCode  Species
    df = pd.read_csv(path, sep="\t", header=None, dtype=str)
    if df.shape[1] < 6:
        raise ValueError("Ensembl2Reactome.txt format unexpected (expected >=6 columns)")
    df = df.iloc[:, :6]
    df.columns = ["ensembl", "pathway_id", "reactome_url", "pathway_name", "evidence", "species"]
    df["ensembl"] = df["ensembl"].astype(str).str.strip()
    df["pathway_id"] = df["pathway_id"].astype(str).str.strip()
    df["pathway_name"] = df["pathway_name"].astype(str).str.strip()
    df["species"] = df["species"].astype(str).str.strip()
    # Focus on Homo sapiens only
    df = df[df["species"].str.lower().isin(["homo sapiens", "homo\u00a0sapiens", "homo sapiens (human)"])]
    df = df[(df["ensembl"] != "") & (df["pathway_id"] != "")]
    df = df.drop_duplicates(subset=["ensembl", "pathway_id"])
    return df[["ensembl", "pathway_id", "pathway_name"]]


def map_symbols_to_ensembl(symbols: Iterable[str], cache_path: Path, species: str = "human") -> Dict[str, str]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        cache_df = pd.read_csv(cache_path, sep="\t", dtype=str)
        cache = dict(zip(cache_df["symbol"], cache_df["ensembl"]))
    else:
        cache = {}

    missing = [s for s in set(symbols) if s and s not in cache]
    if missing:
        mg = mygene.MyGeneInfo()
        # Query in batches to be polite
        results = []
        batch_size = 1000
        for i in range(0, len(missing), batch_size):
            batch = missing[i : i + batch_size]
            res = mg.querymany(
                batch,
                scopes="symbol",
                fields="ensembl.gene",
                species=species,
                as_dataframe=True,
                returnall=False,
                verbose=False,
            )
            # res index is the query; columns include 'ensembl.gene' potentially
            if not isinstance(res, pd.DataFrame):
                res = pd.DataFrame(res)
            if "ensembl.gene" in res.columns:
                # pick first if list-like
                mapped = res["ensembl.gene"].copy()
                mapped = mapped.apply(lambda x: x[0] if isinstance(x, list) and x else x)
            elif "ensembl" in res.columns:
                mapped = res["ensembl"].apply(
                    lambda x: x.get("gene") if isinstance(x, dict) else (x[0].get("gene") if isinstance(x, list) and x else None)
                )
            else:
                mapped = pd.Series(index=res.index, data=None)

            for sym, ens in mapped.items():
                if isinstance(sym, str):
                    cache[sym] = str(ens) if ens and str(ens) != "nan" else ""

        # persist cache
        out_df = pd.DataFrame({"symbol": list(cache.keys()), "ensembl": list(cache.values())})
        out_df.to_csv(cache_path, sep="\t", index=False)

    return cache


def build_effect_matrix(
    dgidb_df: pd.DataFrame,
    reactome_df: pd.DataFrame,
    top_drugs: int = 60,
    top_pathways: int = 40,
    seed: int = 42,
) -> EffectMatrix:
    # Map gene symbols to Ensembl, then join to pathways
    cache_path = Path("data/processed/symbol_to_ensembl.tsv")
    sym2ens = map_symbols_to_ensembl(dgidb_df["gene_symbol"].unique(), cache_path=cache_path)

    dgidb_df = dgidb_df.copy()
    dgidb_df["ensembl"] = dgidb_df["gene_symbol"].map(sym2ens).fillna("")
    dgidb_df = dgidb_df[dgidb_df["ensembl"] != ""].drop_duplicates(subset=["drug", "ensembl"])

    joined = dgidb_df.merge(reactome_df, on="ensembl", how="inner")
    if joined.empty:
        raise ValueError(
            "After mapping symbolstoEnsembl and joining to Reactome, no rows remained. "

            "This can happen if mygene mapping failed or Reactome download is missing."

        )

    # Pick top drugs by # unique pathways covered
    drug_counts = joined.groupby("drug")["pathway_id"].nunique().sort_values(ascending=False)
    drugs = drug_counts.head(top_drugs).index.tolist()
    joined = joined[joined["drug"].isin(drugs)]

    # Pick top pathways by # unique drugs (coverage)
    path_counts = joined.groupby("pathway_id")["drug"].nunique().sort_values(ascending=False)
    pathway_ids = path_counts.head(top_pathways).index.tolist()
    joined = joined[joined["pathway_id"].isin(pathway_ids)]

    # Build pathway name lookup (stable)
    path_name = joined.drop_duplicates(subset=["pathway_id"]).set_index("pathway_id")["pathway_name"].to_dict()
    pathway_names = [path_name.get(pid, pid) for pid in pathway_ids]

    # Drug to pathway coverage matrix (counts of unique target genes)
    pivot = (
        joined.groupby(["drug", "pathway_id"])["ensembl"]
        .nunique()
        .reset_index()
        .pivot(index="drug", columns="pathway_id", values="ensembl")
        .fillna(0.0)
    )
    pivot = pivot.reindex(index=drugs, columns=pathway_ids).fillna(0.0)

    # Convert counts to normalized effects in [0,1]
    mat = pivot.to_numpy(dtype=np.float32)
    # normalize per drug so effects are comparable
    row_sums = mat.sum(axis=1, keepdims=True) + 1e-6
    effects = mat / row_sums

    return EffectMatrix(drug_names=drugs, pathway_ids=pathway_ids, pathway_names=pathway_names, effects=effects)


def save_effects(em: EffectMatrix, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        outdir / "drug_pathway_effects.npz",
        effects=em.effects,
        drug_names=np.array(em.drug_names, dtype=object),
        pathway_ids=np.array(em.pathway_ids, dtype=object),
        pathway_names=np.array(em.pathway_names, dtype=object),
    )


def load_effects(path: Path) -> EffectMatrix:
    z = np.load(path, allow_pickle=True)
    return EffectMatrix(
        drug_names=list(z["drug_names"].tolist()),
        pathway_ids=list(z["pathway_ids"].tolist()),
        pathway_names=list(z["pathway_names"].tolist()),
        effects=z["effects"].astype(np.float32),
    )