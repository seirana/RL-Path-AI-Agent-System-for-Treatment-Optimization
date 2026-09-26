"""Build a drug-to-pathway effect matrix from DGIdb and Reactome inputs.

The generated matrix is a research abstraction for the RL simulator. It represents
normalized pathway coverage derived from mapped drug-gene interactions; it is not a
clinical drug-effect estimate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import mygene
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EffectMatrix:
    drug_names: List[str]
    pathway_ids: List[str]
    pathway_names: List[str]
    effects: np.ndarray


def _normalize_symbol(sym: str) -> str:
    if sym is None:
        return ""
    return re.sub(r"\s+", "", str(sym)).upper()


def load_dgidb_interactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    cols = {column.lower(): column for column in df.columns}

    drug_col = None
    gene_col = None

    for candidate in [
        "drug_name",
        "drug",
        "drugclaim",
        "drug_claim_name",
    ]:
        for normalized, original in cols.items():
            if candidate in normalized:
                drug_col = original
                break
        if drug_col:
            break

    for candidate in [
        "gene_name",
        "gene",
        "geneclaim",
        "gene_claim_name",
        "gene_symbol",
        "genesymbol",
    ]:
        for normalized, original in cols.items():
            if candidate in normalized:
                gene_col = original
                break
        if gene_col:
            break

    if drug_col is None or gene_col is None:
        raise ValueError(
            "Could not detect drug/gene columns in DGIdb file. "
            f"Columns={list(df.columns)}"
        )

    out = df[[drug_col, gene_col]].rename(
        columns={
            drug_col: "drug",
            gene_col: "gene_symbol",
        }
    )
    out["drug"] = out["drug"].astype(str).str.strip()
    out["gene_symbol"] = (
        out["gene_symbol"].astype(str).map(_normalize_symbol)
    )
    out = out[
        (out["drug"] != "") & (out["gene_symbol"] != "")
    ].drop_duplicates()

    if out.empty:
        raise ValueError(
            "DGIdb input contained no usable drug-gene rows."
        )
    return out


def load_reactome_ensembl2reactome(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, dtype=str)
    if df.shape[1] < 6:
        raise ValueError(
            "Ensembl2Reactome.txt format unexpected (expected >=6 columns)"
        )

    df = df.iloc[:, :6].copy()
    df.columns = [
        "ensembl",
        "pathway_id",
        "reactome_url",
        "pathway_name",
        "evidence",
        "species",
    ]

    for column in [
        "ensembl",
        "pathway_id",
        "pathway_name",
        "species",
    ]:
        df[column] = df[column].astype(str).str.strip()

    df = df[
        df["species"].str.lower().isin(
            [
                "homo sapiens",
                "homo\u00a0sapiens",
                "homo sapiens (human)",
            ]
        )
    ]
    df = df[
        (df["ensembl"] != "") & (df["pathway_id"] != "")
    ]
    df = df.drop_duplicates(subset=["ensembl", "pathway_id"])

    if df.empty:
        raise ValueError(
            "Reactome input contained no usable human pathway rows."
        )

    return df[["ensembl", "pathway_id", "pathway_name"]]


def map_symbols_to_ensembl(
    symbols: Iterable[str],
    cache_path: Path,
    species: str = "human",
) -> Dict[str, str]:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache: Dict[str, str] = {}
    if cache_path.exists():
        cache_df = pd.read_csv(cache_path, sep="\t", dtype=str)
        required = {"symbol", "ensembl"}
        if not required.issubset(cache_df.columns):
            raise ValueError(
                "Symbol cache must contain columns 'symbol' and 'ensembl'."
            )
        cache = dict(
            zip(\n                cache_df["symbol"],\n                cache_df["ensembl"],\n                strict=True,\n            )
        )

    requested = sorted(
        {symbol for symbol in symbols if symbol}
    )
    missing = [
        symbol
        for symbol in requested
        if symbol not in cache
    ]

    if missing:
        mg = mygene.MyGeneInfo()
        batch_size = 1000

        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            result = mg.querymany(
                batch,
                scopes="symbol",
                fields="ensembl.gene",
                species=species,
                as_dataframe=True,
                returnall=False,
                verbose=False,
            )

            if not isinstance(result, pd.DataFrame):
                result = pd.DataFrame(result)

            if "ensembl.gene" in result.columns:
                mapped = result["ensembl.gene"].copy()
                mapped = mapped.apply(
                    lambda value: (
                        value[0]
                        if isinstance(value, list) and value
                        else value
                    )
                )
            elif "ensembl" in result.columns:

                def extract_gene(value: object) -> object:
                    if isinstance(value, dict):
                        return value.get("gene")
                    if (
                        isinstance(value, list)
                        and value
                        and isinstance(value[0], dict)
                    ):
                        return value[0].get("gene")
                    return None

                mapped = result["ensembl"].apply(extract_gene)
            else:
                mapped = pd.Series(
                    index=result.index,
                    data=None,
                    dtype=object,
                )

            for symbol, ensembl in mapped.items():
                if not isinstance(symbol, str):
                    continue
                if ensembl is None or str(ensembl) == "nan":
                    cache[symbol] = ""
                else:
                    cache[symbol] = str(ensembl).split(".")[0]

        out_df = pd.DataFrame({"symbol": sorted(cache)})
        out_df["ensembl"] = out_df["symbol"].map(cache)
        out_df.to_csv(cache_path, sep="\t", index=False)

    return cache


def build_effect_matrix(
    dgidb_df: pd.DataFrame,
    reactome_df: pd.DataFrame,
    *,
    top_drugs: int = 60,
    top_pathways: int = 40,
    symbol_cache_path: Path = Path(
        "data/processed/symbol_to_ensembl.tsv"
    ),
) -> EffectMatrix:
    if top_drugs <= 0:
        raise ValueError("top_drugs must be greater than 0")
    if top_pathways <= 0:
        raise ValueError("top_pathways must be greater than 0")

    sym2ens = map_symbols_to_ensembl(
        dgidb_df["gene_symbol"].unique(),
        cache_path=symbol_cache_path,
    )

    interactions = dgidb_df.copy()
    interactions["ensembl"] = (
        interactions["gene_symbol"].map(sym2ens).fillna("")
    )
    interactions = interactions[
        interactions["ensembl"] != ""
    ].drop_duplicates(subset=["drug", "ensembl"])

    joined = interactions.merge(
        reactome_df,
        on="ensembl",
        how="inner",
    )
    if joined.empty:
        raise ValueError(
            "After symbol-to-Ensembl mapping and Reactome joining, "
            "no rows remained. Check the raw files and symbol cache."
        )

    drug_counts = (
        joined.groupby("drug")["pathway_id"]
        .nunique()
        .sort_values(ascending=False, kind="stable")
    )
    drugs = drug_counts.head(top_drugs).index.tolist()
    joined = joined[joined["drug"].isin(drugs)]

    path_counts = (
        joined.groupby("pathway_id")["drug"]
        .nunique()
        .sort_values(ascending=False, kind="stable")
    )
    pathway_ids = path_counts.head(top_pathways).index.tolist()
    joined = joined[joined["pathway_id"].isin(pathway_ids)]

    path_name = (
        joined.drop_duplicates(subset=["pathway_id"])
        .set_index("pathway_id")["pathway_name"]
        .to_dict()
    )
    pathway_names = [
        path_name.get(pathway_id, pathway_id)
        for pathway_id in pathway_ids
    ]

    pivot = (
        joined.groupby(["drug", "pathway_id"])["ensembl"]
        .nunique()
        .reset_index()
        .pivot(
            index="drug",
            columns="pathway_id",
            values="ensembl",
        )
        .fillna(0.0)
    )
    pivot = pivot.reindex(
        index=drugs,
        columns=pathway_ids,
    ).fillna(0.0)

    matrix = pivot.to_numpy(dtype=np.float32)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    effects = (matrix / row_sums).astype(np.float32)

    return EffectMatrix(
        drug_names=drugs,
        pathway_ids=pathway_ids,
        pathway_names=pathway_names,
        effects=effects,
    )


def save_effects(em: EffectMatrix, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        effects=em.effects.astype(np.float32),
        drug_names=np.asarray(em.drug_names, dtype=str),
        pathway_ids=np.asarray(em.pathway_ids, dtype=str),
        pathway_names=np.asarray(em.pathway_names, dtype=str),
    )
    return path


def load_effects(path: Path) -> EffectMatrix:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        effects = archive["effects"].astype(np.float32)
        drug_names = list(archive["drug_names"].astype(str))
        pathway_ids = list(archive["pathway_ids"].astype(str))
        pathway_names = list(
            archive["pathway_names"].astype(str)
        )

    if effects.shape != (
        len(drug_names),
        len(pathway_names),
    ):
        raise ValueError(
            "Stored effect matrix shape does not match "
            "drug/pathway metadata lengths."
        )

    return EffectMatrix(
        drug_names=drug_names,
        pathway_ids=pathway_ids,
        pathway_names=pathway_names,
        effects=effects,
    )
