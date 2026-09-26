#!/usr/bin/env python3
"""Rank PSC pathway overlaps and drug records from local research inputs.

The scores in this script are pathway-overlap research signals. They are not
clinical efficacy estimates or treatment recommendations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_psc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype=str,
    )
    required = {
        "gene_symbol",
        "ENSG",
    }
    if not required.issubset(
        df.columns
    ):
        raise ValueError(
            "PSC file must contain columns "
            f"{required}. Found: {set(df.columns)}"
        )

    df["gene_symbol"] = (
        df["gene_symbol"]
        .str.strip()
        .str.upper()
    )
    df["ENSG"] = (
        df["ENSG"]
        .str.strip()
        .str.replace(
            r"\.\d+$",
            "",
            regex=True,
        )
    )
    df = df.dropna(
        subset=[
            "gene_symbol",
            "ENSG",
        ]
    )
    df = df[
        df["ENSG"].str.startswith(
            "ENSG",
            na=False,
        )
    ]
    return df


def load_hgnc_sym2ensg(
    path: Path,
) -> dict[str, str]:
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
    )
    required = {
        "symbol",
        "ensembl_gene_id",
    }
    if not required.issubset(
        frame.columns
    ):
        raise ValueError(
            "HGNC file must contain columns "
            f"{required}."
        )

    frame = frame[
        [
            "symbol",
            "ensembl_gene_id",
        ]
    ].dropna()
    frame["symbol"] = (
        frame["symbol"]
        .str.strip()
        .str.upper()
    )
    frame["ensembl_gene_id"] = (
        frame["ensembl_gene_id"]
        .str.strip()
        .str.replace(
            r"\.\d+$",
            "",
            regex=True,
        )
    )
    frame = frame.drop_duplicates(
        "symbol"
    )
    return dict(
        zip(
            frame["symbol"],
            frame["ensembl_gene_id"],
            strict=True,
        )
    )


def load_reactome_path2genes(
    path: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, set[str]],
]:
    frame = pd.read_csv(
        path,
        sep="\t",
        header=None,
        dtype=str,
        names=[
            "gene_id",
            "pathway_id",
            "url",
            "pathway_name",
            "evidence",
            "species",
        ],
    )
    frame["species"] = (
        frame["species"]
        .str.strip()
        .str.lower()
    )
    frame = frame[
        frame["species"].eq(
            "homo sapiens"
        )
    ].copy()
    frame["gene_id"] = (
        frame["gene_id"]
        .str.strip()
        .str.replace(
            r"\.\d+$",
            "",
            regex=True,
        )
    )
    frame = frame[
        frame["gene_id"].str.startswith(
            "ENSG",
            na=False,
        )
    ].copy()
    frame["pathway_name"] = (
        frame["pathway_name"]
        .str.strip()
    )

    path2genes = (
        frame.groupby("pathway_name")[
            "gene_id"
        ]
        .apply(
            lambda values: set(values)
        )
        .to_dict()
    )
    return frame, path2genes


def load_dgidb_drug2sym(
    path: Path,
) -> dict[str, set[str]]:
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
    )
    required = {
        "drug_name",
        "gene_name",
    }
    if not required.issubset(
        frame.columns
    ):
        raise ValueError(
            "DGIdb file must contain columns "
            f"{required}."
        )

    frame = frame[
        [
            "drug_name",
            "gene_name",
        ]
    ].dropna()
    frame["drug_name"] = (
        frame["drug_name"].str.strip()
    )
    frame["gene_name"] = (
        frame["gene_name"]
        .str.strip()
        .str.upper()
    )
    return (
        frame.groupby("drug_name")[
            "gene_name"
        ]
        .apply(
            lambda values: set(values)
        )
        .to_dict()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute PSC pathway-overlap research "
            "signals from local data."
        )
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/raw"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "artifacts/psc_from_wes"
        ),
    )
    parser.add_argument(
        "--psc_file",
        type=str,
        default="PSC_risk_genes.csv",
    )
    parser.add_argument(
        "--dgidb_file",
        type=str,
        default="dgidb_interactions.tsv",
    )
    parser.add_argument(
        "--reactome_file",
        type=str,
        default="Ensembl2Reactome.txt",
    )
    parser.add_argument(
        "--hgnc_file",
        type=str,
        default="hgnc_complete_set.tsv",
    )
    parser.add_argument(
        "--top_pathways",
        type=int,
        default=50,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.top_pathways <= 0:
        raise ValueError(
            "top_pathways must be greater than 0"
        )

    args.outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    psc = load_psc(
        args.raw_dir / args.psc_file
    )
    psc_symbols = set(
        psc["gene_symbol"]
    )
    psc_ensembl = set(
        psc["ENSG"]
    )

    print(
        "[ok] PSC genes: "
        f"{len(psc)} rows | "
        f"symbols: {len(psc_symbols)} | "
        f"ENSG: {len(psc_ensembl)}"
    )

    reactome_frame, path2genes = (
        load_reactome_path2genes(
            args.raw_dir
            / args.reactome_file
        )
    )
    print(
        "[ok] Reactome human ENSG rows: "
        f"{len(reactome_frame):,} | "
        f"pathways: {len(path2genes):,}"
    )

    pathway_rows = []
    for pathway, genes in path2genes.items():
        overlap = genes & psc_ensembl
        if overlap:
            pathway_rows.append(
                {
                    "pathway": pathway,
                    "n_psc_genes_in_pathway": len(
                        overlap
                    ),
                    "n_pathway_genes": len(
                        genes
                    ),
                    "psc_ensg_genes": ";".join(
                        sorted(overlap)
                    ),
                }
            )

    pathway_frame = pd.DataFrame(
        pathway_rows
    )
    if not pathway_frame.empty:
        pathway_frame = (
            pathway_frame.sort_values(
                [
                    "n_psc_genes_in_pathway",
                    "n_pathway_genes",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
        )

    pathway_output = (
        args.outdir
        / "psc_affected_pathways.csv"
    )
    pathway_frame.to_csv(
        pathway_output,
        index=False,
    )
    print(
        f"[ok] wrote {pathway_output} "
        f"({len(pathway_frame)} pathways)"
    )

    symbol_to_ensembl = (
        load_hgnc_sym2ensg(
            args.raw_dir
            / args.hgnc_file
        )
    )
    drug_to_symbols = (
        load_dgidb_drug2sym(
            args.raw_dir
            / args.dgidb_file
        )
    )

    drug_rows = []
    top_paths = pathway_frame.head(
        args.top_pathways
    )
    pathway_weights = dict(
        zip(
            top_paths["pathway"],
            top_paths[
                "n_psc_genes_in_pathway"
            ].astype(float),
            strict=True,
        )
    )
    top_path_to_genes = {
        pathway: path2genes[pathway]
        for pathway
        in pathway_weights
    }

    for drug, symbols in drug_to_symbols.items():
        targets = {
            symbol_to_ensembl[symbol]
            for symbol in symbols
            if symbol in symbol_to_ensembl
        }
        targets = {
            target
            for target in targets
            if isinstance(target, str)
            and target.startswith("ENSG")
        }
        if not targets:
            continue

        score = 0.0
        hits_total = 0
        for (
            pathway,
            weight,
        ) in pathway_weights.items():
            hit_count = len(
                targets
                & top_path_to_genes[
                    pathway
                ]
            )
            if hit_count:
                score += (
                    weight
                    * (
                        hit_count
                        / len(targets)
                    )
                )
                hits_total += hit_count

        if score > 0:
            drug_rows.append(
                {
                    "drug": drug,
                    "score": score,
                    "n_targets": len(
                        targets
                    ),
                    "total_pathway_hits": (
                        hits_total
                    ),
                }
            )

    drug_frame = pd.DataFrame(
        drug_rows
    )
    if not drug_frame.empty:
        drug_frame = (
            drug_frame.sort_values(
                "score",
                ascending=False,
            )
        )

    drug_output = (
        args.outdir
        / "psc_drug_ranking_from_pathways.csv"
    )
    drug_frame.to_csv(
        drug_output,
        index=False,
    )
    print(
        f"[ok] wrote {drug_output} "
        f"({len(drug_frame)} drugs)"
    )

    if len(pathway_frame):
        print(
            "\nTop pathway-overlap records:"
        )
        print(
            pathway_frame.head(10)[
                [
                    "pathway",
                    "n_psc_genes_in_pathway",
                    "n_pathway_genes",
                ]
            ].to_string(
                index=False
            )
        )

    if len(drug_frame):
        print(
            "\nTop drug/pathway-overlap records "
            "(research signal only; not a treatment "
            "recommendation):"
        )
        print(
            drug_frame.head(15)[
                [
                    "drug",
                    "score",
                    "n_targets",
                    "total_pathway_hits",
                ]
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
