#!/usr/bin/env python3
"""Rank PSC-affected pathways and drugs from local raw files."""

from pathlib import Path

import pandas as pd

RAW = Path("./data/raw")
OUT = Path("./artifacts/psc_from_wes")
OUT.mkdir(parents=True, exist_ok=True)
PSC_FILE = RAW / "PSC_risk_genes.csv"
DGIDB_FILE = RAW / "dgidb_interactions.tsv"
REACTOME_FILE = RAW / "Ensembl2Reactome.txt"
HGNC_FILE = RAW / "hgnc_complete_set.tsv"


def load_psc():
    df = pd.read_csv(PSC_FILE, dtype=str)
    need = {"gene_symbol", "ENSG"}
    if not need.issubset(df.columns):
        raise ValueError(f"PSC file must contain columns {need}. Found: {set(df.columns)}")
    df["gene_symbol"] = df["gene_symbol"].str.strip().str.upper()
    df["ENSG"] = df["ENSG"].str.strip().str.replace(r"\.\d+$", "", regex=True)
    df = df.dropna(subset=["gene_symbol", "ENSG"])
    df = df[df["ENSG"].str.startswith("ENSG", na=False)]
    return df


def load_hgnc_sym2ensg():
    h = pd.read_csv(HGNC_FILE, sep="\t", dtype=str)
    h = h[["symbol", "ensembl_gene_id"]].dropna()
    h["symbol"] = h["symbol"].str.strip().str.upper()
    h["ensembl_gene_id"] = h["ensembl_gene_id"].str.strip().str.replace(r"\.\d+$", "", regex=True)
    h = h.drop_duplicates("symbol")
    return dict(zip(h["symbol"], h["ensembl_gene_id"]))


def load_reactome_path2genes():
    re_df = pd.read_csv(
        REACTOME_FILE,
        sep="\t",
        header=None,
        dtype=str,
        names=["gene_id", "pathway_id", "url", "pathway_name", "evidence", "species"],
    )
    re_df["species"] = re_df["species"].str.strip().str.lower()
    re_df = re_df[re_df["species"].eq("homo sapiens")].copy()
    re_df["gene_id"] = re_df["gene_id"].str.strip().str.replace(r"\.\d+$", "", regex=True)
    re_df = re_df[re_df["gene_id"].str.startswith("ENSG", na=False)].copy()
    re_df["pathway_name"] = re_df["pathway_name"].str.strip()
    path2genes = re_df.groupby("pathway_name")["gene_id"].apply(lambda x: set(x)).to_dict()
    return re_df, path2genes


def load_dgidb_drug2sym():
    dg = pd.read_csv(DGIDB_FILE, sep="\t", dtype=str)
    dg = dg[["drug_name", "gene_name"]].dropna()
    dg["drug_name"] = dg["drug_name"].str.strip()
    dg["gene_name"] = dg["gene_name"].str.strip().str.upper()
    return dg.groupby("drug_name")["gene_name"].apply(lambda x: set(x)).to_dict()


def main():
    psc = load_psc()
    psc_syms = set(psc["gene_symbol"])
    psc_ensg = set(psc["ENSG"])
    print(f"[ok] PSC genes: {len(psc)} rows | symbols: {len(psc_syms)} | ENSG: {len(psc_ensg)}")

    re_df, path2genes = load_reactome_path2genes()
    print(f"[ok] Reactome human ENSG rows: {len(re_df):,} | pathways: {len(path2genes):,}")

    path_rows = []
    for pathway, genes in path2genes.items():
        overlap = genes & psc_ensg
        if overlap:
            path_rows.append(
                {
                    "pathway": pathway,
                    "n_psc_genes_in_pathway": len(overlap),
                    "n_pathway_genes": len(genes),
                    "psc_ensg_genes": ";".join(sorted(overlap)),
                }
            )
    path_df = pd.DataFrame(path_rows).sort_values(
        ["n_psc_genes_in_pathway", "n_pathway_genes"],
        ascending=[False, True],
    )
    out_path = OUT / "psc_affected_pathways.csv"
    path_df.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path} ({len(path_df)} pathways)")

    sym2ensg = load_hgnc_sym2ensg()
    drug2sym = load_dgidb_drug2sym()
    drug_rows = []
    top_paths = path_df.head(50)
    p_weight = dict(zip(top_paths["pathway"], top_paths["n_psc_genes_in_pathway"].astype(float)))
    top_path2genes = {p: path2genes[p] for p in p_weight.keys()}

    for drug, syms in drug2sym.items():
        targets = {sym2ensg[s] for s in syms if s in sym2ensg}
        targets = {t for t in targets if isinstance(t, str) and t.startswith("ENSG")}
        if not targets:
            continue
        score = 0.0
        hits_total = 0
        for pathway, weight in p_weight.items():
            hit = len(targets & top_path2genes[pathway])
            if hit:
                score += weight * (hit / len(targets))
                hits_total += hit
        if score > 0:
            drug_rows.append(
                {
                    "drug": drug,
                    "score": score,
                    "n_targets": len(targets),
                    "total_pathway_hits": hits_total,
                }
            )

    drug_df = pd.DataFrame(drug_rows).sort_values("score", ascending=False)
    out_drug = OUT / "psc_drug_ranking_from_pathways.csv"
    drug_df.to_csv(out_drug, index=False)
    print(f"[ok] wrote {out_drug} ({len(drug_df)} drugs)")

    if len(path_df):
        print("\nTop affected pathways:")
        print(path_df.head(10)[["pathway", "n_psc_genes_in_pathway", "n_pathway_genes"]].to_string(index=False))
    if len(drug_df):
        print("\nTop candidate drugs:")
        print(drug_df.head(15)[["drug", "score", "n_targets", "total_pathway_hits"]].to_string(index=False))


if __name__ == "__main__":
    main()
