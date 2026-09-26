# Data inputs and provenance

RL-Path does not commit the external biological datasets used to build the
drug-to-pathway matrix.

## Default raw inputs

Place these files under `data/raw/` unless you pass another `--raw_dir`.

### DGIdb

Expected default filename:

```text
dgidb_interactions.tsv
```

The preprocessing code looks for drug and gene-symbol columns using common DGIdb
column-name variants.

### Reactome

Expected default filename:

```text
Ensembl2Reactome.txt
```

The loader expects the tab-delimited Ensembl-to-Reactome mapping format and keeps
human rows.

## Identifier mapping

DGIdb records are generally gene-symbol based while the Reactome file used here is
Ensembl based. `mygene` is used to create a local symbol-to-Ensembl cache under the
processed-data directory.

Because external databases and identifier mappings change over time, publication
or benchmark runs should record:

- database/source name;
- release/version when available;
- retrieval date;
- exact filename;
- checksum;
- any filtering performed before use.

## Processed output

Training creates a file such as:

```text
data/processed/drug_pathway_effects_N60_P40.npz
```

The archive contains:

- normalized effect matrix;
- drug names;
- Reactome pathway IDs;
- pathway names.

The current format stores string arrays without Python object pickles.

## Research interpretation

The processed matrix represents normalized pathway coverage derived from mapped
drug-gene records. It is an input to a simulator and should not be interpreted as a
measured pharmacological effect matrix.

## PSC-oriented optional inputs

`scripts/psc_pathways_and_drugs.py` can additionally use local files such as:

- `PSC_risk_genes.csv`;
- `hgnc_complete_set.tsv`;
- the DGIdb and Reactome files above.

Those files are not supplied by this repository. Their provenance and versions
must be documented by the user running the analysis.
