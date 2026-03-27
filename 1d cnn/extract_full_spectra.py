"""
Extract the COMPLETE raw_spectra DataFrame with all metadata labels.

Structure of raw_spectra.rda:
  col[0]     : STR  - polymer/material label (e.g. "PE", "PP", ...)
  col[1]     : VEC  - nested 1-column data.frame with column "ID" (e.g. "P")
  col[2-257] : REAL - 256 spectral intensity values per sample
  Total rows : 11,885 samples

pyreadr cannot parse this file (unsupported R serialization format).
rdata.read_rda() fails because col[1] is a nested data.frame that
pandas cannot coerce. We use rdata.parser + manual extraction.
"""

import numpy as np
import pandas as pd
import rdata

RDA_PATH = "../raw_spectra.rda"
OUT_CSV  = "../full_spectra_with_labels.csv"

# ── 1. Parse the raw R object ───────────────────────────────────────────────
print("Parsing raw_spectra.rda ...")
parsed   = rdata.parser.parse_file(RDA_PATH)
data_obj = parsed.object.value[0]
col_nodes = data_obj.value

# ── Helper: walk a pairlist and return a dict of {tag_str: value_RObject} ────
def pairlist_to_dict(node):
    result = {}
    while node is not None:
        # extract tag name
        tag_v = node.tag
        while hasattr(tag_v, 'value'):
            tag_v = tag_v.value
        tag_str = tag_v.decode() if isinstance(tag_v, bytes) else str(tag_v) if tag_v else None

        if tag_str and hasattr(node, 'value') and node.value:
            result[tag_str] = node.value[0]
            if len(node.value) > 1:
                node = node.value[1]
                continue
        break
    return result

# ── Helper: extract a string list from an STR RObject ────────────────────────
def str_robject_to_list(robj):
    out = []
    for item in robj.value:
        v = item.value if hasattr(item, 'value') else item
        if isinstance(v, bytes):
            out.append(v.decode())
        else:
            out.append(str(v))
    return out

# ── 2. Get parent data.frame column names ────────────────────────────────────
parent_attrs = pairlist_to_dict(data_obj.attributes)
if 'names' in parent_attrs:
    col_names = str_robject_to_list(parent_attrs['names'])
else:
    col_names = [f"V{i}" for i in range(len(col_nodes))]

print(f"Found {len(col_nodes)} columns, {len(col_names)} names")

# ── 3. Build columns dict ───────────────────────────────────────────────────
columns = {}

for i, cn in enumerate(col_nodes):
    cname = col_names[i] if i < len(col_names) else f"V{i}"
    rtype = cn.info.type.name  # e.g. 'REAL', 'STR', 'VEC', 'INT'

    if isinstance(cn.value, np.ndarray):
        # Numeric column
        columns[cname] = cn.value

    elif rtype == "STR" and isinstance(cn.value, list):
        # Character vector
        columns[cname] = str_robject_to_list(cn)

    elif rtype == "VEC" and isinstance(cn.value, list):
        # Nested data.frame or list — flatten its inner columns
        inner_attrs = pairlist_to_dict(cn.attributes) if cn.info.attributes else {}
        inner_names = str_robject_to_list(inner_attrs['names']) if 'names' in inner_attrs else []

        for j, inner_cn in enumerate(cn.value):
            inner_cname = inner_names[j] if j < len(inner_names) else f"{cname}.{j}"
            inner_rtype = inner_cn.info.type.name if hasattr(inner_cn, 'info') else ''

            if isinstance(inner_cn.value, np.ndarray):
                columns[inner_cname] = inner_cn.value
            elif inner_rtype == "STR" and isinstance(inner_cn.value, list):
                columns[inner_cname] = str_robject_to_list(inner_cn)
            else:
                columns[inner_cname] = [str(inner_cn.value)] * 11885  # fallback

    elif rtype == "INT" and isinstance(cn.value, np.ndarray):
        # Integer factor — check for levels
        if cn.info.attributes:
            cat_attrs = pairlist_to_dict(cn.attributes)
            if 'levels' in cat_attrs:
                levels = str_robject_to_list(cat_attrs['levels'])
                columns[cname] = [levels[idx - 1] if 1 <= idx <= len(levels)
                                  else None for idx in cn.value]
            else:
                columns[cname] = cn.value
        else:
            columns[cname] = cn.value
    else:
        columns[cname] = [str(cn.value)]

# ── 4. Build DataFrame ──────────────────────────────────────────────────────
raw_spectra_df = pd.DataFrame(columns)

# ── 5. Print exploration info ────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Total number of columns: {len(raw_spectra_df.columns)}")
print(f"DataFrame shape        : {raw_spectra_df.shape}")
print(f"{'='*60}")

print(f"\nFirst 5 column names:")
for c in raw_spectra_df.columns[:5]:
    print(f"  - {c}  (dtype: {raw_spectra_df[c].dtype})")

print(f"\nLast 10 column names:")
for c in raw_spectra_df.columns[-10:]:
    print(f"  - {c}  (dtype: {raw_spectra_df[c].dtype})")

print(f"\nFirst 10 Row Names (DataFrame index):")
for idx_val in raw_spectra_df.index[:10]:
    print(f"  {idx_val}")

# Non-numeric columns (labels/metadata)
non_numeric = raw_spectra_df.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"\nNon-numeric (metadata/label) columns: {non_numeric}")
    for col in non_numeric:
        uvals = raw_spectra_df[col].nunique()
        examples = raw_spectra_df[col].unique()[:10].tolist()
        print(f"  '{col}': {uvals} unique values -> {examples}")

# ── 6. Save complete DataFrame ──────────────────────────────────────────────
raw_spectra_df.to_csv(OUT_CSV, index=True)
print(f"\nSaved -> {OUT_CSV}  ({raw_spectra_df.shape[0]} rows x {raw_spectra_df.shape[1]} cols)")
print("Done!")
