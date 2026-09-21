"""
make_replication_data.py

Rebuilds Kroencke & Thimme's three `Date_3Factors_25Returns_{m,q,a}.csv` files
from CURRENT Ken French data, in byte-compatible format, so that their
`replication.py` can be run on a sample that does not stop in 2019.

The original files are NOT documented. Format was recovered by inspection:

    no header row, no index
    col 0      date   -- monthly YYYYMM / quarterly YYYYMM (last month of qtr)
                         / annual YYYY
    cols 1-3   Mkt-RF, SMB, HML      (decimals, not percent)
    cols 4-28  the 25 size x book-to-market portfolios, EXCESS returns,
               in Ken French's native column order (SMALL LoBM ... BIG HiBM)

Time aggregation was recovered by matching their own _a and _q files against
their own _m file to machine precision (max abs error 0.000000 in 1964, 1990
and 2019):

    factors     compound as returns:        prod(1 + f) - 1
    portfolios  compound GROSS, then net:   prod(1 + R) - prod(1 + Rf)

                NOT prod(1 + (R - Rf)) - 1, which is wrong by up to 2.6pp
                per year, and NOT the simple sum, which is wrong by ~4pp.

Quarterly rows are stamped with the quarter's LAST month (196403, 196406, ...).
Only complete quarters / complete calendar years are written.

Output goes to 05_Omnibus/data_updated/ -- it does NOT overwrite the authors'
original files.
"""

import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "00_Data"
OUT = HERE / "data_updated"
START = 196401


def _load():
    ff = pd.read_csv(DATA / "clean_F-F_Research_Data_Factors.csv", parse_dates=["Date"])
    p25 = pd.read_csv(DATA / "clean_25_Portfolios_5x5.csv", parse_dates=["Date"])
    for d in (ff, p25):
        d["ym"] = d.Date.dt.year * 100 + d.Date.dt.month
    ports = [c for c in p25.columns if c not in ("Date", "ym")]
    assert len(ports) == 25, ports
    d = p25.merge(ff[["ym", "Mkt-RF", "SMB", "HML", "RF"]], on="ym", how="inner")
    d = d[d.ym >= START].sort_values("ym").reset_index(drop=True)
    for c in ports + ["Mkt-RF", "SMB", "HML", "RF"]:
        d[c] = d[c] / 100.0          # French publishes percent
    return d, ports


def _aggregate(d, ports, key):
    """key -> Series giving the group label for each month."""
    rows = []
    for g, blk in d.groupby(key, sort=True):
        fac = np.prod(1 + blk[["Mkt-RF", "SMB", "HML"]].values, axis=0) - 1
        gross_rf = np.prod(1 + blk["RF"].values)
        # portfolio EXCESS returns -> gross total returns -> compound -> net
        gross_R = np.prod(1 + blk[ports].values + 0.0, axis=0)
        exc = gross_R - gross_rf
        rows.append(np.concatenate(([g], fac, exc)))
    return np.array(rows)


def build():
    d, ports = _load()
    OUT.mkdir(exist_ok=True)

    # --- monthly: no aggregation, portfolios are already total returns ---
    mon = np.column_stack([
        d.ym.values.astype(float),
        d[["Mkt-RF", "SMB", "HML"]].values,
        d[ports].values - d["RF"].values[:, None],
    ])

    d["_y"] = d.ym // 100
    d["_q"] = d["_y"] * 100 + ((d.ym % 100 - 1) // 3 * 3 + 3)

    qtr = _aggregate(d[d.groupby("_q").ym.transform("size") == 3], ports, "_q")
    ann = _aggregate(d[d.groupby("_y").ym.transform("size") == 12], ports, "_y")

    for name, arr, dfmt in (("m", mon, "%d"), ("q", qtr, "%d"), ("a", ann, "%d")):
        path = OUT / f"Date_3Factors_25Returns_{name}.csv"
        with open(path, "w") as fh:
            for row in arr:
                fh.write(dfmt % int(row[0]) + "," +
                         ",".join(repr(float(v)) for v in row[1:]) + "\n")
        print(f"{path.name:34s} {arr.shape[0]:4d} rows  "
              f"{int(arr[0,0])}-{int(arr[-1,0])}  {arr.shape[1]} cols")


if __name__ == "__main__":
    build()
