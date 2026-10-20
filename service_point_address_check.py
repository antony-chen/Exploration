import re
import pandas as pd

SUFFIXES = {"STREET": "ST", "AVENUE": "AVE", "ROAD": "RD", "DRIVE": "DR", "BOULEVARD": "BLVD",
            "LANE": "LN", "COURT": "CT", "PLACE": "PL", "CIRCLE": "CIR", "PARKWAY": "PKWY"}
DIRS = {"NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W"}

def parse_address(addr):
    """'101-B N Main Street Apt 4' -> (101, 'N MAIN ST')"""
    a = re.sub(r"[.,#]", " ", str(addr).upper())
    a = re.sub(r"\b(APT|UNIT|STE|SUITE|BLDG|FL)\b.*$", "", a)       # drop unit designators
    m = re.match(r"\s*(\d+)(?:-?[A-Z]\b|-\s*\d+)?\s+(.*)", a)  # number, optional 101A / 101-105
    if not m:
        return pd.Series([pd.NA, a.strip()])
    words = [SUFFIXES.get(w, DIRS.get(w, w)) for w in m.group(2).split()]
    return pd.Series([int(m.group(1)), " ".join(words)])

def summarize(g):
    nums = g["street_num"].dropna().astype(int).sort_values().unique()
    gaps = pd.Series(nums).diff().dropna()
    return pd.Series({
        "n_addresses":    len(g),
        "n_streets":      g["street_name"].nunique(),
        "streets":        "; ".join(sorted(g["street_name"].unique())),
        "min_num":        nums.min() if len(nums) else pd.NA,
        "max_num":        nums.max() if len(nums) else pd.NA,
        "span":           nums.max() - nums.min() if len(nums) else pd.NA,
        "max_gap":        gaps.max() if len(gaps) else 0,
        "same_parity":    len({n % 2 for n in nums}) <= 1,
    })

def check_service_points(df, sp_col="service_point_id", addr_col="address", max_gap=10):
    df = df.copy()
    df[["street_num", "street_name"]] = df[addr_col].apply(parse_address)
    out = df.groupby(sp_col)[["street_num", "street_name"]].apply(summarize)
    out = out[out["n_addresses"] > 1]                                   # multi-address SPs only
    out["same_street"] = out["n_streets"] == 1
    out["numbers_close"] = out["max_gap"] <= max_gap
    out["verdict"] = "REVIEW"
    out.loc[out.same_street & out.numbers_close, "verdict"] = "LIKELY SAME PROPERTY"
    out.loc[~out.same_street, "verdict"] = "DIFFERENT STREETS (corner lot?)"
    out.loc[out.same_street & ~out.numbers_close, "verdict"] = "SAME STREET, FAR APART"
    return df, out.sort_values(["verdict", "n_addresses"], ascending=[True, False])

if __name__ == "__main__":
    df = pd.DataFrame({
        "service_point_id": ["SP1","SP1","SP1","SP2","SP2","SP3","SP3","SP4"],
        "address": ["101 Main Street", "103 Main St.", "105-B MAIN ST Apt 2",
                    "200 Oak Ave", "450 Oak Avenue",
                    "12 Elm St", "300 N Pine Rd",
                    "9 Birch Ln"],
    })
    parsed, report = check_service_points(df)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print(parsed); print(); print(report)
