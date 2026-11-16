"""Find devices whose SCADA readings and connectivity-graph status disagree.

The input DataFrame has one row per device, named by ``OMS_NAME``, and a pair
of flag columns for each week:

    WK<n>_VAL   - the device had SCADA values recorded that week
    WK<n>_CONN  - the device was found in the connectivity graph that week

Flags can be bool, 0/1, or strings such as "Y"/"N" or "TRUE"/"FALSE". A
missing value counts as False.

Main entry point: ``analyze_scada_connectivity(df)``. It returns a
``ScadaConnectivityReport`` with these tables:

    device_summary         per-device metrics and flags (every device)
    suspicious             devices with at least one suspicious pattern, ranked
    active_suspicious      suspicious devices whose problem is happening now
    validation_candidates  devices to check by hand to confirm the frame is right
    week_summary           fleet-wide counts per week, with outlier weeks flagged
    data_quality           problems with the frame itself (duplicates, etc.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NAME_COL = "OMS_NAME"
_WEEK_COL_RE = re.compile(r"^WK(\d+)_(VAL|CONN)$", re.IGNORECASE)
_TRUE_STRINGS = {"1", "1.0", "Y", "YES", "T", "TRUE", "ON", "CONNECTED"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_bool(s: pd.Series) -> pd.Series:
    """Convert a flag column to a plain bool Series. Missing values become False."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).ne(0)
    return s.astype("string").str.strip().str.upper().isin(_TRUE_STRINGS).fillna(False).astype(bool)


def _find_week_columns(df: pd.DataFrame) -> tuple[list[int], list[str], list[str], list[str]]:
    """Return the paired week numbers (sorted), their VAL and CONN columns, and any
    columns that have no partner."""
    found: dict[int, dict[str, str]] = {}
    for col in df.columns:
        m = _WEEK_COL_RE.match(str(col))
        if m:
            found.setdefault(int(m.group(1)), {})[m.group(2).upper()] = col
    weeks = sorted(w for w, d in found.items() if {"VAL", "CONN"} <= d.keys())
    unpaired = [c for w, d in found.items() if w not in weeks for c in d.values()]
    return weeks, [found[w]["VAL"] for w in weeks], [found[w]["CONN"] for w in weeks], unpaired


def _on_runs(m: np.ndarray) -> np.ndarray:
    """Count separate runs of True in each row: [1,1,0,1] -> 2."""
    if m.shape[1] == 0:
        return np.zeros(m.shape[0], dtype=int)
    starts = m[:, 0].astype(int)
    if m.shape[1] > 1:
        starts = starts + (m[:, 1:] & ~m[:, :-1]).sum(axis=1)
    return starts


def _trailing_run(m: np.ndarray) -> np.ndarray:
    """Count how many of the most recent weeks in a row are True (columns run oldest to newest)."""
    if m.shape[1] == 0:
        return np.zeros(m.shape[0], dtype=int)
    rev = m[:, ::-1]
    # argmin finds the first False; rows that are all True need the full width.
    first_false = np.argmin(rev, axis=1)
    return np.where(rev.all(axis=1), m.shape[1], first_false)


def _last_change_age(m: np.ndarray) -> np.ndarray:
    """Weeks since the flag last changed (0 = changed going into the latest week).
    A flag that never changed gets -1."""
    if m.shape[1] < 2:
        return np.full(m.shape[0], -1)
    changed = m[:, 1:] != m[:, :-1]                     # change into week j+1
    rev = changed[:, ::-1]
    age = np.argmax(rev, axis=1)
    return np.where(changed.any(axis=1), age, -1)


def _device_metrics(V: np.ndarray, C: np.ndarray) -> dict[str, np.ndarray]:
    """Compute the metric arrays for every device (rows) from oldest-to-newest week matrices."""
    orphan = V & ~C          # readings but not in the connectivity graph
    silent = C & ~V          # in the graph but no readings
    val_runs, conn_runs = _on_runs(V), _on_runs(C)
    val_trail_off, conn_trail_off = _trailing_run(~V), _trailing_run(~C)
    any_v, any_c = V.any(axis=1), C.any(axis=1)
    n = V.shape[1]

    # Readings stopped while the device is still connected: a telemetry outage
    # rather than a removed device.
    lost_val = any_v & (val_trail_off > 0) & C[:, -1] if n else np.zeros_like(any_v)
    lost_conn = any_c & (conn_trail_off > 0) if n else np.zeros_like(any_c)

    return dict(
        weeks_val=V.sum(axis=1),
        weeks_conn=C.sum(axis=1),
        weeks_both=(V & C).sum(axis=1),
        weeks_orphan=orphan.sum(axis=1),
        weeks_silent=silent.sum(axis=1),
        val_on_runs=val_runs,
        conn_on_runs=conn_runs,
        intermittent_val=val_runs >= 2,
        intermittent_conn=conn_runs >= 2,
        lost_readings=lost_val,
        lost_readings_weeks=np.where(lost_val, val_trail_off, 0),
        lost_connectivity=lost_conn,
        lost_connectivity_weeks=np.where(lost_conn, conn_trail_off, 0),
        current_orphan_streak=_trailing_run(orphan),
        val_last_change_age=_last_change_age(V),
        conn_last_change_age=_last_change_age(C),
        never_seen=~any_v & ~any_c,
    )


def _is_suspicious(m: dict[str, np.ndarray]) -> np.ndarray:
    return (
        (m["weeks_orphan"] > 0)
        | m["intermittent_val"]
        | m["intermittent_conn"]
        | m["lost_readings"]
        | m["lost_connectivity"]
    )


# --------------------------------------------------------------------------- #
# Report container
# --------------------------------------------------------------------------- #
@dataclass
class ScadaConnectivityReport:
    device_summary: pd.DataFrame
    suspicious: pd.DataFrame
    active_suspicious: pd.DataFrame
    validation_candidates: pd.DataFrame
    week_summary: pd.DataFrame
    data_quality: list[str] = field(default_factory=list)
    weeks_oldest_to_newest: list[int] = field(default_factory=list)

    def summary(self) -> str:
        n = len(self.device_summary)
        lines = [
            f"Weeks analysed (oldest -> newest): {self.weeks_oldest_to_newest}",
            f"Devices: {n}",
            f"Suspicious devices: {len(self.suspicious)} "
            f"({int(self.suspicious['likely_data_artifact'].sum())} of them likely extract artifacts)",
            f"Actively suspicious devices (latest week): {len(self.active_suspicious)}",
        ]
        bad_weeks = self.week_summary.loc[self.week_summary["anomalous_week"], "week"].tolist()
        if bad_weeks:
            lines.append(f"Fleet-wide outlier weeks (check the extract itself): {bad_weeks}")
        if len(self.suspicious):
            counts = (
                self.suspicious["reasons"].str.split("; ").explode().str.split(" \\(").str[0].value_counts()
            )
            lines.append("Suspicious patterns:")
            lines += [f"  {k}: {v}" for k, v in counts.items()]
        if self.data_quality:
            lines.append("Data-quality notes:")
            lines += [f"  - {q}" for q in self.data_quality]
        if len(self.validation_candidates):
            lines.append("Suggested validation checks:")
            for check, grp in self.validation_candidates.groupby("check_type", sort=False):
                lines.append(f"  {check}: {', '.join(map(str, grp[NAME_COL].head(10)))}")
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.summary()


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def analyze_scada_connectivity(
    df: pd.DataFrame,
    latest_week: str = "highest",
    recent_window: int = 2,
    week_drop_tolerance: float = 0.20,
    n_controls: int = 5,
    n_top_suspects: int = 5,
    random_state: int | None = 0,
) -> ScadaConnectivityReport:
    """Find devices whose SCADA readings and connectivity status are suspicious.

    Parameters
    ----------
    df : DataFrame
        Must contain ``OMS_NAME`` and paired ``WK<n>_VAL`` / ``WK<n>_CONN`` columns.
    latest_week : {"highest", "lowest"}
        Which end of the week numbers is the most recent. Use "highest" if WK1 is
        the oldest week, or "lowest" if WK1 is the most recent (week numbers
        counting backwards).
    recent_window : int
        A change within this many weeks of the latest week counts as current
        ("active"). Older changes are reported as historical.
    week_drop_tolerance : float
        A week whose fleet-wide VAL or CONN count falls more than this fraction
        below the median week is flagged as an outlier week. Such a drop usually
        points to an extract problem, not a device problem.
    n_controls : int
        How many known-good and known-negative control devices to suggest checking.
    n_top_suspects : int
        How many of the top suspects in each category to suggest checking.
    random_state : int or None
        Seed used to choose control devices, so repeated runs pick the same ones.

    Returns
    -------
    ScadaConnectivityReport
    """
    if NAME_COL not in df.columns:
        raise KeyError(f"DataFrame must contain a '{NAME_COL}' column")
    if latest_week not in ("highest", "lowest"):
        raise ValueError("latest_week must be 'highest' or 'lowest'")

    data_quality: list[str] = []
    weeks, val_cols, conn_cols, unpaired = _find_week_columns(df)
    if not weeks:
        raise ValueError("No paired WK<n>_VAL / WK<n>_CONN columns found")
    if unpaired:
        data_quality.append(f"Ignored week columns that have no VAL/CONN partner: {sorted(unpaired)}")
    gaps = sorted(set(range(weeks[0], weeks[-1] + 1)) - set(weeks))
    if gaps:
        data_quality.append(f"Week numbers missing from the frame: {gaps}")

    if latest_week == "lowest":
        weeks, val_cols, conn_cols = weeks[::-1], val_cols[::-1], conn_cols[::-1]

    # ---- normalise ---------------------------------------------------------
    work = pd.DataFrame({NAME_COL: df[NAME_COL]})
    missing_names = work[NAME_COL].isna() | work[NAME_COL].astype("string").str.strip().eq("")
    if missing_names.any():
        data_quality.append(f"{int(missing_names.sum())} row(s) have no OMS_NAME and were dropped")
    for w, vc, cc in zip(weeks, val_cols, conn_cols):
        work[f"V{w}"] = _to_bool(df[vc])
        work[f"C{w}"] = _to_bool(df[cc])
    work = work.loc[~missing_names]

    dup_mask = work[NAME_COL].duplicated(keep=False)
    duplicated_names = sorted(work.loc[dup_mask, NAME_COL].unique().tolist(), key=str)
    if duplicated_names:
        data_quality.append(
            f"{len(duplicated_names)} OMS_NAME value(s) appear on more than one row; their flags were "
            f"combined with OR. Examples: {duplicated_names[:10]}"
        )
        work = work.groupby(NAME_COL, sort=False, as_index=False).max()

    names = work[NAME_COL].to_numpy()
    V = work[[f"V{w}" for w in weeks]].to_numpy(dtype=bool)
    C = work[[f"C{w}" for w in weeks]].to_numpy(dtype=bool)

    # ---- fleet-wide per-week check -----------------------------------------
    week_summary = pd.DataFrame(
        {
            "week": weeks,
            "devices_val": V.sum(axis=0),
            "devices_conn": C.sum(axis=0),
            "devices_both": (V & C).sum(axis=0),
            "devices_orphan": (V & ~C).sum(axis=0),
            "devices_silent": (C & ~V).sum(axis=0),
        }
    )
    med_v, med_c = week_summary["devices_val"].median(), week_summary["devices_conn"].median()
    med_o = week_summary["devices_orphan"].median()
    week_summary["anomalous_week"] = (
        (week_summary["devices_val"] < (1 - week_drop_tolerance) * med_v)
        | (week_summary["devices_conn"] < (1 - week_drop_tolerance) * med_c)
        | (week_summary["devices_orphan"] > max(2 * med_o, med_o + 0.05 * len(names)))
    )
    if len(weeks) < 3:
        week_summary["anomalous_week"] = False  # a median of 1-2 weeks tells us nothing
    bad_idx = np.flatnonzero(week_summary["anomalous_week"].to_numpy())

    # ---- per-device metrics -------------------------------------------------
    m = _device_metrics(V, C)
    suspicious = _is_suspicious(m)

    # Check again with the outlier weeks removed. A device that only looks
    # suspicious because of those weeks is probably an extract problem.
    if len(bad_idx) and len(bad_idx) < len(weeks):
        keep = np.setdiff1d(np.arange(len(weeks)), bad_idx)
        suspicious_clean = _is_suspicious(_device_metrics(V[:, keep], C[:, keep]))
        likely_artifact = suspicious & ~suspicious_clean
    else:
        likely_artifact = np.zeros(len(names), dtype=bool)

    latest_v, latest_c = V[:, -1], C[:, -1]
    active_orphan = latest_v & ~latest_c
    recent_val_flip = m["intermittent_val"] & (m["val_last_change_age"].clip(min=0) < recent_window) & (m["val_last_change_age"] >= 0)
    recent_conn_flip = m["intermittent_conn"] & (m["conn_last_change_age"].clip(min=0) < recent_window) & (m["conn_last_change_age"] >= 0)
    recent_lost_val = m["lost_readings"] & (m["lost_readings_weeks"] <= recent_window)
    recent_lost_conn = m["lost_connectivity"] & (m["lost_connectivity_weeks"] <= recent_window)
    active = (
        suspicious
        & ~likely_artifact
        & (active_orphan | recent_val_flip | recent_conn_flip | recent_lost_val | recent_lost_conn)
    )

    # ---- explanations and score ---------------------------------------------
    reasons: list[str] = []
    for i in range(len(names)):
        r = []
        if m["weeks_orphan"][i]:
            r.append(f"readings while not connected ({m['weeks_orphan'][i]} wk"
                     + (f", ongoing {m['current_orphan_streak'][i]} wk" if active_orphan[i] else "") + ")")
        if m["intermittent_val"][i]:
            r.append(f"intermittent readings ({m['val_on_runs'][i]} separate on-periods)")
        if m["intermittent_conn"][i]:
            r.append(f"intermittent connectivity ({m['conn_on_runs'][i]} separate on-periods)")
        if m["lost_readings"][i]:
            r.append(f"readings stopped while still connected (last {m['lost_readings_weeks'][i]} wk)")
        if m["lost_connectivity"][i]:
            r.append(f"dropped from connectivity graph (last {m['lost_connectivity_weeks'][i]} wk)")
        if likely_artifact[i]:
            r.append("only suspicious in fleet-wide outlier weeks (likely extract issue)")
        reasons.append("; ".join(r))

    score = (
        3.0 * m["weeks_orphan"]
        + 4.0 * active_orphan
        + 2.0 * np.clip(m["val_on_runs"] - 1, 0, None)
        + 2.0 * np.clip(m["conn_on_runs"] - 1, 0, None)
        + 2.0 * m["lost_readings"]
        + 1.5 * m["lost_connectivity"]
        + 3.0 * active
    ) * np.where(likely_artifact, 0.25, 1.0)

    def _pattern(i: int) -> str:
        v, c = V[i], C[i]
        return "".join("B" if a and b else "V" if a else "C" if b else "." for a, b in zip(v, c))

    device_summary = pd.DataFrame({NAME_COL: names, **m})
    device_summary.insert(1, "pattern", [_pattern(i) for i in range(len(names))])
    device_summary["latest_val"] = latest_v
    device_summary["latest_conn"] = latest_c
    device_summary["suspicious"] = suspicious
    device_summary["active_suspicious"] = active
    device_summary["likely_data_artifact"] = likely_artifact
    device_summary["suspicion_score"] = score.round(2)
    device_summary["reasons"] = reasons
    device_summary["is_duplicate_name"] = device_summary[NAME_COL].isin(duplicated_names)

    out_cols = [NAME_COL, "pattern", "suspicion_score", "active_suspicious", "likely_data_artifact",
                "latest_val", "latest_conn", "weeks_val", "weeks_conn", "weeks_orphan", "reasons"]
    ranked = device_summary.sort_values(["suspicion_score", NAME_COL], ascending=[False, True])
    suspicious_df = ranked.loc[ranked["suspicious"], out_cols].reset_index(drop=True)
    active_df = ranked.loc[ranked["active_suspicious"], out_cols].reset_index(drop=True)

    # ---- validation candidates ------------------------------------------------
    validation = _validation_candidates(
        device_summary, V, C, weeks, bad_idx, duplicated_names, n_controls, n_top_suspects, random_state
    )

    return ScadaConnectivityReport(
        device_summary=device_summary,
        suspicious=suspicious_df,
        active_suspicious=active_df,
        validation_candidates=validation,
        week_summary=week_summary,
        data_quality=data_quality,
        weeks_oldest_to_newest=list(weeks),
    )


def _validation_candidates(
    ds: pd.DataFrame,
    V: np.ndarray,
    C: np.ndarray,
    weeks: list[int],
    bad_idx: np.ndarray,
    duplicated_names: list,
    n_controls: int,
    n_top: int,
    random_state: int | None,
) -> pd.DataFrame:
    """Pick devices whose ground truth is easy to check (in the SCADA historian and
    the connectivity model), so a person can confirm whether the frame is right."""
    rows: list[dict] = []

    def add(mask: np.ndarray | pd.Series, check: str, why: str, verify: str, n: int, by_score: bool = False):
        sub = ds.loc[np.asarray(mask)]
        if sub.empty or n <= 0:
            return
        if by_score:
            sub = sub.sort_values(["suspicion_score", NAME_COL], ascending=[False, True]).head(n)
        else:
            sub = sub.sample(n=min(n, len(sub)), random_state=random_state).sort_values(NAME_COL)
        for name, pat in zip(sub[NAME_COL], sub["pattern"]):
            rows.append({NAME_COL: name, "check_type": check, "pattern": pat, "why": why, "what_to_verify": verify})

    all_both = (V & C).all(axis=1)
    always_silent = (C & ~V).all(axis=1)

    # Top suspects: confirm the anomaly is real.
    add(ds["active_suspicious"] & (ds["latest_val"] & ~ds["latest_conn"]), "top_suspect_orphan",
        "Has readings this week but is not in the connectivity graph.",
        "Look the device up in the network model. Is it energized and connected, or is the OMS_NAME mapping wrong?", n_top, True)
    add((ds["intermittent_val"] | ds["intermittent_conn"]) & ~ds["likely_data_artifact"], "top_suspect_intermittent",
        "Flags switch on and off from week to week.",
        "Pull the raw SCADA trend and graph snapshots for the weeks with no data. Is the device flapping, or is the extract dropping it?", n_top, True)
    add((ds["lost_readings"] | ds["lost_connectivity"]) & ~ds["likely_data_artifact"], "top_suspect_dropped",
        "Readings or connectivity stopped recently.",
        "Check for work orders, decommissioning or RTU/comms outages around the week it stopped.", n_top, True)

    # Controls: confirm the pipeline gets the easy cases right.
    add(all_both, "positive_control",
        "Has readings and is connected in every week, so it should be correct.",
        "Confirm the device has SCADA points and appears in the connectivity graph. If not, the join or flag logic is wrong.",
        n_controls)
    add(always_silent, "negative_control_no_scada",
        "Connected in every week but never has readings.",
        "Confirm the device really has no SCADA point. If it does, VAL is not joining for it (name mismatch).",
        n_controls)
    add(ds["never_seen"], "never_seen",
        "On the list but has no readings and no connectivity in any week.",
        "Check why the row is in the frame at all (stale device list, retired asset, name typo).",
        n_controls)

    # Outlier weeks: one device that changed only in an outlier week shows
    # whether the problem is in the extract.
    if len(bad_idx):
        bad_weeks = [weeks[i] for i in bad_idx]
        add(ds["likely_data_artifact"], "outlier_week_check",
            f"Only suspicious in fleet-wide outlier week(s) {bad_weeks}.",
            "Compare that week's raw source extract with the frame. A fleet-wide drop usually means a failed load.",
            n_controls)

    for name in duplicated_names[:n_controls]:
        rows.append({NAME_COL: name, "check_type": "duplicate_name",
                     "pattern": ds.loc[ds[NAME_COL] == name, "pattern"].iloc[0],
                     "why": "OMS_NAME appears on more than one row.",
                     "what_to_verify": "Decide whether these are separate devices with the same name or duplicate rows."})

    return pd.DataFrame(rows, columns=[NAME_COL, "check_type", "pattern", "why", "what_to_verify"])


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    demo = pd.DataFrame(
        {
            "OMS_NAME": ["SW-100", "SW-101", "XFMR-7", "REC-22", "CB-9", "FUSE-3", "SW-200", "SW-100"],
            "WK1_VAL":  [1, 1, 1, 1, 1, 0, 0, 1],
            "WK1_CONN": [1, 1, 1, 1, 1, 1, 0, 1],
            "WK2_VAL":  [1, 1, 0, 1, 1, 0, 0, 1],
            "WK2_CONN": [1, 1, 1, 1, 1, 1, 0, 1],
            "WK3_VAL":  [1, 1, 1, 1, 1, 0, 0, 1],
            "WK3_CONN": [1, 1, 1, 0, 1, 1, 0, 1],
            "WK4_VAL":  [1, 1, 0, 1, 0, 0, 0, 1],
            "WK4_CONN": [1, 0, 1, 0, 1, 1, 0, 1],
        }
    )
    report = analyze_scada_connectivity(demo)
    print(report.summary())
    print()
    print(report.active_suspicious.to_string(index=False))
