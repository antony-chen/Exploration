"""Train and evaluate global-horizon plus local-correction Direct168 forecasting.

Protocol
--------
* Input load: complete previous week, 168 hours.
* Known future exogenous input: complete forecast week, 168 hours.
* Output: complete forecast week, 168 load values, in one forward pass.
* No actual load from the forecast week is supplied to the model.
* Split: target weeks fully inside Jan-Oct=train, November=validation,
  December=test. Weeks crossing a split boundary are excluded.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, TensorDataset

# Use the corrected data_utils.py shipped beside this Main file.  Resolve the
# project root separately for processed_data and output paths.
THIS_DIR = Path(__file__).resolve().parent


def _resolve_project_root() -> Path:
    configured = os.getenv("ONCOR_PROJECT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in [THIS_DIR, *THIS_DIR.parents]:
        if (candidate / "processed_data").is_dir():
            return candidate.resolve()
    return THIS_DIR.parent.resolve()


PROJECT_ROOT = _resolve_project_root()
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from data_utils import get_data_oncor_load_weekly  # noqa: E402
from Model_168hr_MLP_reverseweek import Direct168GlobalLocalModel, WEEK_HOURS  # noqa: E402


@dataclass
class Config:
    run_name: str = os.getenv(
        "ONCOR_RUN_NAME",
        "168hr_mlp_reverseweek",
    )
    output_root: str = os.getenv(
        "ONCOR_OUTPUT_ROOT",
        str(PROJECT_ROOT / "oncor_reverseweek"),
    )
    profile_ids_env: str = os.getenv("ONCOR_XFMR_IDS", "")
    split_years_env: str = os.getenv("ONCOR_SPLIT_YEARS", "")

    seed: int = int(os.getenv("ONCOR_SEED", "42"))
    batch_size: int = int(os.getenv("ONCOR_BATCH_SIZE", "16"))
    total_updates: int = int(os.getenv("ONCOR_TOTAL_UPDATES", "6000"))
    base_lr: float = float(os.getenv("ONCOR_LR", "3e-4"))
    weight_decay: float = float(os.getenv("ONCOR_WEIGHT_DECAY", "1e-4"))
    grad_clip: float = float(os.getenv("ONCOR_GRAD_CLIP", "1.0"))
    num_workers: int = int(os.getenv("ONCOR_NUM_WORKERS", "0"))

    lr_warmup_updates: int = int(os.getenv("ONCOR_LR_WARMUP_UPDATES", "720"))
    lr_decay_1: int = int(os.getenv("ONCOR_LR_DECAY_UPDATE_1", "1800"))
    lr_decay_2: int = int(os.getenv("ONCOR_LR_DECAY_UPDATE_2", "2700"))
    lr_decay_3: int = int(os.getenv("ONCOR_LR_DECAY_UPDATE_3", "3600"))
    lr_decay_4: int = int(os.getenv("ONCOR_LR_DECAY_UPDATE_4", "4200"))
    kl_anneal_updates: int = int(os.getenv("ONCOR_KL_ANNEAL_UPDATES", "360"))
    moe_warmup_updates: int = int(os.getenv("ONCOR_MOE_WARMUP_UPDATES", "480"))

    xprime_dim: int = 16
    hidden_dim: int = 96
    latent_dim: int = 64
    reverse_hidden_dim: int = 48
    fusion_bottleneck: int = 32
    residual_scale: float = 0.1
    phase_summary_dim: int = int(os.getenv("ONCOR_PHASE_SUMMARY_DIM", "32"))
    future_summary_dim: int = int(os.getenv("ONCOR_FUTURE_SUMMARY_DIM", "32"))
    global_hidden: int = int(os.getenv("ONCOR_GLOBAL_HIDDEN", "64"))
    local_ext_hidden: int = int(os.getenv("ONCOR_LOCAL_EXT_HIDDEN", "24"))
    hourly_hidden: int = int(os.getenv("ONCOR_HOURLY_HIDDEN", "64"))
    local_correction_gain: float = float(os.getenv("ONCOR_LOCAL_CORRECTION_GAIN", "0.25"))
    dropout: float = float(os.getenv("ONCOR_DROPOUT", "0.1"))
    output_len: int = WEEK_HOURS
    output_dim: int = 1
    input_dim: int = 1
    encoder_mode: str = "reverse_week_dual"
    top_k: int = int(os.getenv("ONCOR_TOP_K", "4"))
    logvar_min: float = -10.0
    logvar_max: float = -3.8

    kl_weight: float = float(os.getenv("ONCOR_KL_WEIGHT", "1e-4"))
    mse_weight: float = float(os.getenv("ONCOR_MSE_WEIGHT", "2.0"))
    ramp_weight: float = float(os.getenv("ONCOR_BOUNDARY_RAMP_WEIGHT", "0.5"))
    ramp_hours: int = int(os.getenv("ONCOR_BOUNDARY_RAMP_HOURS", "3"))

    eval_only: bool = os.getenv("ONCOR_EVAL_ONLY", "0") == "1"
    force_retrain: bool = os.getenv("ONCOR_FORCE_RETRAIN", "0") == "1"
    save_predictions: bool = os.getenv("ONCOR_SAVE_PREDICTIONS", "1") == "1"
    require_midnight_origin: bool = os.getenv("ONCOR_REQUIRE_MIDNIGHT_ORIGIN", "1") == "1"

    def profile_ids(self) -> Optional[List[str]]:
        values = [v.strip() for v in self.profile_ids_env.split(",") if v.strip()]
        return values or None

    def split_years(self) -> Optional[List[int]]:
        values = [int(v.strip()) for v in self.split_years_env.split(",") if v.strip()]
        return sorted(set(values)) or None

    def validate(self) -> None:
        if self.output_len != 168:
            raise ValueError("Direct168 requires output_len=168")
        if not (
            0 < self.lr_warmup_updates < self.lr_decay_1 < self.lr_decay_2
            < self.lr_decay_3 < self.lr_decay_4 < self.total_updates
        ):
            raise ValueError("Invalid update schedule")
        if self.ramp_hours < 1 or self.ramp_hours > self.output_len:
            raise ValueError("ramp_hours must be in [1,168]")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _ensure_week_matrix(arr, name: str, dtype=None):
    values = np.asarray(arr, dtype=dtype)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or values.shape[1] != WEEK_HOURS:
        raise ValueError(f"{name} must be [weeks,168], got {values.shape}")
    return values


def split_merged_profiles(times, load, temp, workday, season) -> List[Dict]:
    times = _ensure_week_matrix(times, "times")
    load = _ensure_week_matrix(load, "load", dtype=float)
    temp = _ensure_week_matrix(temp, "temp", dtype=float)
    workday = _ensure_week_matrix(workday, "workday", dtype=float)
    season = _ensure_week_matrix(season, "season", dtype=float)
    if not (times.shape == load.shape == temp.shape == workday.shape == season.shape):
        raise ValueError("Merged weekly arrays must share shape")

    starts, ends = [], []
    for row in times:
        idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(row).reshape(-1)))
        starts.append(idx[0])
        ends.append(idx[-1])

    # Match v9_F: a new profile starts when weekly timestamps reset/overlap.
    boundaries = [0]
    for i in range(1, len(times)):
        if starts[i] <= ends[i - 1]:
            boundaries.append(i)
    boundaries.append(len(times))

    profiles = []
    for idx, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        profiles.append({
            "name": f"profile_{idx}",
            "times": times[a:b],
            "load": load[a:b],
            "temp": temp[a:b],
            "workday": workday[a:b],
            "season": season[a:b],
        })
        print(
            f"[PROFILE] profile_{idx}: weeks={b-a}, "
            f"range={starts[a]} -> {ends[b-1]}"
        )
    if not profiles:
        raise RuntimeError("No profiles recovered from merged weekly arrays")
    print(f"[PROFILE] recovered {len(profiles)} profiles from XFMR='all'.")
    return profiles


def load_profiles(profile_ids: Optional[Sequence[str]]) -> List[Dict]:
    if profile_ids:
        profiles = []
        for value in profile_ids:
            arrays = get_data_oncor_load_weekly(XFMR=str(value))
            t, l, temp, wd, season = arrays
            profiles.append({
                "name": str(value),
                "times": _ensure_week_matrix(t, "times"),
                "load": _ensure_week_matrix(l, "load", dtype=float),
                "temp": _ensure_week_matrix(temp, "temp", dtype=float),
                "workday": _ensure_week_matrix(wd, "workday", dtype=float),
                "season": _ensure_week_matrix(season, "season", dtype=float),
            })
        return profiles
    print("[PROFILE] ONCOR_XFMR_IDS is empty; recovering profiles from XFMR='all'.")
    return split_merged_profiles(*get_data_oncor_load_weekly(XFMR="all"))


def _future_path(values: np.ndarray, horizon: int, include_current: bool) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    start = 0 if include_current else 1
    offsets = np.arange(start, start + horizon)[None, :]
    indices = np.clip(np.arange(len(values))[:, None] + offsets, 0, len(values) - 1)
    return values[indices]


def build_frame(profile: Dict) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(np.asarray(profile["times"]).reshape(-1)))
    load = np.asarray(profile["load"], dtype=float).reshape(-1)
    temp = np.asarray(profile["temp"], dtype=float).reshape(-1)
    workday = np.asarray(profile["workday"], dtype=float).reshape(-1)
    if not (len(index) == len(load) == len(temp) == len(workday)):
        raise ValueError(f"Length mismatch for {profile['name']}")
    if index.has_duplicates:
        raise ValueError(f"Duplicate timestamps for {profile['name']}")
    if not np.isfinite(load).all() or not np.isfinite(temp).all():
        raise ValueError(f"Non-finite load/temperature for {profile['name']}")

    df = pd.DataFrame({"load": load, "temp": temp, "workday": workday}, index=index)
    angle = 2.0 * np.pi * (index.month.to_numpy() - 1) / 12.0
    df["month_sin"] = np.sin(angle)
    df["month_cos"] = np.cos(angle)

    # Use the next 24 temperatures (t+1 ... t+24), aligned with the
    # 24-hour load horizon; workday summary also excludes the current hour.
    temp_path = _future_path(temp, 24, include_current=False)
    for h in range(24):
        df[f"temp_fc_tplus{h:02d}"] = temp_path[:, h]
    wd_path = _future_path(workday, 24, include_current=False)
    df["workday_future24_mean"] = wd_path.mean(axis=1)
    return df


def infer_split_years(frames: Dict[str, pd.DataFrame], requested: Optional[List[int]]) -> List[int]:
    if requested:
        return requested
    complete = []
    for df in frames.values():
        years = set()
        for year, group in df.groupby(df.index.year):
            if set(range(1, 13)).issubset(set(group.index.month.unique())):
                years.add(int(year))
        complete.append(years)
    common = set.intersection(*complete) if complete else set()
    if common:
        return sorted(common)
    return sorted(set(int(y) for df in frames.values() for y in df.index.year.unique()))


def _is_hourly_contiguous(index: pd.DatetimeIndex, expected_len: int) -> bool:
    """Compare timestamp values, independent of pandas datetime unit (s/ns)."""
    idx = pd.DatetimeIndex(index)
    if len(idx) != expected_len:
        return False
    values = idx.to_numpy(dtype="datetime64[ns]")
    if len(values) <= 1:
        return True
    return bool(np.all(np.diff(values) == np.timedelta64(1, "h")))


def classify_target_week(index: pd.DatetimeIndex, split_years: set) -> Optional[str]:
    if not _is_hourly_contiguous(index, WEEK_HOURS):
        return None
    years = set(int(v) for v in index.year)
    if len(years) != 1 or next(iter(years)) not in split_years:
        return None
    months = index.month.to_numpy()
    if np.all((months >= 1) & (months <= 10)):
        return "train"
    if np.all(months == 11):
        return "val"
    if np.all(months == 12):
        return "test"
    return None


def _inverse_params(scaler: MinMaxScaler) -> Tuple[float, float]:
    lo, hi = float(scaler.data_min_[0]), float(scaler.data_max_[0])
    fr_lo, fr_hi = map(float, scaler.feature_range)
    scale = (hi - lo) / max(fr_hi - fr_lo, 1e-12)
    shift = lo - fr_lo * scale
    return scale, shift


def prepare_data(profiles: List[Dict], cfg: Config, device: torch.device) -> Dict:
    profile_names = [str(p["name"]) for p in profiles]
    profile_to_idx = {name: i for i, name in enumerate(profile_names)}
    profile_keys = [f"profile_onehot_{i:02d}" for i in range(len(profile_names))]

    frames = {str(p["name"]): build_frame(p) for p in profiles}
    for name, df in frames.items():
        idx = profile_to_idx[name]
        for j, key in enumerate(profile_keys):
            df[key] = 1.0 if j == idx else 0.0

    split_years = infer_split_years(frames, cfg.split_years())
    split_year_set = set(split_years)
    print(f"[SPLIT] years={split_years}; full target weeks: Jan-Oct train, Nov val, Dec test")

    temp_keys = ["temp"] + [f"temp_fc_tplus{h:02d}" for h in range(24)]
    workday_keys = ["workday", "workday_future24_mean"]
    month_keys = ["month_sin", "month_cos"]
    enc_ext_keys = temp_keys + workday_keys + month_keys + profile_keys
    future_ext_keys = ["temp", "workday", "month_sin", "month_cos"] + profile_keys

    expert_specs = []
    offset = 0
    for name, keys in (
        ("temp", temp_keys),
        ("workday", workday_keys),
        ("month", month_keys),
        ("profile", profile_keys),
    ):
        expert_specs.append({"name": name, "indices": list(range(offset, offset + len(keys)))})
        offset += len(keys)

    load_scalers: Dict[str, MinMaxScaler] = {}
    load_inverse: Dict[str, Tuple[float, float]] = {}
    scaler_meta = {"load": {}, "temperature": None}
    for name, df in frames.items():
        mask = df.index.year.isin(split_years) & (df.index.month <= 10)
        scaler = MinMaxScaler().fit(df.loc[mask, ["load"]].to_numpy(dtype=float))
        load_scalers[name] = scaler
        load_inverse[name] = _inverse_params(scaler)
        scaler_meta["load"][name] = {
            "data_min": float(scaler.data_min_[0]),
            "data_max": float(scaler.data_max_[0]),
            "fit_scope": "selected years January-October only",
        }
        print(f"[SCALER][LOAD] {name}: min={scaler.data_min_[0]:.6f}, max={scaler.data_max_[0]:.6f}")

    temp_fit = []
    for df in frames.values():
        mask = df.index.year.isin(split_years) & (df.index.month <= 10)
        temp_fit.append(df.loc[mask, ["temp"]].to_numpy(dtype=float))
    temp_scaler = MinMaxScaler().fit(np.concatenate(temp_fit, axis=0))
    scaler_meta["temperature"] = {
        "data_min": float(temp_scaler.data_min_[0]),
        "data_max": float(temp_scaler.data_max_[0]),
        "fit_scope": "shared selected years January-October only",
    }
    print(f"[SCALER][TEMP] min={temp_scaler.data_min_[0]:.6f}, max={temp_scaler.data_max_[0]:.6f}")

    processed = {}
    for name, df in frames.items():
        z = pd.DataFrame(index=df.index)
        z["load"] = load_scalers[name].transform(df[["load"]].to_numpy(dtype=float)).reshape(-1)
        for key in temp_keys:
            z[key] = temp_scaler.transform(df[[key]].to_numpy(dtype=float)).reshape(-1)
        for key in workday_keys + month_keys + profile_keys:
            z[key] = df[key].to_numpy(dtype=float)
        processed[name] = z

    stores = {
        split: {k: [] for k in (
            "enc_l", "enc_ext", "future_ext", "target", "scale", "shift", "metadata"
        )}
        for split in ("train", "val", "test")
    }

    for profile in profiles:
        name = str(profile["name"])
        df = processed[name]
        raw_df = frames[name]
        n_weeks = np.asarray(profile["load"]).shape[0]
        counts = {"train": 0, "val": 0, "test": 0, "excluded": 0}
        for week in range(n_weeks - 1):
            enc_start = week * WEEK_HOURS
            dec_start = enc_start + WEEK_HOURS
            dec_end = dec_start + WEEK_HOURS
            if dec_end > len(df):
                continue
            full_index = df.index[enc_start:dec_end]
            if not _is_hourly_contiguous(full_index, 2 * WEEK_HOURS):
                counts["excluded"] += 1
                continue
            target_index = df.index[dec_start:dec_end]
            if cfg.require_midnight_origin:
                origin = pd.Timestamp(target_index[0])
                if origin.hour != 0 or origin.minute != 0 or origin.second != 0:
                    raise ValueError(
                        f"The 168-hour forecast must start at 00:00 for {name}; "
                        f"found {origin}. Set ONCOR_REQUIRE_MIDNIGHT_ORIGIN=0 only "
                        "for an intentional shifted-origin experiment."
                    )
            split = classify_target_week(target_index, split_year_set)
            if split is None:
                counts["excluded"] += 1
                continue

            enc = df.iloc[enc_start:dec_start]
            future = df.iloc[dec_start:dec_end]
            scale, shift = load_inverse[name]
            store = stores[split]
            store["enc_l"].append(enc["load"].to_numpy(dtype=float)[:, None])
            store["enc_ext"].append(enc[enc_ext_keys].to_numpy(dtype=float))
            store["future_ext"].append(future[future_ext_keys].to_numpy(dtype=float))
            store["target"].append(future["load"].to_numpy(dtype=float)[:, None])
            store["scale"].append([scale])
            store["shift"].append([shift])
            store["metadata"].append({
                "profile": name,
                "week": int(week),
                "encoder_start": str(enc.index[0]),
                "encoder_end": str(enc.index[-1]),
                "target_start": str(future.index[0]),
                "target_end": str(future.index[-1]),
                "raw_target_min": float(raw_df.iloc[dec_start:dec_end]["load"].min()),
                "raw_target_max": float(raw_df.iloc[dec_start:dec_end]["load"].max()),
            })
            counts[split] += 1
        print(f"[WINDOWS] {name}: {counts}")

    def pack(split: str) -> Dict:
        s = stores[split]
        if not s["enc_l"]:
            raise RuntimeError(f"No samples generated for {split}")
        tensor = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
        out = {
            "enc_l": tensor(s["enc_l"]),
            "enc_ext": tensor(s["enc_ext"]),
            "future_ext": tensor(s["future_ext"]),
            "target": tensor(s["target"]),
            "scale": tensor(s["scale"]),
            "shift": tensor(s["shift"]),
            "metadata": s["metadata"],
        }
        print(
            f"[{split.upper()}] weeks={len(s['metadata'])}, points={len(s['metadata']) * 168}, "
            f"enc={tuple(out['enc_l'].shape)}, future_ext={tuple(out['future_ext'].shape)}, "
            f"target={tuple(out['target'].shape)}"
        )
        return out

    tensors = {split: pack(split) for split in ("train", "val", "test")}
    metadata = []
    for split, store in stores.items():
        metadata.extend({"split": split, **row} for row in store["metadata"])
    return {
        "tensors": tensors,
        "enc_ext_keys": enc_ext_keys,
        "future_ext_keys": future_ext_keys,
        "expert_specs": expert_specs,
        "profile_names": profile_names,
        "split_years": split_years,
        "scaler_meta": scaler_meta,
        "metadata": pd.DataFrame(metadata),
    }


def make_loader(data: Dict, batch_size: int, shuffle: bool, num_workers: int, seed: int = 42):
    dataset = TensorDataset(
        data["enc_l"], data["enc_ext"], data["future_ext"], data["target"],
        data["scale"], data["shift"],
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator if shuffle else None,
    )


def gaussian_nll(mu, logvar, target):
    return 0.5 * (logvar + (target - mu).pow(2) * torch.exp(-logvar)).mean()


def gaussian_crps(mu, sigma, target):
    z = (target - mu) / sigma.clamp_min(1e-8)
    normal = Normal(torch.zeros_like(z), torch.ones_like(z))
    return sigma * (
        z * (2.0 * normal.cdf(z) - 1.0)
        + 2.0 * torch.exp(normal.log_prob(z))
        - 1.0 / math.sqrt(math.pi)
    )


def kl_loss(mu_z, logvar_z):
    return (-0.5 * (1.0 + logvar_z - mu_z.pow(2) - logvar_z.exp()).sum(dim=-1)).mean()


def repeated_daily_horizon_weights(device: torch.device) -> torch.Tensor:
    day = np.ones(24, dtype=np.float32)
    day[:3] = 2.0
    day[3:6] = 1.5
    day[9:16] = np.maximum(day[9:16], 1.3)
    week = np.tile(day, 7)
    week /= week.mean()
    return torch.tensor(week, dtype=torch.float32, device=device).view(1, 168, 1)


def weighted_mse(mu, target, weights):
    return ((mu - target).pow(2) * weights).mean()


def boundary_ramp_loss(mu, target, last_enc, hours: int):
    pred_path = torch.cat([last_enc.unsqueeze(1), mu[:, :hours, :]], dim=1)
    true_path = torch.cat([last_enc.unsqueeze(1), target[:, :hours, :]], dim=1)
    return (torch.diff(pred_path, dim=1) - torch.diff(true_path, dim=1)).pow(2).mean()


def scheduled_lr(cfg: Config, update: int) -> float:
    if update <= cfg.lr_warmup_updates:
        return cfg.base_lr * update / max(cfg.lr_warmup_updates, 1)
    if update < cfg.lr_decay_1:
        return cfg.base_lr
    if update < cfg.lr_decay_2:
        return cfg.base_lr * 0.5
    if update < cfg.lr_decay_3:
        return cfg.base_lr * 0.25
    if update < cfg.lr_decay_4:
        return cfg.base_lr / 6.0
    return cfg.base_lr / 12.0


def build_model(cfg: Config, n_enc_ext: int, expert_specs, future_ext_dim: int, device):
    model = Direct168GlobalLocalModel(
        xprime_dim=cfg.xprime_dim,
        input_dim=cfg.input_dim,
        hidden_size=cfg.hidden_dim,
        latent_size=cfg.latent_dim,
        n_encoder_externals=n_enc_ext,
        encoder_expert_specs=expert_specs,
        future_ext_dim=future_ext_dim,
        output_len=cfg.output_len,
        output_dim=cfg.output_dim,
        encoder_mode=cfg.encoder_mode,
        dropout=cfg.dropout,
        logvar_min=cfg.logvar_min,
        logvar_max=cfg.logvar_max,
        reverse_hidden_size=cfg.reverse_hidden_dim,
        fusion_bottleneck=cfg.fusion_bottleneck,
        residual_scale=cfg.residual_scale,
        phase_summary_size=cfg.phase_summary_dim,
        future_summary_size=cfg.future_summary_dim,
        global_hidden=cfg.global_hidden,
        local_ext_hidden=cfg.local_ext_hidden,
        hourly_hidden=cfg.hourly_hidden,
        local_correction_gain=cfg.local_correction_gain,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] mode={cfg.encoder_mode}, parameters={n_params:,}, reverse_context=168h full week")
    return model


@torch.no_grad()
def evaluate(model, loader, metadata, cfg: Config, device, prediction_path: Optional[Path] = None):
    model.eval()
    sum_sq = sum_abs = target_abs = 0.0
    crps_sum = nll_sum = 0.0
    count = 0
    horizon_sq = np.zeros(168, dtype=np.float64)
    horizon_abs = np.zeros(168, dtype=np.float64)
    day_sq = np.zeros(7, dtype=np.float64)
    day_count = np.zeros(7, dtype=np.int64)
    week_peak_sq = 0.0
    week_peak_under = 0
    week_peak_hour_abs = 0.0
    week_count = 0
    rows = []
    meta_pos = 0

    for batch in loader:
        enc_l, enc_ext, future_ext, target, scale, shift = [x.to(device) for x in batch]
        mu, logvar, _, _ = model(
            enc_l, enc_ext, future_ext,
            epoch=cfg.total_updates,
            top_k=cfg.top_k,
            warmup_epochs=cfg.moe_warmup_updates,
            latent_mode="mean",
        )
        sigma = torch.exp(0.5 * logvar)
        error = mu - target
        sum_sq += error.pow(2).sum().item()
        sum_abs += error.abs().sum().item()
        count += error.numel()
        nll_sum += float(gaussian_nll(mu, logvar, target)) * error.numel()
        crps_sum += gaussian_crps(mu, sigma, target).sum().item()

        scale3 = scale[:, None, :]
        shift3 = shift[:, None, :]
        mu_raw = mu * scale3 + shift3
        target_raw = target * scale3 + shift3
        sigma_raw = sigma * scale3.abs()
        raw_error = mu_raw - target_raw
        target_abs += target_raw.abs().sum().item()

        horizon_sq += raw_error.pow(2).sum(dim=(0, 2)).cpu().numpy()
        horizon_abs += raw_error.abs().sum(dim=(0, 2)).cpu().numpy()
        for d in range(7):
            e = raw_error[:, d * 24:(d + 1) * 24, :]
            day_sq[d] += e.pow(2).sum().item()
            day_count[d] += e.numel()

        for b in range(mu.size(0)):
            y = target_raw[b, :, 0]
            p = mu_raw[b, :, 0]
            true_peak, true_idx = y.max(dim=0)
            pred_peak, pred_idx = p.max(dim=0)
            diff = pred_peak - true_peak
            week_peak_sq += diff.pow(2).item()
            week_peak_under += int(diff.item() < 0)
            week_peak_hour_abs += abs(int(pred_idx) - int(true_idx))
            week_count += 1

            if prediction_path is not None:
                meta = metadata[meta_pos + b]
                start = pd.Timestamp(meta["target_start"])
                for h in range(168):
                    rows.append({
                        "profile": meta["profile"],
                        "week": meta["week"],
                        "target_start": meta["target_start"],
                        "timestamp": str(start + pd.Timedelta(hours=h)),
                        "horizon": h + 1,
                        "target_norm": float(target[b, h, 0].cpu()),
                        "pred_norm": float(mu[b, h, 0].cpu()),
                        "sigma_norm": float(sigma[b, h, 0].cpu()),
                        "target_raw": float(target_raw[b, h, 0].cpu()),
                        "pred_raw": float(mu_raw[b, h, 0].cpu()),
                        "sigma_raw": float(sigma_raw[b, h, 0].cpu()),
                    })
        meta_pos += mu.size(0)

    if prediction_path is not None:
        pd.DataFrame(rows).to_csv(prediction_path, index=False)

    mse_norm = sum_sq / max(count, 1)
    raw_sq_total = float(horizon_sq.sum())
    raw_abs_total = float(horizon_abs.sum())
    return {
        "mse_norm": mse_norm,
        "crps_norm": crps_sum / max(count, 1),
        "nll_norm": nll_sum / max(count, 1),
        "rmse_raw": math.sqrt(raw_sq_total / max(count, 1)),
        "mae_raw": raw_abs_total / max(count, 1),
        "wape_raw": raw_abs_total / max(target_abs, 1e-12),
        "week_peak_rmse_raw": math.sqrt(week_peak_sq / max(week_count, 1)),
        "week_peak_under_rate": week_peak_under / max(week_count, 1),
        "week_peak_hour_mae": week_peak_hour_abs / max(week_count, 1),
        "day_rmse_raw": [
            math.sqrt(day_sq[d] / max(int(day_count[d]), 1)) for d in range(7)
        ],
        "horizon_rmse_raw": [
            math.sqrt(horizon_sq[h] / max(week_count, 1)) for h in range(168)
        ],
        "horizon_mae_raw": [
            horizon_abs[h] / max(week_count, 1) for h in range(168)
        ],
        "n_weeks": week_count,
        "n_points": count,
        "booster": model.encoder.booster_diagnostics(),
    }


def save_metric_tables(output: Path, name: str, split: str, metrics: Dict):
    pd.DataFrame({
        "horizon": np.arange(1, 169),
        "rmse_raw": metrics["horizon_rmse_raw"],
        "mae_raw": metrics["horizon_mae_raw"],
    }).to_csv(output / f"{name}_{split}_horizon_metrics.csv", index=False)
    pd.DataFrame({
        "day": np.arange(1, 8),
        "rmse_raw": metrics["day_rmse_raw"],
    }).to_csv(output / f"{name}_{split}_day_metrics.csv", index=False)


def train(model, train_loader, val_loader, val_meta, cfg: Config, device, output: Path):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.base_lr, weight_decay=cfg.weight_decay
    )
    weights = repeated_daily_horizon_weights(device)
    steps_per_epoch = len(train_loader)
    max_epochs = math.ceil(cfg.total_updates / max(steps_per_epoch, 1))
    print(f"[SCHEDULE] steps/epoch={steps_per_epoch}, max_epochs={max_epochs}, updates={cfg.total_updates}")
    print("[PROTOCOL] one-shot 168h; horizon-specific global head + per-hour local correction; no hard persistence skip")

    best_mse = float("inf")
    best_crps = float("inf")
    global_update = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = nll_sum = mse_sum = ramp_sum = kl_sum = 0.0
        batches = 0
        started = time.time()
        for batch in train_loader:
            if global_update >= cfg.total_updates:
                break
            global_update += 1
            lr = scheduled_lr(cfg, global_update)
            for group in optimizer.param_groups:
                group["lr"] = lr
            enc_l, enc_ext, future_ext, target, _, _ = [x.to(device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            mu, logvar, mu_z, logvar_z = model(
                enc_l, enc_ext, future_ext,
                epoch=global_update,
                top_k=cfg.top_k,
                warmup_epochs=cfg.moe_warmup_updates,
                latent_mode="sample",
            )
            nll = gaussian_nll(mu, logvar, target)
            mse = weighted_mse(mu, target, weights)
            ramp = boundary_ramp_loss(mu, target, enc_l[:, -1, :], cfg.ramp_hours)
            kl = kl_loss(mu_z, logvar_z)
            kl_w = cfg.kl_weight * min(1.0, global_update / max(cfg.kl_anneal_updates, 1))
            loss = nll + cfg.mse_weight * mse + cfg.ramp_weight * ramp + kl_w * kl
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at update {global_update}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            loss_sum += float(loss.detach())
            nll_sum += float(nll.detach())
            mse_sum += float(mse.detach())
            ramp_sum += float(ramp.detach())
            kl_sum += float(kl.detach())
            batches += 1

        val = evaluate(model, val_loader, val_meta, cfg, device)
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_update": global_update,
            "val": val,
            "config": asdict(cfg),
        }
        if val["mse_norm"] < best_mse:
            best_mse = val["mse_norm"]
            torch.save(payload, output / "mse.pt")
        if val["crps_norm"] < best_crps:
            best_crps = val["crps_norm"]
            torch.save(payload, output / "crps.pt")
        torch.save(payload, output / "latest.pt")

        record = {
            "epoch": epoch,
            "global_update": global_update,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": loss_sum / max(batches, 1),
            "train_nll": nll_sum / max(batches, 1),
            "train_weighted_mse": mse_sum / max(batches, 1),
            "train_boundary_ramp": ramp_sum / max(batches, 1),
            "train_kl": kl_sum / max(batches, 1),
            "val_mse": val["mse_norm"],
            "val_crps": val["crps_norm"],
            "val_rmse_raw": val["rmse_raw"],
            "val_wape_raw": val["wape_raw"],
            "seconds": time.time() - started,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)

        if epoch == 1 or epoch % 5 == 0 or global_update >= cfg.total_updates:
            print(
                f"[TRAIN] epoch={epoch:04d}/{max_epochs} update={global_update:04d}/{cfg.total_updates} "
                f"loss={record['train_loss']:.6f} valMSE={val['mse_norm']:.7f} "
                f"CRPS={val['crps_norm']:.7f} RMSE={val['rmse_raw']:.4f} "
                f"WAPE={val['wape_raw']:.4f} lr={record['lr']:.2e}"
            )
        if global_update >= cfg.total_updates:
            break


def load_checkpoint(path: Path, model: nn.Module, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(obj["model"] if isinstance(obj, dict) and "model" in obj else obj)
    return obj


def run() -> None:
    cfg = Config()
    cfg.validate()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[PROJECT_ROOT] {PROJECT_ROOT}")

    # The long-used project data_utils_v2.py resolves processed_data/... relative
    # to the current working directory. Always run its data-loading stage from
    # M2oE2_For_Zhe, even when this script is launched inside the package folder.
    os.chdir(PROJECT_ROOT)
    print(f"[WORKDIR] {Path.cwd()}")

    output = Path(cfg.output_root) / cfg.run_name
    output.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles(cfg.profile_ids())
    prepared = prepare_data(profiles, cfg, device)
    prepared["metadata"].to_csv(output / "split_metadata.csv", index=False)

    train_loader = make_loader(
        prepared["tensors"]["train"], cfg.batch_size, True, cfg.num_workers, cfg.seed
    )
    val_loader = make_loader(
        prepared["tensors"]["val"], cfg.batch_size, False, cfg.num_workers
    )
    test_loader = make_loader(
        prepared["tensors"]["test"], cfg.batch_size, False, cfg.num_workers
    )
    model = build_model(
        cfg,
        len(prepared["enc_ext_keys"]),
        prepared["expert_specs"],
        len(prepared["future_ext_keys"]),
        device,
    )

    config_payload = {
        **asdict(cfg),
        "method": "direct one-shot 168-hour global horizon head plus local hourly correction",
        "uses_target_week_load": False,
        "uses_same_hour_previous_week_as_input": True,
        "uses_hard_previous_week_skip": False,
        "uses_hourly_future_external_directly": True,
        "uses_explicit_hour_or_dow_encoding": False,
        "input_load_hours": 168,
        "output_hours": 168,
        "enc_ext_keys": prepared["enc_ext_keys"],
        "future_ext_keys": prepared["future_ext_keys"],
        "expert_specs": prepared["expert_specs"],
        "profile_names": prepared["profile_names"],
        "split_years": prepared["split_years"],
        "scaler_meta": prepared["scaler_meta"],
        "split_definition": (
            "target weeks wholly inside Jan-Oct=train, wholly inside November=val, "
            "wholly inside December=test; boundary-crossing weeks excluded"
        ),
        "checkpoint_policy": "load an existing mse/crps/latest checkpoint; otherwise train from scratch",
        "reverse_context_hours": 168,
        "reverse_time_order": "Sunday 23:00 to Monday 00:00",
        "booster_definition": "full previous week encoded in reverse and shared across seven daily phases",
        "decoder_change": (
            "168 horizon-specific global base plus zero-initialized shared local correction; "
            "previous-week same-hour load is input only"
        ),
    }
    with (output / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    best_path = output / "mse.pt"
    checkpoint_candidates = [
        best_path,
        output / "crps.pt",
        output / "latest.pt",
    ]
    existing_checkpoint = next((p for p in checkpoint_candidates if p.exists()), None)
    if cfg.eval_only:
        if existing_checkpoint is None:
            raise FileNotFoundError(f"No checkpoint available for eval-only in: {output}")
    elif cfg.force_retrain or existing_checkpoint is None:
        train(
            model,
            train_loader,
            val_loader,
            prepared["tensors"]["val"]["metadata"],
            cfg,
            device,
            output,
        )
    else:
        print(f"[LOAD] existing checkpoint: {existing_checkpoint}")

    results = {}
    for ckpt_name in ("mse", "crps"):
        ckpt = output / f"{ckpt_name}.pt"
        if not ckpt.exists():
            continue
        load_checkpoint(ckpt, model, device)
        results[ckpt_name] = {}
        for split, loader in (("val", val_loader), ("test", test_loader)):
            pred_path = (
                output / f"{ckpt_name}_{split}_predictions.csv"
                if cfg.save_predictions else None
            )
            metrics = evaluate(
                model,
                loader,
                prepared["tensors"][split]["metadata"],
                cfg,
                device,
                prediction_path=pred_path,
            )
            results[ckpt_name][split] = metrics
            save_metric_tables(output, ckpt_name, split, metrics)
            print(
                f"[{ckpt_name.upper()}][{split.upper()}] "
                f"MSE={metrics['mse_norm']:.7f} CRPS={metrics['crps_norm']:.7f} "
                f"RMSE={metrics['rmse_raw']:.5f} WAPE={metrics['wape_raw']:.4f} "
                f"weekPeakRMSE={metrics['week_peak_rmse_raw']:.5f} "
                f"weekPeakUnder={metrics['week_peak_under_rate']:.4f} "
                f"peakHourMAE={metrics['week_peak_hour_mae']:.3f}"
            )

    with (output / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[DONE] {output}")


if __name__ == "__main__":
    run()
