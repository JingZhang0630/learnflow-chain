"""Dataset construction exactly as frozen in docs/experiment_protocol.md §1."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..rasch import Responses

LEARNFLOW_DATA = Path(os.environ.get("LEARNFLOW_DATA", "data/raw"))
CACHE = Path(os.environ.get("LFCHAIN_CACHE", Path(__file__).resolve().parents[3] / "data" / "cache"))
MASTER_SEED = 20261018


@dataclass
class MultiOrgData:
    name: str
    orgs: List[str]            # institution labels
    train: List[Responses]     # per institution, global user/item indices
    test: List[Responses]
    n_users: int
    n_items: int
    item_meta: pd.DataFrame    # index = item idx
    true_b: np.ndarray = None  # synthetic only
    n_raw_rows: int = 0


def _encode(df: pd.DataFrame, org_col: str, user_col: str, item_col: str, y_col: str, seed: int, name: str,
            item_meta=None, n_raw=0) -> MultiOrgData:
    df = df.copy()
    df["uu"] = df[user_col].astype("category").cat.codes.astype(np.int64)
    items = df[item_col].astype("category")
    df["ii"] = items.cat.codes.astype(np.int64)
    n_users, n_items = int(df.uu.max() + 1), int(df.ii.max() + 1)
    rng = np.random.default_rng(seed)
    df["test"] = rng.random(len(df)) < 0.2
    orgs = sorted(df[org_col].unique().tolist(), key=lambda o: -int((df[org_col] == o).sum()))
    train, test = [], []
    for o in orgs:
        s = df[df[org_col] == o]
        for flag, out in ((False, train), (True, test)):
            t = s[s.test == flag]
            out.append(Responses(t.uu.to_numpy(), t.ii.to_numpy(), t[y_col].to_numpy().astype(float), n_users, n_items))
    meta = pd.DataFrame({"item_key": items.cat.categories})
    if item_meta is not None:
        meta = meta.merge(item_meta, left_on="item_key", right_index=True, how="left")
    return MultiOrgData(name, [str(o) for o in orgs], train, test, n_users, n_items, meta, n_raw_rows=n_raw)


def load_assist09(seed: int = MASTER_SEED, level: str = "item") -> MultiOrgData:
    """level="skill" calibrates the same responses and the same train/test split per skill instead of per item."""
    d = pd.read_csv(LEARNFLOW_DATA / "assist09_corrected.csv", encoding="latin-1", low_memory=False,
                    usecols=["order_id", "user_id", "problem_id", "skill_id", "school_id", "correct", "original"])
    n_raw = len(d)
    d = d[d.original == 1].drop_duplicates("order_id", keep="first")
    sizes = d.school_id.value_counts()
    d = d[d.school_id.isin(sizes[sizes >= 5000].index)]
    ic = d.problem_id.value_counts()
    d = d[d.problem_id.isin(ic[ic >= 20].index)]
    uc = d.user_id.value_counts()
    d = d[d.user_id.isin(uc[uc >= 5].index)]
    if level == "skill":
        # responses without a skill label keep their own item as the calibration unit
        d = d.assign(skill_key=d.skill_id.astype(object).where(d.skill_id.notna(), "item_" + d.problem_id.astype(str)).astype(str))
        return _encode(d, "school_id", "user_id", "skill_key", "correct", seed, "ASSIST09-skill", n_raw=n_raw)
    return _encode(d, "school_id", "user_id", "problem_id", "correct", seed, "ASSIST09", n_raw=n_raw)


def load_junyi(seed: int = MASTER_SEED, n_orgs: int = 10) -> MultiOrgData:
    log = pd.read_parquet(CACHE / "junyi_log.parquet", columns=["user", "item", "correct"])
    n_raw = len(log)
    users = pd.read_parquet(CACHE / "junyi_users.parquet")
    items = pd.read_parquet(CACHE / "junyi_items.parquet").set_index("item")
    log = log.merge(users[["user", "user_city"]], on="user", how="left")
    log = log[log.user_city.notna() & (log.user_city.astype(str).str.len() > 0)]
    top = log.user_city.value_counts().index[:n_orgs]
    log = log[log.user_city.isin(top)]
    ic = log.item.value_counts()
    log = log[log.item.isin(ic[ic >= 50].index)]
    uc = log.user.value_counts()
    log = log[log.user.isin(uc[uc >= 10].index)]
    meta = items[["difficulty", "subject", "learning_stage", "level2_id"]]
    return _encode(log, "user_city", "user", "item", "correct", seed, "Junyi", item_meta=meta, n_raw=n_raw)


def make_synthetic(K: int, iid: bool, seed: int, n_items: int = 500, n_students: int = 300, per_student: int = 40) -> MultiOrgData:
    rng = np.random.default_rng(seed)
    b = rng.normal(0, 1, n_items)
    rows = []
    uid = 0
    mus = np.zeros(K) if iid else np.linspace(-1, 1, K)
    for k in range(K):
        if iid:
            pool = np.arange(n_items)
        else:
            start = int(round(k * (n_items - 250) / max(K - 1, 1)))
            pool = np.arange(start, start + 250)
        for _ in range(n_students):
            th = rng.normal(mus[k], 1)
            its = rng.choice(pool, per_student, replace=False)
            y = rng.random(per_student) < 1 / (1 + np.exp(-(th - b[its])))
            rows.append(pd.DataFrame({"org": k, "user": uid, "item": its, "y": y.astype(int)}))
            uid += 1
    df = pd.concat(rows, ignore_index=True)
    data = _encode(df, "org", "user", "item", "y", seed + 1, f"SYN-K{K}-{'IID' if iid else 'nonIID'}")
    data.true_b = b[data.item_meta.item_key.to_numpy().astype(int)]
    return data
