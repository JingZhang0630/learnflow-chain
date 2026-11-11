"""Convert the Junyi Academy public log (Log_Problem.csv, ~3 GB) into a compact parquet cache.

Institution key: Info_UserData.user_city (city code). Item key: ucid (exercise).
Raw CSV is read in chunks; only the columns needed downstream are kept.
"""
import sys, time
from pathlib import Path
import pandas as pd

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/junyi")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "data/cache")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    users = pd.read_csv(SRC / "Info_UserData.csv", usecols=["uuid", "user_city", "user_grade"])
    content = pd.read_csv(SRC / "Info_Content.csv")
    uid = {u: i for i, u in enumerate(users.uuid)}
    item_codes = {c: i for i, c in enumerate(content.ucid)}
    parts = []
    cols = ["timestamp_TW", "uuid", "ucid", "upid", "is_correct", "total_sec_taken", "total_attempt_cnt", "used_hint_cnt"]
    for k, ch in enumerate(pd.read_csv(SRC / "Log_Problem.csv", usecols=cols, chunksize=2_000_000)):
        ch = ch[ch.ucid.isin(item_codes.keys()) & ch.uuid.isin(uid.keys())]
        parts.append(pd.DataFrame({
            "ts": pd.to_datetime(ch.timestamp_TW.str.replace(" UTC", ""), utc=True).dt.tz_localize(None).astype("datetime64[s]").astype("int64"),
            "user": ch.uuid.map(uid).astype("int32"),
            "item": ch.ucid.map(item_codes).astype("int32"),
            "correct": ch.is_correct.astype("int8"),
            "sec": ch.total_sec_taken.astype("float32"),
            "attempts": ch.total_attempt_cnt.astype("int16"),
            "hints": ch.used_hint_cnt.astype("int16"),
            "upid": ch.upid,
        }))
        print(f"chunk {k} rows_total={sum(len(p) for p in parts):,} t={time.time()-t0:.0f}s", flush=True)
    log = pd.concat(parts, ignore_index=True)
    log["upid"] = log.upid.astype("category").cat.codes.astype("int32")
    log.to_parquet(OUT / "junyi_log.parquet", index=False)
    users.assign(user=range(len(users)))[["user", "user_city", "user_grade"]].to_parquet(OUT / "junyi_users.parquet", index=False)
    content.assign(item=range(len(content)))[["item", "ucid", "difficulty", "subject", "learning_stage", "level2_id", "level3_id"]].to_parquet(OUT / "junyi_items.parquet", index=False)
    print(f"done rows={len(log):,} users={log.user.nunique():,} items={log.item.nunique():,} secs={time.time()-t0:.0f}")


if __name__ == "__main__":
    main()
