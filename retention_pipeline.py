#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

SEED = 42
MAX_FOLDS = 3
GEN_CHUNKSIZE = 250_000

CAT_FEATURES = [
    "country_code",
    "tx_card_brand",
    "tx_card_funding",
    "quiz_source",
    "quiz_experience",
    "quiz_frustration",
    "quiz_first_feature",
    "quiz_role",
    "quiz_usage_plan",
]

CLUSTER_FEATURES = [
    "gen_total",
    "gen_completed",
    "gen_total_credits",
    "gen_trend",
    "days_since_last_gen",
    "total_purchases",
    "total_spend",
    "tx_failure_rate",
    "gen_intensity",
]

ZERO_COLS = [
    "gen_total",
    "gen_completed",
    "gen_failed",
    "gen_nsfw",
    "gen_canceled",
    "completion_rate",
    "nsfw_rate",
    "gen_total_credits",
    "gen_avg_credits",
    "gen_video_count",
    "gen_image_count",
    "gen_video_ratio",
    "gen_unique_types",
    "gen_avg_duration",
    "gen_avg_hour",
    "gen_active_days",
    "gen_intensity",
    "gen_has_generations",
    "gen_week1_count",
    "gen_week2_count",
    "gen_trend",
    "credits_week1",
    "credits_week2",
    "credits_trend",
    "deceleration_flag",
    "days_since_last_gen",
    "days_since_first_gen",
    "gen_day01_count",
    "credits_day01",
    "day01_ratio",
    "day01_credits_ratio",
    "burst_flag",
    "is_short_life",
    "is_one_day_user",
    "first_quit_day",
    "max_active_streak",
    "week1_active_days",
    "week2_active_days",
] + [f"active_day_{d}" for d in range(14)]


@dataclass
class PreparedModelData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    X_train_encoded: pd.DataFrame
    X_test_encoded: pd.DataFrame
    y_train: np.ndarray
    feature_cols: list[str]
    cat_features: list[str]
    label_enc: LabelEncoder
    class_names: np.ndarray
    n_classes: int
    n_folds: int
    seed: int = SEED


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, low_memory=False)


def load_raw_data(data_root: str | Path = ".") -> dict[str, pd.DataFrame]:
    root = Path(data_root)
    train_dir = root / "Train Data"
    test_dir = root / "Test Data"

    data = {
        "train_users": read_csv(train_dir / "train_users.csv"),
        "train_props": read_csv(train_dir / "train_users_properties.csv"),
        "train_purch": read_csv(train_dir / "train_users_purchases.csv"),
        "train_tx": read_csv(train_dir / "train_users_transaction_attempts.csv"),
        "train_quiz": read_csv(train_dir / "train_users_quizzes.csv"),
        "test_users": read_csv(test_dir / "test_users.csv"),
        "test_props": read_csv(test_dir / "test_users_properties.csv"),
        "test_purch": read_csv(test_dir / "test_users_purchases.csv"),
        "test_tx": read_csv(test_dir / "test_users_transaction_attempts.csv"),
        "test_quiz": read_csv(test_dir / "test_users_quizzes.csv"),
        "test_gen": read_csv(test_dir / "test_users_generations.csv"),
    }

    print(
        f"Train users: {len(data['train_users'])}, "
        f"Test users: {len(data['test_users'])}"
    )
    print(
        "Target distribution:\n"
        f"{data['train_users'].churn_status.value_counts(normalize=True)}"
    )
    print(f"Test generations: {len(data['test_gen']):,} rows")
    return data


def build_properties_features(props_df: pd.DataFrame) -> pd.DataFrame:
    df = props_df.copy()
    dt_parts = df["subscription_start_date"].str.extract(
        r"(\d+)-(\d+)-(\d+)\s+(\d+):(\d+):(\d+)"
    ).astype(float)
    dt_parts.columns = ["year", "month", "day", "hour", "minute", "second"]

    df["sub_month"] = dt_parts["month"].astype(int)
    y = dt_parts["year"].astype(int)
    m = dt_parts["month"].astype(int)
    d = dt_parts["day"].astype(int)
    m_adj = m.copy()
    y_adj = y.copy()
    mask = m <= 2
    m_adj[mask] = m[mask] + 12
    y_adj[mask] = y[mask] - 1
    df["sub_day_of_week"] = (
        (d + (13 * (m_adj + 1)) // 5 + y_adj + y_adj // 4 - y_adj // 100 + y_adj // 400) % 7
    ).astype(int)
    df["sub_hour"] = dt_parts["hour"].astype(int)
    df["sub_date_rank"] = df["subscription_start_date"].rank(pct=True)

    country_freq = df["country_code"].value_counts(normalize=True)
    df["country_freq"] = df["country_code"].map(country_freq)

    return df[
        [
            "user_id",
            "subscription_plan",
            "country_code",
            "country_freq",
            "sub_month",
            "sub_day_of_week",
            "sub_hour",
            "sub_date_rank",
        ]
    ].copy()


def build_purchase_features(purch_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    all_uids = users_df[["user_id"]].copy()
    if len(purch_df) == 0:
        for col in [
            "total_purchases",
            "total_spend",
            "avg_purchase_amount",
            "max_purchase",
            "min_purchase",
            "n_purchase_types",
            "ptype_credits_package",
            "ptype_subscription_create",
            "ptype_subscription_update",
            "has_purchases",
            "days_between_first_last",
        ]:
            all_uids[col] = 0
        return all_uids

    agg = purch_df.groupby("user_id").agg(
        total_purchases=("transaction_id", "count"),
        total_spend=("purchase_amount_dollars", "sum"),
        avg_purchase_amount=("purchase_amount_dollars", "mean"),
        max_purchase=("purchase_amount_dollars", "max"),
        min_purchase=("purchase_amount_dollars", "min"),
        n_purchase_types=("purchase_type", "nunique"),
    ).reset_index()

    ptype_counts = purch_df.groupby(["user_id", "purchase_type"]).size().unstack(fill_value=0)
    ptype_counts.columns = ["ptype_" + col.lower().replace(" ", "_") for col in ptype_counts.columns]
    ptype_counts = ptype_counts.reset_index()
    agg = agg.merge(ptype_counts, on="user_id", how="left")

    purch_dates = purch_df[["user_id", "purchase_time"]].copy()
    parts = purch_dates["purchase_time"].str.extract(
        r"(\d+)-(\d+)-(\d+)\s+(\d+):(\d+)"
    ).astype(float)
    parts.columns = ["y", "m", "d", "h", "mi"]
    purch_dates["day_num"] = parts["m"] * 31 + parts["d"] + parts["h"] / 24.0

    day_range = purch_dates.groupby("user_id")["day_num"].agg(["min", "max"])
    day_range["days_between_first_last"] = day_range["max"] - day_range["min"]
    day_range = day_range[["days_between_first_last"]].reset_index()
    agg = agg.merge(day_range, on="user_id", how="left")

    result = all_uids.merge(agg, on="user_id", how="left")
    result["has_purchases"] = result["total_purchases"].notna().astype(int)
    return result.fillna(0)


def build_transaction_features(
    tx_df: pd.DataFrame,
    purch_df: pd.DataFrame,
    users_df: pd.DataFrame,
) -> pd.DataFrame:
    all_uids = users_df[["user_id"]].copy()
    successful_tx_ids = set(purch_df["transaction_id"].unique())

    tx = tx_df.copy()
    tx["is_success"] = tx["transaction_id"].isin(successful_tx_ids)
    tx_uid_map = purch_df[["transaction_id", "user_id"]].drop_duplicates()
    tx = tx.merge(tx_uid_map, on="transaction_id", how="left")

    success_tx = tx[tx["is_success"]].copy()
    failed_tx = tx[~tx["is_success"]].copy().drop(columns=["user_id"], errors="ignore")

    if len(success_tx) > 0:
        card_risk = success_tx.groupby("user_id").agg(
            tx_success_count=("transaction_id", "count"),
            tx_has_digital_wallet=("digital_wallet", lambda x: (x.fillna("none") != "none").any()),
            tx_has_3d_secure=("is_3d_secure", lambda x: x.fillna(False).any()),
            tx_is_prepaid=("is_prepaid", lambda x: x.fillna(False).any()),
            tx_is_virtual=("is_virtual", lambda x: x.fillna(False).any()),
            tx_is_business=("is_business", lambda x: x.fillna(False).any()),
            tx_cvc_pass_rate=("cvc_check", lambda x: (x.fillna("") == "pass").mean()),
            tx_avg_amount=("amount_in_usd", "mean"),
            tx_max_amount=("amount_in_usd", "max"),
        ).reset_index()

        success_tx["country_mismatch"] = (
            success_tx["billing_address_country"].fillna("") != success_tx["card_country"].fillna("")
        ).astype(int)
        mismatch = success_tx.groupby("user_id")["country_mismatch"].mean().reset_index()
        mismatch.columns = ["user_id", "tx_country_mismatch_rate"]
        card_risk = card_risk.merge(mismatch, on="user_id", how="left")

        for col_name in ["card_brand", "card_funding"]:
            mode_df = success_tx.groupby("user_id")[col_name].agg(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown"
            ).reset_index()
            mode_df.columns = ["user_id", f"tx_{col_name}"]
            card_risk = card_risk.merge(mode_df, on="user_id", how="left")
    else:
        card_risk = all_uids.copy()
        card_risk["tx_success_count"] = 0

    fp_cols = [
        "card_brand",
        "card_country",
        "card_funding",
        "bank_name",
        "billing_address_country",
        "is_prepaid",
        "is_virtual",
    ]

    success_tx["fingerprint"] = success_tx[fp_cols].fillna("unk").astype(str).agg("|".join, axis=1)
    failed_tx["fingerprint"] = failed_tx[fp_cols].fillna("unk").astype(str).agg("|".join, axis=1)
    fp_to_users = success_tx.groupby("fingerprint")["user_id"].agg(lambda x: list(set(x))).to_dict()

    failed_tx["matched_users"] = failed_tx["fingerprint"].map(fp_to_users)
    matched = failed_tx.dropna(subset=["matched_users"]).copy()

    if len(matched) > 0:
        matched = matched.explode("matched_users")
        matched["user_id"] = matched["matched_users"]

        fp_n_users = {fp: len(uids) for fp, uids in fp_to_users.items()}
        matched["weight"] = matched["fingerprint"].map(fp_n_users).apply(lambda n: 1.0 / n)

        fail_agg = matched.groupby("user_id").agg(
            tx_fail_count_weighted=("weight", "sum"),
            tx_fail_count_raw=("weight", "count"),
        ).reset_index()

        matched["failure_code"] = matched["failure_code"].fillna("unknown")
        for code in [
            "card_declined",
            "expired_card",
            "incorrect_cvc",
            "incorrect_number",
            "processing_error",
        ]:
            code_agg = matched[matched["failure_code"] == code].groupby("user_id")["weight"].sum().reset_index()
            code_agg.columns = ["user_id", f"fail_{code}"]
            fail_agg = fail_agg.merge(code_agg, on="user_id", how="left")

        fail_agg = fail_agg.fillna(0)
    else:
        fail_agg = pd.DataFrame(columns=["user_id", "tx_fail_count_weighted", "tx_fail_count_raw"])

    result = all_uids.merge(card_risk, on="user_id", how="left")
    result = result.merge(fail_agg, on="user_id", how="left")

    result["tx_success_count"] = result["tx_success_count"].fillna(0)
    result["tx_fail_count_weighted"] = result["tx_fail_count_weighted"].fillna(0)
    result["tx_failure_rate"] = result["tx_fail_count_weighted"] / (
        result["tx_fail_count_weighted"] + result["tx_success_count"] + 1e-6
    )

    num_cols = result.select_dtypes(include=[np.number]).columns
    result[num_cols] = result[num_cols].fillna(0)

    for col in ["tx_card_brand", "tx_card_funding"]:
        if col in result.columns:
            result[col] = result[col].fillna("unknown")

    for col in [
        "tx_has_digital_wallet",
        "tx_has_3d_secure",
        "tx_is_prepaid",
        "tx_is_virtual",
        "tx_is_business",
    ]:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)

    return result


def parse_gen_day_num(date_series: pd.Series) -> pd.Series:
    parts = date_series.str.extract(r"(\d+)-(\d+)-(\d+)\s+(\d+):(\d+)").astype(float)
    parts.columns = ["y", "m", "d", "h", "mi"]
    return parts["m"] * 31 + parts["d"] + parts["h"] / 24.0


def agg_gen_chunk(g: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    g = g.copy()
    g["day_num"] = parse_gen_day_num(g["created_at"])
    g["hour"] = g["created_at"].str.extract(r"\d+-\d+-\d+\s+(\d+):").astype(float).iloc[:, 0]
    g["day_date"] = g["created_at"].str.extract(r"(\d+-\d+-\d+)").iloc[:, 0]
    g["credit_cost"] = pd.to_numeric(g["credit_cost"], errors="coerce").fillna(0)
    g["duration"] = pd.to_numeric(g["duration"], errors="coerce")
    g["is_video"] = g["generation_type"].str.contains("video", case=False, na=False).astype(int)
    g["is_image"] = g["generation_type"].str.contains("image", case=False, na=False).astype(int)

    agg = g.groupby("user_id").agg(
        gen_total=("generation_id", "count"),
        gen_completed=("status", lambda x: (x == "completed").sum()),
        gen_failed=("status", lambda x: (x == "failed").sum()),
        gen_nsfw=("status", lambda x: (x == "nsfw").sum()),
        gen_canceled=("status", lambda x: (x == "canceled").sum()),
        gen_total_credits=("credit_cost", "sum"),
        gen_video_count=("is_video", "sum"),
        gen_image_count=("is_image", "sum"),
        duration_sum=("duration", lambda x: x.dropna().sum()),
        duration_count=("duration", lambda x: x.dropna().count()),
        hour_sum=("hour", "sum"),
        day_num_min=("day_num", "min"),
        day_num_max=("day_num", "max"),
    )

    day_dates = g.groupby("user_id")["day_date"].apply(set).reset_index()
    day_dates.columns = ["user_id", "day_date_set"]
    gen_types = g.groupby("user_id")["generation_type"].apply(set).reset_index()
    gen_types.columns = ["user_id", "gen_type_set"]
    mini = g[["user_id", "generation_id", "day_num", "credit_cost"]].copy()
    return agg, mini, day_dates, gen_types


def finalize_gen_features(base: pd.DataFrame, trend_df: pd.DataFrame, all_uids: pd.DataFrame) -> pd.DataFrame:
    base["gen_avg_credits"] = base["gen_total_credits"] / (base["gen_total"] + 1e-6)
    base["completion_rate"] = base["gen_completed"] / (base["gen_total"] + 1e-6)
    base["nsfw_rate"] = base["gen_nsfw"] / (base["gen_total"] + 1e-6)
    base["gen_intensity"] = base["gen_total"] / (base["gen_active_days"] + 1e-6)
    base["gen_video_ratio"] = base["gen_video_count"] / (base["gen_total"] + 1e-6)

    base["gen_avg_duration"] = base["duration_sum"] / (base["duration_count"] + 1e-6)
    base["gen_avg_hour"] = base["hour_sum"] / (base["gen_total"] + 1e-6)
    base = base.drop(columns=["duration_sum", "duration_count", "hour_sum"], errors="ignore")

    trend_df = trend_df.merge(base[["day_num_min"]].reset_index(), on="user_id", how="left")
    trend_df["rel_day"] = trend_df["day_num"] - trend_df["day_num_min"]
    trend_df["week"] = np.where(trend_df["rel_day"] < 7, 1, 2)

    day01_agg = trend_df[trend_df["rel_day"] < 2].groupby("user_id").agg(
        gen_day01_count=("generation_id", "count"),
        credits_day01=("credit_cost", "sum"),
    ).reset_index()

    week_agg = trend_df.groupby(["user_id", "week"]).agg(
        count=("generation_id", "count"),
        credits=("credit_cost", "sum"),
    ).reset_index()

    trend_df["day_int"] = trend_df["rel_day"].astype(int).clip(0, 13)
    daily = trend_df.groupby(["user_id", "day_int"]).size().unstack(fill_value=0)
    daily = (daily > 0).astype(int)
    daily.columns = [f"active_day_{col}" for col in daily.columns]
    for day in range(14):
        col = f"active_day_{day}"
        if col not in daily.columns:
            daily[col] = 0
    daily = daily[[f"active_day_{day}" for day in range(14)]].reset_index()

    daily_arr = daily[[f"active_day_{day}" for day in range(14)]].values
    first_quit = np.zeros(len(daily), dtype=int)
    max_streak = np.zeros(len(daily), dtype=int)
    for i in range(len(daily)):
        row = daily_arr[i]
        first_active = np.argmax(row) if row.any() else 0
        after_first = row[first_active:]
        zeros_after = np.where(after_first == 0)[0]
        first_quit[i] = first_active + zeros_after[0] if len(zeros_after) > 0 else 14

        streak = 0
        max_value = 0
        for value in row:
            if value == 1:
                streak += 1
                max_value = max(max_value, streak)
            else:
                streak = 0
        max_streak[i] = max_value

    daily["first_quit_day"] = first_quit
    daily["max_active_streak"] = max_streak
    daily["week1_active_days"] = daily_arr[:, :7].sum(axis=1)
    daily["week2_active_days"] = daily_arr[:, 7:].sum(axis=1)

    del trend_df
    gc.collect()

    w1 = week_agg[week_agg.week == 1][["user_id", "count", "credits"]].rename(
        columns={"count": "gen_week1_count", "credits": "credits_week1"}
    )
    w2 = week_agg[week_agg.week == 2][["user_id", "count", "credits"]].rename(
        columns={"count": "gen_week2_count", "credits": "credits_week2"}
    )

    base = base.reset_index()
    base = base.merge(w1, on="user_id", how="left")
    base = base.merge(w2, on="user_id", how="left")
    base = base.merge(day01_agg, on="user_id", how="left")
    base = base.merge(daily, on="user_id", how="left")

    daily_cols = [f"active_day_{day}" for day in range(14)] + [
        "first_quit_day",
        "max_active_streak",
        "week1_active_days",
        "week2_active_days",
    ]
    for col in [
        "gen_week1_count",
        "gen_week2_count",
        "credits_week1",
        "credits_week2",
        "gen_day01_count",
        "credits_day01",
        *daily_cols,
    ]:
        if col in base.columns:
            base[col] = base[col].fillna(0)

    base["day01_ratio"] = base["gen_day01_count"] / (base["gen_total"] + 1)
    base["day01_credits_ratio"] = base["credits_day01"] / (base["gen_total_credits"] + 1)
    base["burst_flag"] = (base["day01_ratio"] > 0.5).astype(int)
    base["is_short_life"] = (base["gen_active_days"] <= 2).astype(int)
    base["is_one_day_user"] = (base["gen_active_days"] == 1).astype(int)
    base["gen_trend"] = (base["gen_week2_count"] - base["gen_week1_count"]) / (base["gen_week1_count"] + 1)
    base["credits_trend"] = (base["credits_week2"] - base["credits_week1"]) / (base["credits_week1"] + 1)
    base["deceleration_flag"] = (base["gen_week2_count"] < base["gen_week1_count"] * 0.7).astype(int)

    global_max_day = base["day_num_max"].max()
    base["days_since_last_gen"] = global_max_day - base["day_num_max"]
    base["days_since_first_gen"] = global_max_day - base["day_num_min"]
    base["gen_has_generations"] = 1
    base = base.drop(columns=["day_num_min", "day_num_max"], errors="ignore")

    result = all_uids.merge(base, on="user_id", how="left")
    result["gen_has_generations"] = result["gen_has_generations"].fillna(0).astype(int)
    num_cols = result.select_dtypes(include=[np.number]).columns
    result[num_cols] = result[num_cols].fillna(0)
    return result


def build_generation_features_chunked(
    csv_path: str | Path,
    users_df: pd.DataFrame,
    chunksize: int = GEN_CHUNKSIZE,
) -> pd.DataFrame:
    all_uids = users_df[["user_id"]].copy()
    user_set = set(users_df["user_id"])

    agg_parts: list[pd.DataFrame] = []
    trend_parts: list[pd.DataFrame] = []
    day_date_parts: list[pd.DataFrame] = []
    gen_type_parts: list[pd.DataFrame] = []

    print(f"Reading {csv_path} in chunks of {chunksize:,}...")
    for i, chunk in enumerate(pd.read_csv(csv_path, index_col=0, chunksize=chunksize, low_memory=False)):
        chunk = chunk[chunk["user_id"].isin(user_set)]
        if len(chunk) == 0:
            print(f"  chunk {i}: 0 matching rows, skipping")
            continue
        agg, mini, dd, gt = agg_gen_chunk(chunk)
        agg_parts.append(agg)
        trend_parts.append(mini)
        day_date_parts.append(dd)
        gen_type_parts.append(gt)
        print(f"  chunk {i}: {len(chunk):,} rows processed")
        del chunk
        gc.collect()

    if not agg_parts:
        for col in ZERO_COLS:
            all_uids[col] = 0
        return all_uids

    combined = pd.concat(agg_parts)
    del agg_parts
    gc.collect()

    base = combined.groupby("user_id").agg(
        {
            "gen_total": "sum",
            "gen_completed": "sum",
            "gen_failed": "sum",
            "gen_nsfw": "sum",
            "gen_canceled": "sum",
            "gen_total_credits": "sum",
            "gen_video_count": "sum",
            "gen_image_count": "sum",
            "duration_sum": "sum",
            "duration_count": "sum",
            "hour_sum": "sum",
            "day_num_min": "min",
            "day_num_max": "max",
        }
    )
    del combined
    gc.collect()

    all_dd = pd.concat(day_date_parts)
    del day_date_parts
    exact_active = all_dd.groupby("user_id")["day_date_set"].apply(
        lambda sets: len(set().union(*[s for s in sets if isinstance(s, set)])) if len(sets) > 0 else 0
    ).reset_index()
    exact_active.columns = ["user_id", "gen_active_days"]
    del all_dd
    gc.collect()

    all_gt = pd.concat(gen_type_parts)
    del gen_type_parts
    exact_types = all_gt.groupby("user_id")["gen_type_set"].apply(
        lambda sets: len(set().union(*[s for s in sets if isinstance(s, set)])) if len(sets) > 0 else 0
    ).reset_index()
    exact_types.columns = ["user_id", "gen_unique_types"]
    del all_gt
    gc.collect()

    base = base.reset_index()
    base = base.drop(columns=["gen_active_days", "gen_unique_types"], errors="ignore")
    base = base.merge(exact_active, on="user_id", how="left")
    base = base.merge(exact_types, on="user_id", how="left")
    base = base.set_index("user_id")

    trend_df = pd.concat(trend_parts)
    del trend_parts
    gc.collect()
    return finalize_gen_features(base, trend_df, all_uids)


def build_generation_features(gen_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    all_uids = users_df[["user_id"]].copy()
    if len(gen_df) == 0:
        for col in ZERO_COLS:
            all_uids[col] = 0
        return all_uids

    agg, mini, dd, gt = agg_gen_chunk(gen_df.copy())
    agg["gen_active_days"] = dd.set_index("user_id")["day_date_set"].apply(len)
    agg["gen_unique_types"] = gt.set_index("user_id")["gen_type_set"].apply(len)
    return finalize_gen_features(agg, mini, all_uids)


def build_quiz_features(quiz_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    all_uids = users_df[["user_id"]].copy()
    q = quiz_df[
        ["user_id", "source", "team_size", "experience", "usage_plan", "frustration", "first_feature", "role"]
    ].copy()

    q["team_size"] = pd.to_numeric(q["team_size"], errors="coerce")
    q["has_quiz_data"] = q[
        ["source", "experience", "frustration", "first_feature", "role"]
    ].notna().any(axis=1).astype(int)

    q = q.rename(
        columns={
            "source": "quiz_source",
            "experience": "quiz_experience",
            "frustration": "quiz_frustration",
            "first_feature": "quiz_first_feature",
            "role": "quiz_role",
            "usage_plan": "quiz_usage_plan",
            "team_size": "quiz_team_size",
        }
    )

    result = all_uids.merge(q, on="user_id", how="left")
    result["has_quiz_data"] = result["has_quiz_data"].fillna(0).astype(int)
    result["quiz_team_size"] = result["quiz_team_size"].fillna(0)

    for col in [
        "quiz_source",
        "quiz_experience",
        "quiz_frustration",
        "quiz_first_feature",
        "quiz_role",
        "quiz_usage_plan",
    ]:
        if col in result.columns:
            result[col] = result[col].fillna("missing")

    return result


def merge_all_features(
    users_df: pd.DataFrame,
    props: pd.DataFrame,
    purch: pd.DataFrame,
    tx: pd.DataFrame,
    gen: pd.DataFrame,
    quiz: pd.DataFrame,
) -> pd.DataFrame:
    df = users_df.copy()
    for feat_df in [props, purch, tx, gen, quiz]:
        df = df.merge(feat_df, on="user_id", how="left")

    df["credits_x_failure_rate"] = df["gen_total_credits"] * df["tx_failure_rate"]
    df["trend_x_recency"] = df["gen_trend"] * df["days_since_last_gen"]
    df["spend_per_gen"] = df["total_spend"] / (df["gen_total"] + 1)
    df["gen_total_x_purchases"] = df["gen_total"] * df["total_purchases"]

    df["prepaid_x_failure"] = df["tx_is_prepaid"] * df["tx_failure_rate"]
    df["prepaid_x_cvc"] = df["tx_is_prepaid"] * (1 - df["tx_cvc_pass_rate"])
    df["payment_risk_score"] = (
        df["tx_is_prepaid"] + df["tx_is_virtual"] + (1 - df["tx_cvc_pass_rate"]) + df["tx_failure_rate"]
    ) / 4

    df["burst_x_short_life"] = df["burst_flag"] * df["is_short_life"]
    df["high_usage_short_life"] = df["gen_total"] * df["is_short_life"]
    df["credits_per_active_day"] = df["gen_total_credits"] / (df["gen_active_days"] + 1)
    df["abrupt_decline"] = df["day01_ratio"] * (1 - df["completion_rate"])

    df["sustained_engagement"] = df["gen_active_days"] * df["completion_rate"]
    df["exploration_score"] = df["gen_unique_types"] * df["gen_active_days"]
    df["purchase_engagement"] = df["has_purchases"] * df["gen_active_days"]
    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    clip_rules = {
        "gen_trend": (-5, 10),
        "credits_trend": (-5, 10),
        "gen_total_credits": (0, 500000),
        "gen_avg_credits": (0, 10000),
        "total_spend": (0, 500),
        "tx_avg_amount": (0, 500),
        "tx_max_amount": (0, 500),
        "gen_total": (0, 500),
        "gen_intensity": (0, 100),
    }
    for col, (lo, hi) in clip_rules.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    log_cols = [
        "gen_total",
        "gen_total_credits",
        "gen_completed",
        "gen_failed",
        "total_spend",
        "tx_success_count",
        "tx_fail_count_weighted",
        "gen_video_count",
        "gen_image_count",
        "credits_week1",
        "credits_week2",
        "gen_week1_count",
        "gen_week2_count",
    ]
    for col in log_cols:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
    return df


def add_cluster_features(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = SEED) -> None:
    scaler = StandardScaler()
    train_cluster_data = scaler.fit_transform(train_df[CLUSTER_FEATURES].fillna(0))
    test_cluster_data = scaler.transform(test_df[CLUSTER_FEATURES].fillna(0))

    n_clusters = min(6, len(train_df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    train_df["cluster_id"] = kmeans.fit_predict(train_cluster_data)
    test_df["cluster_id"] = kmeans.predict(test_cluster_data)
    print(f"Cluster distribution:\n{train_df.cluster_id.value_counts().sort_index()}")


def align_feature_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    for col in train_df.columns:
        if col not in test_df.columns and col not in ["churn_status", "target"]:
            test_df[col] = 0
    for col in test_df.columns:
        if col not in train_df.columns:
            train_df[col] = 0


def encode_categorical_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, LabelEncoder]]:
    cat_encoders: dict[str, LabelEncoder] = {}
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    for col in cat_features:
        le = LabelEncoder()
        le.fit(X_train[col])
        X_train_encoded[col] = le.transform(X_train[col])

        test_vals = X_test[col].copy()
        unseen_mask = ~test_vals.isin(le.classes_)
        if unseen_mask.any():
            test_vals[unseen_mask] = le.classes_[0]
        X_test_encoded[col] = le.transform(test_vals)
        cat_encoders[col] = le

    return X_train_encoded, X_test_encoded, cat_encoders


def add_country_target_encoding(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_train_encoded: pd.DataFrame,
    X_test_encoded: pd.DataFrame,
    y_train: np.ndarray,
    n_classes: int,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "country_code" not in X_train.columns:
        return X_train, X_test, X_train_encoded, X_test_encoded

    n_splits = max(2, min(5, len(X_train)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_series = X_train["country_code"]
    test_series = X_test["country_code"]

    for cls in range(n_classes):
        y_binary = (y_train == cls).astype(float)
        train_enc = np.zeros(len(train_series))

        for tr_idx, va_idx in kf.split(train_series):
            means = train_series.iloc[tr_idx].to_frame(name="country_code")
            means["target"] = y_binary[tr_idx]
            mapping = means.groupby("country_code")["target"].mean()
            train_enc[va_idx] = train_series.iloc[va_idx].map(mapping).fillna(y_binary[tr_idx].mean())

        feature_name = f"country_code_te_cls{cls}"
        X_train[feature_name] = train_enc
        X_train_encoded[feature_name] = train_enc

        full_mapping = train_series.to_frame(name="country_code")
        full_mapping["target"] = y_binary
        full_map = full_mapping.groupby("country_code")["target"].mean()
        X_test[feature_name] = test_series.map(full_map).fillna(y_binary.mean())
        X_test_encoded[feature_name] = X_test[feature_name]

    return X_train, X_test, X_train_encoded, X_test_encoded


def prepare_model_data(
    data_root: str | Path = ".",
    seed: int = SEED,
    max_folds: int = MAX_FOLDS,
    gen_chunksize: int = GEN_CHUNKSIZE,
) -> PreparedModelData:
    root = Path(data_root)
    train_gen_path = root / "train_users_generations.csv" / "train_users_generations.csv"
    raw = load_raw_data(root)

    train_props_feat = build_properties_features(raw["train_props"])
    test_props_feat = build_properties_features(raw["test_props"])
    print(f"Properties features: {train_props_feat.shape}")

    train_purch_feat = build_purchase_features(raw["train_purch"], raw["train_users"])
    test_purch_feat = build_purchase_features(raw["test_purch"], raw["test_users"])
    print(f"Purchase features: {train_purch_feat.shape}")
    print(f"Users without purchases: {(train_purch_feat.has_purchases == 0).sum()}")

    print("Building transaction features (train)...")
    train_tx_feat = build_transaction_features(raw["train_tx"], raw["train_purch"], raw["train_users"])
    print("Building transaction features (test)...")
    test_tx_feat = build_transaction_features(raw["test_tx"], raw["test_purch"], raw["test_users"])
    print(f"Transaction features: {train_tx_feat.shape}")

    print("Building generation features (train) - chunked reading...")
    train_gen_feat = build_generation_features_chunked(train_gen_path, raw["train_users"], chunksize=gen_chunksize)
    gc.collect()
    print("Building generation features (test)...")
    test_gen_feat = build_generation_features(raw["test_gen"], raw["test_users"])
    print(f"Generation features: {train_gen_feat.shape}")

    train_quiz_feat = build_quiz_features(raw["train_quiz"], raw["train_users"])
    test_quiz_feat = build_quiz_features(raw["test_quiz"], raw["test_users"])
    print(f"Quiz features: {train_quiz_feat.shape}")

    train_df = merge_all_features(
        raw["train_users"],
        train_props_feat,
        train_purch_feat,
        train_tx_feat,
        train_gen_feat,
        train_quiz_feat,
    )
    test_df = merge_all_features(
        raw["test_users"],
        test_props_feat,
        test_purch_feat,
        test_tx_feat,
        test_gen_feat,
        test_quiz_feat,
    )
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    train_df = clean_features(train_df)
    test_df = clean_features(test_df)
    print("Outliers clipped, log-features added")

    add_cluster_features(train_df, test_df, seed=seed)
    align_feature_columns(train_df, test_df)

    label_enc = LabelEncoder()
    train_df["target"] = label_enc.fit_transform(train_df["churn_status"])
    class_names = label_enc.classes_
    n_classes = len(class_names)
    n_folds = int(min(max_folds, train_df["target"].value_counts().min()))
    if n_folds < 2:
        raise ValueError("Need at least 2 samples in each class for CV.")

    for col in CAT_FEATURES:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna("missing").astype(str)
            test_df[col] = test_df[col].fillna("missing").astype(str)

    if "subscription_plan" in train_df.columns:
        train_df = train_df.drop(columns=["subscription_plan"])
    if "subscription_plan" in test_df.columns:
        test_df = test_df.drop(columns=["subscription_plan"])

    feature_cols = [c for c in train_df.columns if c not in ["user_id", "churn_status", "target"]]
    print(f"Classes: {class_names}")
    print(f"Total features: {len(feature_cols)}")
    print(f"Categorical: {len(CAT_FEATURES)}")
    print(f"CV folds: {n_folds}")
    print(f"\nFeature preview: {feature_cols[:15]}")

    X_train = train_df[feature_cols].copy()
    y_train = train_df["target"].values
    X_test = test_df[feature_cols].copy()

    for col in CAT_FEATURES:
        if col in X_train.columns:
            X_train[col] = X_train[col].fillna("missing").astype(str)
            X_test[col] = X_test[col].fillna("missing").astype(str)

    X_train_encoded, X_test_encoded, _ = encode_categorical_features(X_train, X_test, CAT_FEATURES)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    X_train, X_test, X_train_encoded, X_test_encoded = add_country_target_encoding(
        X_train,
        X_test,
        X_train_encoded,
        X_test_encoded,
        y_train,
        n_classes,
        seed=seed,
    )
    feature_cols = list(X_train.columns)
    print(f"After target encoding: {X_train.shape}")

    return PreparedModelData(
        train_df=train_df,
        test_df=test_df,
        X_train=X_train,
        X_test=X_test,
        X_train_encoded=X_train_encoded,
        X_test_encoded=X_test_encoded,
        y_train=y_train,
        feature_cols=feature_cols,
        cat_features=CAT_FEATURES,
        label_enc=label_enc,
        class_names=class_names,
        n_classes=n_classes,
        n_folds=n_folds,
        seed=seed,
    )
