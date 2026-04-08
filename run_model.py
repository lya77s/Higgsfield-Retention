#!/usr/bin/env python3
from __future__ import annotations

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

from retention_pipeline import prepare_model_data

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def train_primary_model():
    data = prepare_model_data()

    print(f"\nTraining primary RandomForest model with {data.n_folds} folds...")

    oof_proba = pd.DataFrame(
        0.0,
        index=range(len(data.X_train_encoded)),
        columns=[f"prob_{name}" for name in data.class_names],
    )
    test_proba = pd.DataFrame(
        0.0,
        index=range(len(data.X_test_encoded)),
        columns=[f"prob_{name}" for name in data.class_names],
    )

    skf = StratifiedKFold(
        n_splits=data.n_folds,
        shuffle=True,
        random_state=data.seed,
    )

    for fold, (tr_idx, va_idx) in enumerate(skf.split(data.X_train_encoded, data.y_train), 1):
        print(f"RF fold {fold}/{data.n_folds}...")
        X_tr = data.X_train_encoded.iloc[tr_idx]
        X_va = data.X_train_encoded.iloc[va_idx]
        y_tr = data.y_train[tr_idx]

        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=data.seed + fold,
        )
        model.fit(X_tr, y_tr)

        oof_proba.iloc[va_idx] = model.predict_proba(X_va)
        test_proba += model.predict_proba(data.X_test_encoded) / data.n_folds

    oof_labels = oof_proba.to_numpy().argmax(axis=1)
    rf_f1 = f1_score(data.y_train, oof_labels, average="macro")
    print(f"\nRandomForest OOF macro F1: {rf_f1:.4f}")
    print(classification_report(data.y_train, oof_labels, target_names=data.class_names))

    test_predictions = test_proba.to_numpy().argmax(axis=1)
    submission = pd.DataFrame(
        {
            "user_id": data.test_df["user_id"],
            "churn_status": data.label_enc.inverse_transform(test_predictions),
        }
    )

    submission_detailed = pd.DataFrame(
        {
            "user_id": data.test_df["user_id"],
            "predicted_class": data.label_enc.inverse_transform(test_predictions),
        }
    )
    for col in test_proba.columns:
        submission_detailed[col] = test_proba[col].to_numpy()
    submission_detailed["confidence"] = test_proba.max(axis=1).to_numpy()

    submission.to_csv("submission.csv", index=False)
    submission_detailed.to_csv("submission_detailed.csv", index=False)

    print("\nSaved: ./submission.csv")
    print("Saved: ./submission_detailed.csv")
    print(f"Prediction distribution:\n{submission.churn_status.value_counts(normalize=True)}")


def main() -> None:
    train_primary_model()


if __name__ == "__main__":
    main()
