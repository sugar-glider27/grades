from pathlib import Path
import pandas as pd


def normalize_login(value):
    if pd.isna(value):
        return None
    return str(value).strip().lower().split("@")[0]


def process_redemption(hw, canvas_csv, original_csv, redemption_csv,
                       checkpoint_csv=None, manual_csv=None,
                       output_csv=None, redemption_credit=0.8):

    canvas_csv = Path(canvas_csv)
    original_csv = Path(original_csv)
    redemption_csv = Path(redemption_csv)
    output_csv = (
        Path(output_csv) if output_csv is not None
        else canvas_csv.parent / f"canvas_{hw}_updated.csv"
    )

    for path in (canvas_csv, original_csv, redemption_csv):
        if not path.exists():
            raise FileNotFoundError(path)

    if checkpoint_csv is not None:
        checkpoint_csv = Path(checkpoint_csv)
        if not checkpoint_csv.exists():
            raise FileNotFoundError(checkpoint_csv)

    if manual_csv is not None:
        manual_csv = Path(manual_csv)
        if not manual_csv.exists():
            raise FileNotFoundError(manual_csv)

    canvas = pd.read_csv(canvas_csv)
    original = pd.read_csv(original_csv)
    checkpoint = pd.read_csv(checkpoint_csv) if checkpoint_csv is not None else None
    redemption = pd.read_csv(redemption_csv)
    manual = pd.read_csv(manual_csv) if manual_csv is not None else None

    canvas["_login"] = canvas["SIS Login ID"].map(normalize_login)
    original["_login"] = original["Email"].map(normalize_login)
    redemption["_login"] = redemption["Email"].map(normalize_login)
    if checkpoint is not None:
        checkpoint["_login"] = checkpoint["Email"].map(normalize_login)
    if manual is not None:
        manual["_login"] = manual["Email"].map(normalize_login)

    hw_columns = [
        col for col in canvas.columns
        if hw.lower() in str(col).lower()
    ]
    if len(hw_columns) != 1:
        raise ValueError(
            f"Expected exactly one Canvas column containing '{hw}', "
            f"but found {hw_columns}"
        )
    hw_col = hw_columns[0]

    original_max = original["Max Points"].dropna().unique()
    redemption_max = redemption["Max Points"].dropna().unique()

    if len(original_max) != 1 or len(redemption_max) != 1:
        raise ValueError("Could not determine a unique max score.")
    if original_max[0] != redemption_max[0]:
        raise ValueError(
            f"Max scores differ: original={original_max[0]}, "
            f"redemption={redemption_max[0]}"
        )

    checkpoint_max = 0
    if checkpoint is not None:
        checkpoint_max_values = checkpoint["Max Points"].dropna().unique()

        if len(checkpoint_max_values) != 1:
            raise ValueError(
                f"Could not determine a unique checkpoint max score: "
                f"{checkpoint_max_values}"
            )
        checkpoint_max = checkpoint_max_values[0]

    manual_max = 0
    if manual is not None:
        manual_max_values = manual["Max Points"].dropna().unique()
        if len(manual_max_values) != 1:
            raise ValueError(
                f"Could not determine a unique manual max score: "
                f"{manual_max_values}"
            )
        manual_max = manual_max_values[0]
        if original_max[0] + manual_max != 100:
            raise ValueError(
                f"Autograder max ({original_max[0]}) + manual max "
                f"({manual_max}) should be 100."
            )
        if not manual["Total Score"].dropna().between(0, manual_max).all():
            raise ValueError(
                f"Manual scores must be between 0 and {manual_max}."
            )

    orig = original[
        ["_login", "First Name", "Last Name", "Email", "Total Score"]
    ].rename(columns={
        "First Name": "first_name",
        "Last Name": "last_name",
        "Email": "email",
        "Total Score": "original_score",
    })

    red = redemption[
        ["_login", "First Name", "Last Name", "Email", "Total Score"]
    ].rename(columns={
        "First Name": "first_name",
        "Last Name": "last_name",
        "Email": "email",
        "Total Score": "redemption_score",
    })

    grades = orig.merge(
        red,
        on="_login",
        how="outer",
        suffixes=("_original", "_redemption"),
        validate="one_to_one",
    )

    grades["first_name"] = grades["first_name_original"].fillna(
        grades["first_name_redemption"]
    )
    grades["last_name"] = grades["last_name_original"].fillna(
        grades["last_name_redemption"]
    )
    grades["email"] = grades["email_original"].fillna(
        grades["email_redemption"]
    )

    grades = grades[[
        "_login", "first_name", "last_name", "email",
        "original_score", "redemption_score"
    ]].copy()

    if checkpoint is not None:
        checkpoint = checkpoint.rename(columns={
            "First Name": "first_name_checkpoint",
            "Last Name": "last_name_checkpoint",
            "Email": "email_checkpoint",
            "Total Score": "checkpoint_score",
        })

        grades = grades.merge(
            checkpoint[[
                "_login", "first_name_checkpoint", "last_name_checkpoint",
                "email_checkpoint", "checkpoint_score"
            ]],
            on="_login",
            how="outer",
            validate="one_to_one",
        )

        for col in ["first_name", "last_name", "email"]:
            grades[col] = grades[col].fillna(grades[f"{col}_checkpoint"])

        grades = grades.drop(columns=[
            "first_name_checkpoint",
            "last_name_checkpoint",
            "email_checkpoint",
        ])
    else:
        grades["checkpoint_score"] = float("nan")

    if manual is not None:
        manual = manual.rename(columns={
            "First Name": "first_name_manual",
            "Last Name": "last_name_manual",
            "Email": "email_manual",
            "Total Score": "manual_score",
        })

        grades = grades.merge(
            manual[[
                "_login", "first_name_manual", "last_name_manual",
                "email_manual", "manual_score"
            ]],
            on="_login",
            how="outer",
            validate="one_to_one",
        )

        for col in ["first_name", "last_name", "email"]:
            grades[col] = grades[col].fillna(grades[f"{col}_manual"])

        grades = grades.drop(columns=[
            "first_name_manual",
            "last_name_manual",
            "email_manual",
        ])
    else:
        grades["manual_score"] = float("nan")

    # No original submission => 0 points as an original score
    grades["original_score"] = grades["original_score"].fillna(0)

    if not grades["checkpoint_score"].dropna().between(0, checkpoint_max).all():
        raise ValueError(
            f"Checkpoint scores must be between 0 and {checkpoint_max}."
        )

    grades["used_redemption"] = (
        grades["redemption_score"].notna()
        & (grades["redemption_score"] > grades["original_score"])
    )

    improvement = (
        grades["redemption_score"] - grades["original_score"]
    ).clip(lower=0).fillna(0)

    # Manual (Part-1) score is added after redemption, so it is never redeemed
    grades["final_score"] = (
        grades["original_score"]
        + redemption_credit * improvement
        + grades["checkpoint_score"].fillna(0)
        + grades["manual_score"].fillna(0)
    ).round(2)

    submitted_redemption = grades.loc[
        grades["redemption_score"].notna(),
        ["first_name", "last_name", "original_score",
         "redemption_score", "checkpoint_score", "manual_score", "final_score"],
    ].copy()

    submitted_checkpoint = grades.loc[
        grades["checkpoint_score"].notna(),
        ["first_name", "last_name", "original_score",
         "redemption_score", "checkpoint_score", "manual_score", "final_score"],
    ].copy()


    unmatched = grades.loc[
        ~grades["_login"].isin(canvas["_login"]),
        ["first_name", "last_name", "email", "original_score",
         "redemption_score", "checkpoint_score", "manual_score", "final_score"],
    ].copy()

    score_map = grades.set_index("_login")["final_score"]
    matched = canvas["_login"].isin(score_map.index)
    canvas.loc[matched, hw_col] = canvas.loc[matched, "_login"].map(score_map)

    # Preview the actual values that will be uploaded
    canvas_preview = canvas.loc[
        matched, ["Student", "SIS Login ID", hw_col]
    ].copy()

    canvas.drop(columns="_login").to_csv(output_csv, index=False)

    return {
        "hw_col": hw_col,
        "max_points": original_max[0],
        "grades": grades,
        "checkpoint": checkpoint,
        "checkpoint_max_points": checkpoint_max,
        "manual_max_points": manual_max,
        "submitted_redemption": submitted_redemption,
        "submitted_checkpoint": submitted_checkpoint,
        "unmatched": unmatched,
        "canvas_preview": canvas_preview,
        "output_csv": output_csv,
        "matched_count": int(matched.sum()),
        "increased_count": int(grades["used_redemption"].sum()),
        "not_increased_count": int((
            grades["redemption_score"].notna()
            & ~grades["used_redemption"]
        ).sum()),
    }
