from pathlib import Path
import pandas as pd


def normalize_login(value):
    if pd.isna(value):
        return None
    return str(value).strip().lower().split("@")[0]


def process_redemption(hw, canvas_csv, original_csv, redemption_csv,
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

    canvas = pd.read_csv(canvas_csv)
    original = pd.read_csv(original_csv)
    redemption = pd.read_csv(redemption_csv)

    canvas["_login"] = canvas["SIS Login ID"].map(normalize_login)
    original["_login"] = original["Email"].map(normalize_login)
    redemption["_login"] = redemption["Email"].map(normalize_login)

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

    # No original submission => 0 points as an original score
    grades["original_score"] = grades["original_score"].fillna(0)

    grades["used_redemption"] = (
        grades["redemption_score"].notna()
        & (grades["redemption_score"] > grades["original_score"])
    )

    improvement = (
        grades["redemption_score"] - grades["original_score"]
    ).clip(lower=0).fillna(0)

    grades["final_score"] = (
        grades["original_score"] + redemption_credit * improvement
    ).round(2)

    submitted_redemption = grades.loc[
        grades["redemption_score"].notna(),
        ["first_name", "last_name", "original_score",
         "redemption_score", "final_score"],
    ].copy()

    unmatched = grades.loc[
        ~grades["_login"].isin(canvas["_login"]),
        ["first_name", "last_name", "email", "original_score",
         "redemption_score", "final_score"],
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
        "submitted_redemption": submitted_redemption,
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
