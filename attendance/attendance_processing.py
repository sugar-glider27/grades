import pandas as pd

def find_assignment_column(gradebook, assignment_name):
    matches = [
        col for col in gradebook.columns
        if col.startswith(assignment_name + " ")
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one column for '{assignment_name}', "
            f"found {len(matches)}: {matches}"
        )

    return matches[0]



def load_lecture_attendance(filename, session, threshold=0.75):
    attendance = pd.read_csv(filename, header=[0, 1])

    columns = []
    current_session = None

    for top, bottom in attendance.columns:
        if not str(top).startswith("Unnamed"):
            current_session = top

        columns.append((current_session, bottom))

    attendance.columns = pd.MultiIndex.from_tuples(columns)

    if session not in attendance.columns.get_level_values(0):
        raise ValueError(
            f"Session not found: {session}"
        )
    
    polls = attendance[session]
    total_polls = polls.shape[1]
    answered = polls.notna().sum(axis=1)


    lecture_grade = attendance[[
        ("Email", "Unnamed: 0_level_1"),
        ("Last Name", "Unnamed: 1_level_1"),
        ("First Name", "Unnamed: 2_level_1")
    ]].copy()

    lecture_grade.columns = [
        "Email",
        "Last Name",
        "First Name"
    ]


    lecture_grade["Email"] = (
        lecture_grade["Email"].str.strip().str.lower()
    )

    lecture_grade["First Name"] = (
       lecture_grade["First Name"].str.strip().str.title()
    )

    lecture_grade["Last Name"] = (
        lecture_grade["Last Name"].str.strip().str.title()
    )

    lecture_grade["Name"] = (
        lecture_grade["Last Name"] + ", " + lecture_grade["First Name"]
    )

    lecture_grade["Answered"] = answered
    lecture_grade["Total"] = total_polls
    lecture_grade["Rate"] = (
        lecture_grade["Answered"] / lecture_grade["Total"]
    )
    lecture_grade["Grade"] = (
        lecture_grade["Rate"] >= threshold
    ).astype(int)

    lecture_grade = lecture_grade[[
        "Email", "Name", "Answered", "Total", "Rate", "Grade"
    ]]

    lecture_grade = (
        lecture_grade.sort_values("Name").reset_index(drop=True)
    )

    return lecture_grade




def load_attendance(filename, session, poll):
    attendance = pd.read_csv(filename, header=[0, 1])
    attendance_need = attendance[[
        ("Email", "Unnamed: 0_level_1"),
        ("Last Name", "Unnamed: 1_level_1"),
        ("First Name", "Unnamed: 2_level_1"),
        (session, poll)
    ]].copy()
    
    attendance_need.columns = [
        "Email", "Last Name", "First Name", "Attendance"]
    
    attendance_need["First Name"] = attendance_need["First Name"].str.strip().str.title()
    attendance_need["Last Name"] = attendance_need["Last Name"].str.strip().str.title()
    attendance_need["Email"] = attendance_need["Email"].str.strip().str.lower()
    attendance_need["Attendance"] = attendance_need["Attendance"].notna().astype(int)

    attendance_need["Name"] = attendance_need["Last Name"] + ", " + attendance_need["First Name"]

    return attendance_need[["Email", "Name", "Attendance"]]




def load_gradescope(filename):
    gradescope = pd.read_csv(filename)

    manual_check_columns = [
        col for col in gradescope.columns
        if "Manual Check" in col
    ]

    if len(manual_check_columns) != 1:
        raise ValueError(
            f"Expected exactly one Manual Check column, "
            f"found {len(manual_check_columns)}: {manual_check_columns}"
        )

    manual_check_column = manual_check_columns[0]
    
    gradescope_need = gradescope[[
        "Email", "Last Name","First Name", manual_check_column
    ]].copy()

    gradescope_need = gradescope_need.rename(
        columns={manual_check_column: "Total Score"}
    )

    gradescope_need["First Name"] = gradescope_need["First Name"].str.strip().str.title()
    gradescope_need["Last Name"] = gradescope_need["Last Name"].str.strip().str.title()
    gradescope_need["Email"] = gradescope_need["Email"].str.strip().str.lower()

    gradescope_need["Total Score"] = gradescope_need["Total Score"].fillna(0).ne(0).astype(int)
    gradescope_need["Name"] = gradescope_need["Last Name"] + ", " + gradescope_need["First Name"]
    return gradescope_need[["Email", "Name", "Total Score"]]




def combine_grades(attendance, gradescope):
    grade = attendance.merge(
        gradescope, on="Email", how="outer", suffixes=("_attendance", "_gradescope"))
    
    grade["Attendance"] = grade["Attendance"].fillna(0)
    grade["Total Score"] = grade["Total Score"].fillna(0)
    
    grade["Name"] = grade["Name_attendance"].combine_first(
        grade["Name_gradescope"]
    )
    
    grade["Grade"] = (
        (grade["Attendance"] == 1) & (grade["Total Score"] == 1)
    ).astype(int)

    grade = grade[["Email", "Name", "Attendance", "Total Score", "Grade"]]

    return grade





def update_gradebook(filename, new_grades, assignment_column):
    gradebook = pd.read_csv(filename)

    actual_column = find_assignment_column(
        gradebook,
        assignment_column
    )

    original_gradebook = gradebook.copy(deep=True)

    new_grades = new_grades.copy()
    
    new_grades["SIS Login ID"] = (
        new_grades["Email"]
        .str.strip()
        .str.lower()
        .str.replace("@ucsd.edu", "", regex=False)
    )

    canvas_logins = (
        gradebook["SIS Login ID"]
        .dropna()
        .str.strip()
        .str.lower()
    )

    new_logins = set(new_grades["SIS Login ID"])

    not_in_canvas = new_grades[
        ~new_grades["SIS Login ID"].isin(canvas_logins)
    ].copy()

    not_in_new_grades = gradebook[
        gradebook["SIS Login ID"].notna()
        & ~gradebook["SIS Login ID"]
            .str.strip()
            .str.lower()
            .isin(new_logins)
    ].copy()

    grade_map = new_grades.set_index("SIS Login ID")["Grade"]

    mask = (
        gradebook["SIS Login ID"]
        .fillna("")
        .str.strip()
        .str.lower()
        .isin(grade_map.index)
    )

    gradebook.loc[mask, actual_column] = (
        gradebook.loc[mask, "SIS Login ID"]
        .str.strip()
        .str.lower()
        .map(grade_map)
    )

    changes = pd.DataFrame({
        "Student": gradebook["Student"],
        "SIS Login ID": gradebook["SIS Login ID"],
        "Before": original_gradebook[actual_column],
        "After": gradebook[actual_column]
    })

    changes = changes[
        ~changes["Before"].fillna("__MISSING__").eq(
            changes["After"].fillna("__MISSING__")
        )
    ]

    points_possible_unchanged = original_gradebook.loc[0].equals(gradebook.loc[0])

    other_columns = [
        col for col in gradebook.columns
        if col != actual_column
    ]

    other_columns_unchanged = (
        original_gradebook[other_columns].equals(
            gradebook[other_columns]
        )
    )

    if not points_possible_unchanged:
        raise ValueError("Safety check failed: The first row of gradebook (points possible) changed!")

    if not other_columns_unchanged:
        raise ValueError("Safety check failed: Other gradebook columns changed!")

    return (
        gradebook,
        not_in_canvas,
        not_in_new_grades,
        changes
    )