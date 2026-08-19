import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# QC thresholds
R2_PASS_THRESHOLD = 0.98
STD_RECOVERY_LOW = 75.0 
STD_RECOVERY_HIGH = 125.0
CV_PASS_THRESHOLD = 25.0
MIN_OD_FOR_CV_CHECK = 0.05

#4PL Model

def four_pl(logx, A, B, logEC50, D):
    """ Four-parameter logistic (4PL) model."""

    return D + (A - D) / (1 + 10 ** ((logEC50 - logx) * B))

# Load excel file

def load_workbook_grid(workbook_path, wanted_od):
    """ Load the ELISA data from selected OD chennel."""

    workbook_path = Path(workbook_path)

    if not workbook_path.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook_path}")
    print(f"\nLoading Excel file: {workbook_path}")

    # Read workbook information
    workbook = pd.ExcelFile(workbook_path)

    # Try the expected sheet name first
    possible_sheets = [
        "Results by plate",
        "Results",
        "Sheet1",
        "Results_by_plate"
    ]

    selected_sheet = next(
        (sheet for sheet in possible_sheets if sheet in workbook.sheet_names),
        None
    )

    if selected_sheet is None:
        raise ValueError(
            f"\nCould not find the Results sheet.\n"
            f"\nAvailable sheets: {workbook.sheet_names}"
        )

    print(f"Using sheet: {selected_sheet}")

    #Read without assuming header

    df = pd.read_excel(
        workbook_path, 
        sheet_name=selected_sheet, 
        header=None
        )

    # Find OD channel
    matches = df.iloc[:, 0].astype(str).str.strip().str.lower()
    matches = matches[matches == wanted_od.strip().lower()]
    if matches.empty:
        raise ValueError(
            f"Could not find '{wanted_od}' in sheet '{selected_sheet}'."
        )

    wanted_idx = matches.index[0]
    print(f"Found '{wanted_od}' at Excel row: {wanted_idx + 1}")

    new_df = df.iloc[wanted_idx:].reset_index(drop=True)
    new_df = new_df.iloc[2:,2:].reset_index(drop=True)

    print("ELISA data extracted")
    return new_df

#Defining blanks samples

def find_blank_wells(label_data):
    """ Return positions of blank wells labelled as blank."""
    mask = label_data.apply(
        lambda col: col.astype(str).str.strip().str.lower().str.contains("blank",na=False)
    )
    return [
        (row, col)
        for col in label_data.columns
        for row in range(len(label_data))
        if mask[col].iloc[row]
    ]

# Find standards

def find_standard_columns(label_data):
    """Return columns whose first label starts with STD."""
    return [
        col for col in label_data.columns
        if pd.notna(label_data[col].iloc[0])
        and str(label_data[col].iloc[0]).strip().lower().startswith("std")
    ]

#PREPROCESS ELISA DATA

def preprocess_elisa_data(df):
    """ Blank-correct OD values and calculate duplicate averages and CVs."""

    print("\nPreprocessing ELISA data...")

    # First 8 rows = OD values
    od_data = df.iloc[:8].apply(pd.to_numeric, errors ="coerce")

    #Remaining rows = sample labels
    label_data = df.iloc[8:].copy()

    #Blank calculation
    blank_wells = find_blank_wells(label_data)

    if not blank_wells:
        raise ValueError(
            "No wells labeled 'blank' were found in label_data."
            " Please check your blank wells are labeled correctly."
        )

    #pull the OD values for each identified blank well
    blank_values = [
        od_data[col].iloc[row] 
        for row, col in blank_wells
    ]
    
    blank_avg = np.mean(blank_values)

    print(f"Found blank wells (row, column): {blank_wells}")
    print(f"Average blank OD: {blank_avg:.6f}")

    #Blank-correction

    od_corrected = (od_data - blank_avg)

    #Exclude any column that is entirely blank wells from sample averaging loop
    #(a column only gets excluded if every row in it was flagged as blank)

    blank_counts = pd.Series(
        [col for _, col in blank_wells]
    ).value_counts()

    fully_blank_cols = blank_counts[
        blank_counts == len(od_data)
    ].index

    sample_cols = [
        col for col in od_corrected.columns
        if col not in fully_blank_cols
    ]

    if len(sample_cols) % 2:
        raise ValueError(
            "An even number of sample columns in required for duplicate measurements."
        )
    #Calculate coefficient of variation (CV)

    averages = {}
    cvs = {}

    for i in range(0, len(sample_cols), 2):
        name = f"Sample_{i // 2 + 1}"
        pair = od_corrected[sample_cols[i:i + 2]]
        mean = pair.mean(axis=1)
        averages[name] = mean
        cvs[name] = pair.std(axis=1) / mean * 100

    corrected_df = pd.DataFrame(averages)
    cv_df = pd.DataFrame(cvs)

    print("\nBlank-corrected duplicate averages:")
    print(corrected_df.to_string(index=False))

    return corrected_df, cv_df, label_data

def plot_fit(x_fit, y_fit, A, B, logEC50, D, r2):
    """Display the fitted 4PL calibration curve."""

    x_smooth = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 300)

    y_smooth = four_pl(np.log10(x_smooth), A, B, logEC50, D)

    fig, ax = plt.subplots(figsize=(7, 5))

    fig.patch.set_facecolor('none')
    ax.patch.set_color('none')

    ax.scatter(
        x_fit,
        y_fit,
        color='black',
        zorder=5,
        label='Data Points',
        s=50
    )

    ax.plot(
        x_smooth,
        y_smooth,
        color='red',
        linewidth=2,
        label= f"4PL Fit (R² = {r2:.4f})"
    )
    ax.set_xscale('log')
    ax.set_xlabel('Concentration (ng/mL)', fontsize=12)
    ax.set_ylabel('OD(450) - OD(570)', fontsize=12)
    ax.set_title(f'4PL Calibration Curve (R² = {r2:.4f})', fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.show()

# Fit 4PL curve

def fit_4pl(x, y):
    """Fit the standard-curve OD values to a 4PL model."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    #Remove non-positive/zero concentration
    mask = x > 0

    x = x[mask]
    y = y[mask]

    if len(x) < 4:
        raise ValueError(
            "At least four positive concentration points are required for fitting a 4PL curve."
        )

    logx = np.log10(x)

    #Initial guesses
    p0 = [max(y) * 2,1.4,np.log10(np.median(x)),min(y)]

    #parameter bounds
    bounds = (
        [0, 0.01, -1, -0.05],
        [50, 10, 4, 0.5] 
    )

    print("\nFitting 4PL calibration curve...")

    parameters, _ = curve_fit(
        four_pl,
        logx,
        y,
        p0=p0,
        bounds=bounds,
        method='trf',
        ftol=1e-15,
        xtol=1e-15,
        gtol=1e-15,
        maxfev=100000,
    )
    A, B, logEC50, D = parameters

    #predicted values
    y_pred = four_pl(logx, A, B, logEC50, D)

    # R2
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = np.nan if ss_tot == 0 else 1 - ss_res / ss_tot
    
    return (A, B, logEC50, D, r2, x, y)

#Calculate STD difference
def calculate_std_percent_diff(sample_name, sample_type, concentration):
    """Calculate % difference bertween calculated and nominal STD concentration."""
    if sample_name.strip().upper() != "STD" or pd.isna(concentration):
        return np.nan
    try:
        nominal = float(str(sample_type).strip().replace(",","."))
    except (ValueError, TypeError):
        return np.nan
    if nominal == 0:
        return np.nan

    return ((concentration - nominal) / nominal) * 100

#Calculate sample concentrations

def calculate_results(df, corrected_df, label_data, A, B, logEC50, D):
    """Back-calculate concentration for all labeled samples."""

    results = []
    columns = df.columns
    C = 10 ** logEC50

    #loop through each pair
    for p in range(0, len(columns), 2):
        col =columns[p]
        sample_col = f"Sample_{p // 2 + 1}"
        #make sure the corresponding averaged OD exists
        if sample_col not in corrected_df.columns:
            continue
        #Each pair has 8 rows of samples
        for i, label in enumerate(label_data[col].iloc[:8]):
            if pd.isna(label):
                continue
            label_str = str(label).strip()
            if not label_str:
                continue

            #Extract dilution (Example 10x, 100x, etc.)
            dilution_match = re.search(
                r"(\d+)\s*x\b",
                label_str,
                re.IGNORECASE
            )
            dilution = (
                int(dilution_match.group(1))
                if dilution_match
                else 1
            )

            #Extract sample name and type
            parts = label_str.split(maxsplit=1)

            sample_name = parts[0]
            sample_type = parts[1] if len(parts) > 1 else ""

            if dilution_match:
                sample_type = sample_type.replace(dilution_match.group(0), "")

            sample_type = re.sub(r"\s+", " ", sample_type).strip()

            #OD
            y = corrected_df[sample_col].iloc[i]

            #Back-calculate concentration
            if (pd.isna(y) or y <= D or y >= A):
                concentration = np.nan
            else:
                ratio = (A - D) / (y - D) - 1

                concentration = (
                    C * ratio ** (-1 / B)
                    if ratio > 0
                    else np.nan
                )
            calculated = concentration * dilution
            results.append({
                "Sample": sample_name,
                "Sample Type": sample_type,
                "Dilution": dilution,
                "OD": y,
                "_raw_concentration_internal": concentration,
                "Concentration (ng/mL)": calculated,
                "Concentration (µg/mL)": calculated / 1000,
                "STD % Difference": calculate_std_percent_diff(
                    sample_name,sample_type, concentration
                )
            })
    return pd.DataFrame(results)
        
#Save results to CSV

def save_results_to_csv(results_df, output_path='sample_concentration_results.csv'):
    """Save calculated concentration to CSV"""
    output_path = Path(output_path)

    results_df.drop(
        columns = [
            c for c in results_df.columns
            if c.startswith("_")
        ]
    ).to_csv(output_path, index=False, encoding = 'utf-8-sig')
    print(f"\nResults saved to: {output_path.resolve()}")

# Function for deciding if sample passes QC criteria
def qc_status(passed, total, warning_fraction = 0.75):
    """Return PASS, WARNING, or FAIL based on the proportion passing."""
    if total == 0:
        return "FAIL"
    fraction = passed / total

    if passed == total:
        return "PASS"
    if fraction >= warning_fraction:
        return "WARNING"
    return "FAIL" 
# QC summary

def generate_qc_summary(r2, results_df, cv_df, std_low, std_high, x, corrected_df):
    """Generate QC summary for assay."""    
    lines =[
        "=" * 95,
        "ELISA ANALYSIS QC SUMMARY".center(95),
        "=" * 95
    ]
    statuses = []

    # Curve fit R2
    r2_status = "PASS" if r2 >= R2_PASS_THRESHOLD else "FAIL"
    statuses.append(r2_status)
    lines.append(f"{'Curve fit R2':<45}{r2:<25.4f}{r2_status}")

    #Standard recovery
    std_rows = results_df[
        results_df["Sample"].str.upper() == "STD"
        ].dropna(subset = ["STD % Difference"]).copy()
    std_rows["STD % Difference"] = 100 + std_rows["STD % Difference"]

    n_std = len(std_rows)
    n_std_pass = (
        std_rows["STD % Difference"].between(
            std_low,
            std_high
        ).sum()
    )

    recovery_status = qc_status(n_std_pass, n_std)
    statuses.append(recovery_status)
    lines.append(
        f"{'Standard recovery':<45}"
        f"{f'{n_std_pass} / {n_std} within limits':<25}{recovery_status}"
    )

    #Lowest standard recovery
    if not std_rows.empty:
        std_rows["Nominal"] = pd.to_numeric(
            std_rows["Sample Type"].astype(str).str.replace(",", ".",regex=False),
            errors ="coerce"
        )

        std_rows["Recovery %"] = (
            std_rows["Concentration (ng/mL)"] /std_rows["Nominal"] * 100
        )
        lowest_index = std_rows["Nominal"].idxmin()
        lowest = std_rows.loc[lowest_index]
        lowest_recovery =lowest["Recovery %"]
        lowest_label = lowest["Nominal"]

        lowest_status = (
            "PASS" 
            if std_low <= lowest_recovery <= std_high 
            else "REVIEW"
        )
        lines.append(
            f"{'Lowest standard':<45}"
            f"{f'{lowest_recovery:.1f}% (STD {lowest_label:.3f})':<25}{lowest_status}"
            )

    # Duplicate precicion (CV)
    cv_values = cv_df.values.flatten()
    mean_values = corrected_df.values.flatten()

    valid = (
        ~np.isnan(cv_values) 
        & (mean_values >= MIN_OD_FOR_CV_CHECK))
    cv_valid = cv_values[valid]

    n_cv = len(cv_valid)
    n_cv_pass = int(np.sum(cv_valid <= CV_PASS_THRESHOLD))

    cv_status = qc_status(n_cv_pass, n_cv)
    statuses.append(cv_status)
    lines.append(
        f"{'Duplicate precision':<45}"
        f"{f'{n_cv_pass} / {n_cv} within limits':<25}{cv_status}"
    )

    #Sample range (within calibration range)
    sample_rows = results_df[
        results_df["Sample"].str.upper() !="STD"
        ].dropna(subset=["_raw_concentration_internal"])
    x_min, x_max = x.min(), x.max()

    in_range = sample_rows[
        "_raw_concentration_internal"
    ].between(x_min, x_max)

    n_sample = len(sample_rows)
    n_in_range = in_range.sum()

    if n_sample == 0:
        range_status = "PASS"
    elif n_in_range == n_sample:
        range_status = "PASS"
    elif n_in_range / n_sample >= 0.75:
        range_status = "Check results"
    else:
        range_status = "FAIL"
    statuses.append(range_status)

    lines.append(
        f"{'Sample range':<45}"
        f"{f'{n_in_range} / {n_sample} in range':<25}{range_status}"
    )
    #Overall status

    overall = (
        "FAIL"
        if "FAIL" in statuses
        else "REVIEW"
        if "WARNING" in statuses or "REVIEW" in statuses
        else "PASS"
    )

    lines.append(f"OVERALL RUN STATUS: {overall}")
    lines.append("=" * 95)

    return "\n".join(lines), overall

#Helper
def get_argument_or_input(value, prompt,converter=str):
    if value is not None:
        return value
    return converter(input(prompt).strip())

#Main

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Automated 4PL ELISA analysis"
            "from an Excel workbook"
        )
    )

    parser.add_argument(
        'excel_path',
        nargs='?',
        default=None,
        help=(
            "Path to the Excel workbook. If omitted, the program will ask for it."
        )
    )

    parser.add_argument(
        '--od-channel',
        default = None,
        help = (
            "The OD channel label to search for in the sheet"
            "(eg. 'OD(450)' or 'OD(450) - OD(570)'. If omitted, the programme will ask for it.)"
        )
    )

    parser.add_argument(
        '--std-start',
        type = float,
        default = None,
        help = "Starting (highest) standard concentration in ng/mL. if omitted, the program will ask for it."
    )

    parser.add_argument(
        '--dilution-factor',
        type = float,
        default = None,
        help = "Serial dilution factor applied at each standard step (e.g. 0.5). If omitted, the program will ask for it."
    )

    parser.add_argument(
        '--num-standards',
        type = int,
        default = None,
        help = "Number of standard curve points. if omitted, the program will ask for it."
    )

    parser.add_argument(
        '--output',
        default = None,
        help=(
            "Output CSV filename. If omitted, the file is generated"
            "from the run ID."
        )
    )

    args = parser.parse_args()

    #Get excel file

    if args.excel_path:
        excel_path = args.excel_path
    else:
        print("\n=========================")
        print(" AUTOMATED 4PL ELISA ANALYSIS")
        print("=========================\n")

        excel_path = input(
        "Enter the path to the Excel file:\n> "
        ).strip().strip('"')

    if not excel_path:
        raise ValueError(
            "No Excel file was provided."
        )

    #Choose OD channel
    if args.od_channel:
        wanted_od = args.od_channel
    else:
        wanted_od = input(
            "Enter the OD channel to look for "
            "(e. g. 'OD(450) - OD(570)'):\n> "
        ).strip()

    if not wanted_od:
        raise ValueError("No OD channel label was provided")

    #Standard curve parameters
    first_std_concentration = get_argument_or_input(
        args.std_start,
        "Enter the starting (highest) standard concentration in ng/mL (e.g. 50):\n> ",
        float
    )
    dilution_factor = get_argument_or_input(
        args.dilution_factor,
        "Enter the serial dilution (e.g. 0.5):\n ",
        float
    )
    number_of_standards = get_argument_or_input(
        args.num_standards,
        "Enter the number of standard curve point (e.g. 8):\n> ",
        int
    )

    standard_concentrations = [
        first_std_concentration * (dilution_factor ** i) for i in range(number_of_standards)
    ]

    #load excel data

    df = load_workbook_grid(excel_path, wanted_od)

    #preprocess

    corrected_df, cv_df, label_data = preprocess_elisa_data(df)

    #Detect standard curve columns
    std_cols_raw = find_standard_columns(label_data)

    if not std_cols_raw:
        raise ValueError(
            "No columns labeled 'STD' were found. "
            "Check that your standard curve wells are labeled correctly."
        )

    print(f"Detected standard curve columns: {std_cols_raw}")

    #Map raw columns to their corresponding averaged Sample_N columns in corected_df

    columns = list(df.columns)

    std_sample_names = list(dict.fromkeys(
        f"Sample_{columns.index(col) // 2 +1}"
        for col in std_cols_raw
    ))
    print(f"Standard curve maps to averages columns: {std_sample_names}")

    #Build y by concatenating OD values from each detected standard column, in order
    y = np.concatenate([
        corrected_df[name].iloc[:8].to_numpy()
        for name in std_sample_names
    ])

    x = np.array(standard_concentrations, dtype = float)

    if len(x) != len(y):
        raise ValueError(
            f"Mismatch: {len(x)} standard concentration were specified, "
            f"but {len(y)} standard OD values were found across the detected STD columns "
            f"({std_sample_names}). Check --num-standards or your plate labeling."
        )
    
    #Fit 4PL

    (A, B, logEC50, D, r2, x_fit, y_fit) = fit_4pl(x, y)

    C = 10 ** logEC50

    #Print curve parameters
    print("\n==========================")
    print(" 4PL CURVE RESULTS")
    print("==========================")

    print(f"Top Asymptote (A): {A:.6f}")
    print(f"Slope (B): {B:.6f}")
    print(f"Inflection point (C): {C:.6f} ng/mL")
    print(f"Bottom Asymptote (D): {D:.6f}")
    print(f"Standard Curve Coefficient (R²): {r2:.6f}") 

    #Calculate sample concentrations

    results_df = calculate_results(
        df=df,corrected_df=corrected_df,label_data=label_data,
        A=A,B=B,logEC50=logEC50,D=D
    )

    #Print results

    print("\n==========================")
    print(" SAMPLE CONCENTRATION")
    print("==========================")

    if results_df.empty:
        print("No sample results were found.")
    else:
        display_df =results_df.drop(columns=[c for c in results_df.columns if c.startswith("_")])
        print(display_df.to_string(index=False))

    #QC table

    qc_summary_text, overall_status = generate_qc_summary(
        r2 = r2, results_df = results_df, cv_df = cv_df,
        std_low = STD_RECOVERY_LOW, std_high = STD_RECOVERY_HIGH, x=x,corrected_df = corrected_df
    )

    print("\n" + qc_summary_text)

    #Ask for a unique run identifier
    run_id = input(
        "Enter a unique run/batch ID for this analysis:\n> "
    ).strip()

    if not run_id:
        raise ValueError("A run ID is required to save results.")

    #Save results to CSV

    if args.output:
        output_filename = args.output
    else:
        output_filename = f"{run_id}_sample_concentration_results.csv" 

    save_results_to_csv(results_df, output_filename)
    #Plot

    plot_fit(
        x_fit, y_fit, A, B,logEC50,D,r2
    )

    #Programme entry point

if __name__ == "__main__":
    main()
        