import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Settings

SHEET_NAME = "Results by plate"

WANTED_OD = "OD(450) - OD(570)"

#Standards

FIRST_STANDARD_CONCENTRATION = 30.0  # ng/mL
NUMBER_OF_STANDARDS = 16
DILUTION_FACTOR = 0.75

STANDARD_CONCENTRATIONS = [
    FIRST_STANDARD_CONCENTRATION * (DILUTION_FACTOR ** i) for i in range(NUMBER_OF_STANDARDS)
]

#4PL Model

def four_pl(logx, A, B, logEC50, D):
    """
    Four-parameter logistic (4PL) model function.
    A = top asymptote, 
    B = slope, 
    logEC50 = logarithm of the inflection point (EC50),
    D = bottom asymptote.
    """
    return D + (A - D) / (1 + 10 ** ((logEC50 - logx) * B))

# Load excel file

def load_workbook_grid(workbook_path):
    """
    Load the Excel workbook and find the OD(450) - OD(570) channel grid from the Results by plate sheet.
    Returns the dataframe beginning at the selected OD channel
    """
    workbook_path = Path(workbook_path)

    if not workbook_path.exists():
        raise FileNotFoundError(f"The specified workbook path does not exist: {workbook_path}")
    print(f"\nLoading Excel file: {workbook_path}")

    # Read worknook information
    workbook = pd.ExcelFile(workbook_path)

    # Try the expected sheet name first
    possible_sheets = [
        SHEET_NAME,
        "Results by plate",
        "Results",
        "Sheet1",
        "Results_by_plate"
    ]

    selected_sheet = None
    for sheet in possible_sheets:
        if sheet in workbook.sheet_names:
            selected_sheet = sheet
            break

    if selected_sheet is None:
        raise ValueError(
            f"\nCould not find the Results sheet.\n"
            f"\nAvailable sheets are:\n"
            f"{workbook.sheet_names}"
        )

    print(f"Using sheet: {selected_sheet}")

    #Read without assuming header

    df = pd.read_excel(
        workbook_path, 
        sheet_name=selected_sheet, 
        header=None
        )

    # Find OD(450) - OD(570)

    wanted_idx = None
    for idx, row in df.iterrows():
        if len(row) == 0:
            continue
        candidate = row.iloc[0]

        if pd.notna(candidate):
            candidate_str = str(candidate).strip().lower()

            if candidate_str == WANTED_OD.lower():
                wanted_idx = idx
                break

    if wanted_idx is None:
        raise ValueError(
            f"\nCould not find '{WANTED_OD} ' in the sheet '{selected_sheet}'.\n"
            f"\nPlease check the sheet name and the OD channel label.\n"
        )
    print(f"Found '{WANTED_OD}' at Excel row: {wanted_idx + 1}")

    new_df = df.iloc[wanted_idx:].reset_index(drop=True)

    new_df = new_df.iloc[2:, 2:].reset_index(drop=True)

    print("ELISA data extracted")
    return new_df

#Defining blanks samples

def find_blank_wells(label_data):
    "Scan label_data for blank samples. Returns"
    "list of blank well"

    blank_wells = []

    for col in label_data.columns:
        for row_idx in range(len(label_data)):
            label_val = label_data[col].iloc[row_idx]
            if pd.isna(label_val):
                continue
            if "blank" in str(label_val).strip().lower():
                blank_wells.append((row_idx, col))
    return blank_wells
    
#PREPROCESS ELISA DATA

def preprocess_elisa_data(df):
    """ Split OD and Label sections.
    Calculates:
    - Average blank OD
    - Blank-corrected OD
    - Duplicate average
    """

    print("\nPreprocessing ELISA data...")

    # First 8 rows = OD values
    od_data = df.iloc[:8].copy()

    #Remaining rows = sample labels
    label_data = df.iloc[8:].copy()

    #Convert OD values to numeric
    od_data = od_data.apply(
        pd.to_numeric, errors="coerce")

    #Blank calculation
    blank_wells = find_blank_wells(label_data)

    if not blank_wells:
        raise ValueError(
            "No wells labeled 'blank' were found in label_data."
            " Please check your blank wells are labeled correctly."
        )
    print(f"Found blank wells (row, column): {blank_wells}")

    #pull the OD values for each identified blank well
    blank_values = [od_data[col].iloc[row_idx] for row_idx, col in blank_wells]
    blank_avg = np.mean(blank_values)

    print(f"Average blank OD: {blank_avg:.6f}")

    #Blank-correction

    od_corrected = (od_data - blank_avg)

    #Exclude any column that is entirely blank wells from sample averaging loop
    #(a column only gets excluded if every row in it was flagged as blank)

    blank_col_counts = {}
    for _, col in blank_wells:
        blank_col_counts[col] = blank_col_counts.get(col,0)+ 1
    fully_blank_cols = [
        col for col, count in blank_col_counts.items()
        if count == len(od_data) #every row in this columnis a blank
    ]

    sample_cols = [c for c in od_corrected.columns if c not in fully_blank_cols]

    if len(sample_cols) % 2 != 0:
        raise ValueError(
            "The OD data does not contain an even number of "
            "columns for duplicate measurements (after excluding fully-blank columns)."
        )

    
    #Average duplicates

    avg_results = {}

    for i in range(0, len(sample_cols), 2):
        col1 = sample_cols[i]
        col2 = sample_cols[i + 1]
        sample_name = f"Sample_{i // 2 + 1}"
        avg_results[sample_name] = od_corrected[[col1, col2]].mean(axis=1)

    corrected_df = pd.DataFrame(avg_results)

    print("\nBlank-corrected duplicate averages:")
    print(corrected_df.to_string(index=False))
    return corrected_df, label_data, blank_avg

def plot_fit(x_fit, y_fit, A, B, logEC50, D, r2):
    """Display the fitted 4PL curve."""

    x_smooth = np.logspace(
        np.log10(x_fit.min()),
        np.log10(x_fit.max()),
        300
    )

    y_smooth = four_pl(
        np.log10(x_smooth),
        A,
        B,
        logEC50,
        D
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    fig.patch.set_facecolor('none')
    ax.patch.set_color('None')

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

    ax.set_xlabel('Log Concentration (ng/mL)', fontsize=12)
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

    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 4:
        raise ValueError('At least four positive concentration points are required for fitting a 4PL curve.')

    logx = np.log10(x_clean)

    #Initial guesses
    p0 = [
        max(y_clean) * 2,
        1.4,
        np.log10(np.median(x_clean)),
        min(y_clean)
    ]

    #parameter bounds
    bounds = (
        [0, 0.01, -1, -0.05],
        [50, 10, 4, 0.5] 
    )

    print("\nFitting 4PL calibration curve...")

    parameters, _ = curve_fit(
        four_pl,
        logx,
        y_clean,
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
    ss_res = np.sum(
        (y_clean - y_pred) ** 2
    )
    ss_tot = np.sum(
        (y_clean - np.mean(y_clean)) ** 2
    )
    if ss_tot == 0:
        r2 = float('nan')
    else:
        r2 = 1 - (ss_res / ss_tot)
    
    return (A, B, logEC50, D, r2, x_clean, y_clean)

#Calculate sample concentrations

def calculate_results(
        df,
        corrected_df,
        label_data,
        A, 
        B, 
        logEC50, 
        D):

    """Back-calculate concentration for all labeled samples.

    The calculated concentration is multiplied by the
    sample-specific dilution factor.
    """

    results = []
    columns = df.columns

    #Number of duplicate pairs
    n_pairs = len(columns) // 2

    #loop through each pair
    for p in range(n_pairs):
        col1 = columns[2 * p]
        col2 = columns[2 * p + 1]
        sample_col_name = f"Sample_{p + 1}"
        #make sure the corresponding averaged OD exists

        if sample_col_name not in corrected_df.columns:
            continue

        y_avg = corrected_df[sample_col_name]

        #Each pair has 8 rows of samples
        for i in range(8):
            if i >= len(label_data):
                continue
            label = label_data[col1].iloc[i]
            if pd.isna(label):
                continue

            label_str = str(label).strip()

            if not label_str:
                continue

            #Extract dilution (Example 10x, 100x, etc.)

            dil_match = re.search(r'(\d+)\s*x\b', label_str, flags=re.IGNORECASE)

            if dil_match:
                dilution = int(dil_match.group(1))
            else:
                dilution = 1

            #Extract sample name and type
            parts = label_str.split(maxsplit=1)

            sample_name = parts[0]

            remainder = parts[1] if len(parts) > 1 else ""

            if dil_match:
                remainder = remainder.replace(dil_match.group(0), "")

            sample_type = re.sub(r'\s+', ' ', remainder).strip()

            #OD
            y = y_avg.iloc[i]

            #Back-calculate concentration
            #y = D + (A - D) / (1 + 10 ** ((logEC50 - logx) * B))
            #conc = C * ((A - D) / (y - D) - 1) ** (1 / (-B))

            if (pd.isna(y) or y <= D or y >= A):
                concentration = np.nan
            else:
                ratio = (A - D) / (y - D) - 1
                if ratio <= 0:
                    concentration = np.nan
                else:
                    C = 10 ** logEC50
                    concentration = C * (ratio ** (1 / (-B)))
                    # Multiply by dilution factor
            results.append({
                "Sample": sample_name,
                "Sample Type": sample_type,
                "Dilution": dilution,
                "OD": y,
                "Concentration (ng/mL)": concentration * dilution,
                "Concentration (µg/mL)": (
                    concentration /1000
                    if pd.notna(concentration) 
                    else np.nan
                )
            })
    return pd.DataFrame(results)
        

#Save results to CSV

def save_results_to_csv(results_df, output_path='sample_concentration_results.csv'):
    """Save calculated concentration to CSV"""
    output_path = Path(output_path)

    results_df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path.resolve()}")

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
        '--output',
        default='sample_concentration_results.csv',
        help=(
            "Output CSV filename "
            "(default: sample_concentration_results.csv)"
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
    #load excel data

    df = load_workbook_grid(excel_path)

    #preprocess

    corrected_df, label_data, blank_avg = preprocess_elisa_data(df)

    #prepare standard curve 
    #first 8 std = sample_1, next 8 std = sample_2
    
    x = np.array(STANDARD_CONCENTRATIONS, dtype=float)

    y = np.concatenate([
        corrected_df["Sample_1"].iloc[0:8].to_numpy(),
        corrected_df["Sample_2"].iloc[0:8].to_numpy()
    ])

    #Fit 4PL

    (
        A,
        B,
        logEC50,
        D,
        r2,
        x_fit,
        y_fit
    ) = fit_4pl(x, y)

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
        df=df,
        corrected_df=corrected_df,
        label_data=label_data,
        A=A,
        B=B,
        logEC50=logEC50,
        D=D
    )

    #Print results

    print("\n==========================")
    print(" SAMPLE CONCENTRATION")
    print("==========================")

    if results_df.empty:
        print("No sample results were found.")
    else:
        print(results_df.to_string(index=False))

    #Ask for a unique run identifier
    run_id = input(
        "Enter a unique run/batch ID for this analysis:\n> "
    ).strip()

    if not run_id:
        raise ValueError("A run ID is required to save results.")

    #Save results to CSV

    output_filename = f"{run_id}_sample_concentration_results.csv" 
    save_results_to_csv(results_df, output_filename)

    #Plot

    plot_fit(x_fit, 
            y_fit, 
            A,
            B,
            logEC50,
            D,
            r2
    )

    #Programme entry point

if __name__ == "__main__":
    main()
        