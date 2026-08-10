import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


STANDARD_CONCENTRATIONS = [30 * (0.75 ** i) for i in range(16)]


def four_pl(logx, A, B, logEC50, D):
    return D + (A - D) / (1 + 10 ** ((logEC50 - logx) * B))


def simulate_elisa_data(seed=42):
    rng = np.random.default_rng(seed)

    STD = [30 * (0.75 ** i) for i in range(16)]

    A_true, B_true, C_true, D_true = 3.0, 1.3, 1.5, 0.03

    def noisy_duplicate(x_vals, A, B, C, D, noise=0.03):
        base = four_pl(np.array(x_vals), A, B, C, D)
        rep1 = base * (1 + rng.normal(0, noise, len(base)))
        rep2 = base * (1 + rng.normal(0, noise, len(base)))
        return rep1, rep2

    std1_r1, std1_r2 = noisy_duplicate(STD[0:8], A_true, B_true, C_true, D_true)
    std2_r1, std2_r2 = noisy_duplicate(STD[8:16], A_true, B_true, C_true, D_true)

    sample_names = ["A1", "A2", "A3"]
    timepoints = [
        "Assay 10x", "Assay 50x", "Assay 100x", "Assay 1000x",
        "Assay 5000x", "Assay 10000x", "TestSample 100x", "TestSample 1000x"
    ]

    sample_cols = {}
    for name in sample_names:
        base_curve = rng.uniform(0.05, 3.5, size=8)
        rep1 = base_curve * (1 + rng.normal(0, 0.05, 8))
        rep2 = base_curve * (1 + rng.normal(0, 0.05, 8))
        sample_cols[name] = (rep1, rep2)

    data = {
        "Unnamed: 2": std1_r1, "Unnamed: 3": std1_r2,
        "Unnamed: 4": std2_r1, "Unnamed: 5": std2_r2,
        "Unnamed: 6": sample_cols["A1"][0], "Unnamed: 7": sample_cols["A1"][1],
        "Unnamed: 8": sample_cols["A2"][0], "Unnamed: 9": sample_cols["A2"][1],
        "Unnamed: 10": sample_cols["A3"][0], "Unnamed: 11": sample_cols["A3"][1],
        "Unnamed: 12": rng.uniform(0.02, 0.03, 8),
        "Unnamed: 13": rng.uniform(0.02, 0.03, 8),
    }
    od_df = pd.DataFrame(data).round(3)

    label_rows = []
    for i in range(8):
        row = {
            "Unnamed: 2": STD[i], "Unnamed: 3": np.nan,
            "Unnamed: 4": STD[i + 8], "Unnamed: 5": np.nan,
            "Unnamed: 6": f"A1 {timepoints[i]}", "Unnamed: 7": np.nan,
            "Unnamed: 8": f"A2 {timepoints[i]}", "Unnamed: 9": np.nan,
            "Unnamed: 10": f"A3 {timepoints[i]}", "Unnamed: 11": np.nan,
            "Unnamed: 12": np.nan, "Unnamed: 13": np.nan,
        }
        label_rows.append(row)
    label_df = pd.DataFrame(label_rows)

    fake_df = pd.concat([od_df, label_df], ignore_index=True)
    return fake_df, STD


def preprocess_elisa_data(df):
    """Preprocess the generated or imported ELISA plate layout."""
    od_data = df.iloc[:8].copy()
    label_data = df.iloc[8:].copy()

    od_data = od_data.apply(pd.to_numeric, errors="coerce")

    blank_cols = od_data.columns[-2:]
    blank_avg = od_data[blank_cols].mean().mean()
    print("Average blank OD:", blank_avg)

    od_corrected = od_data.drop(columns=blank_cols).subtract(blank_avg)

    sample_cols = od_corrected.columns
    avg_results = {}

    for i in range(0, len(sample_cols), 2):
        col1 = sample_cols[i]
        col2 = sample_cols[i + 1]
        sample_name = f"Sample_{i // 2 + 1}"
        avg_results[sample_name] = od_corrected[[col1, col2]].mean(axis=1)

    corrected_df = pd.DataFrame(avg_results)
    return corrected_df, label_data, blank_avg


def fit_4pl(x, y):
    """Fit the standard-curve OD values to a 4PL model."""
    mask = x > 0
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 4:
        raise ValueError('At least four positive concentration points are required for fitting a 4PL curve.')

    logx = np.log10(x_clean)
    y_clean = y_clean.astype(float)

    p0 = [max(y_clean) * 2, 1.4, np.log10(np.median(x_clean)), min(y_clean)]
    bounds = ([0, 0.01, -1, -0.05], [50, 10, 4, 0.5])

    p, _ = curve_fit(
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

    A, B, logEC50, D = p
    y_pred = four_pl(logx, A, B, logEC50, D)
    r2 = 1 - np.sum((y_clean - y_pred) ** 2) / np.sum((y_clean - np.mean(y_clean)) ** 2)

    return A, B, logEC50, D, r2, x_clean, y_clean


def calculate_sample_concentrations(sample_ods, A, B, logEC50, D):
    """Invert the fitted 4PL model for sample OD values."""
    sample_ods = np.asarray(sample_ods, dtype=float)
    concentrations = []

    for y in sample_ods:
        if y <= D:
            concentrations.append(float('nan'))
        else:
            ratio = (A - D) / (y - D) - 1
            if ratio <= 0:
                concentrations.append(float('nan'))
            else:
                log_conc = logEC50 - np.log10(ratio) / B
                concentrations.append(10 ** log_conc)

    return concentrations


def plot_fit(x_fit, y_fit, A, B, logEC50, D, r2):
    x_smooth = np.logspace(np.log10(x_fit.min()), np.log10(x_fit.max()), 300)
    y_smooth = four_pl(np.log10(x_smooth), A, B, logEC50, D)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("none")
    ax.patch.set_color("none")
    ax.scatter(x_fit, y_fit, color='black', zorder=5, label='Data points', s=50)
    ax.plot(x_smooth, y_smooth, color='red', linewidth=2, label=f'4PL Fit (R²={r2:.4f})')
    ax.set_xscale('log')
    ax.set_xlabel('Logarithmic Conc')
    ax.set_ylabel('OD')
    ax.set_title(f'4PL Fit (R2={r2:.4f})')
    ax.legend()
    plt.tight_layout()
    plt.show()


def calculate_results(df, corrected_df, label_data, A, B, logEC50, D, C):
    """Back-calculate concentration values for all labeled samples."""
    columns = df.columns
    n_pairs = len(columns) // 2

    results = []

    for p in range(n_pairs):
        col1 = columns[2 * p]
        col2 = columns[2 * p + 1]
        label_val = label_data[col1].iloc[0]

        if pd.isna(label_val):
            continue

        try:
            float(label_val)
            continue
        except (ValueError, TypeError):
            pass

        sample_col_name = f"Sample_{p + 1}"
        y_avg = corrected_df[sample_col_name]

        for i in range(8):
            label = label_data[col1].iloc[i]
            if pd.isna(label):
                continue

            label_str = str(label).strip()
            dil_match = re.search(r'(\d+)\s*x\b', label_str)
            dilution = int(dil_match.group(1)) if dil_match else 1

            parts = label_str.split(maxsplit=1)
            sample_name = parts[0]
            remainder = parts[1] if len(parts) > 1 else ""
            if dil_match:
                remainder = remainder.replace(dil_match.group(0), "")
            sample_type = re.sub(r'\s+', ' ', remainder).strip()

            y = y_avg.iloc[i]

            if y <= D or y >= A:
                conc_final = np.nan
            else:
                conc = C * ((A - D) / (y - D) - 1) ** (1 / (-B))
                conc_final = conc * dilution

            results.append({
                "Sample": sample_name,
                "Sample Type": sample_type,
                "Dilution": dilution,
                "OD": y,
                "Concentration (ng/mL)": conc_final
            })

    return pd.DataFrame(results)


def save_results_to_csv(results_df, output_path='sample_concentration_results.csv'):
    """Write the final concentration table to a file instead of only printing it."""
    out_path = Path(output_path)
    results_df.to_csv(out_path, index=False)
    print(f'\nSaved concentration results to: {out_path.resolve()}')


def load_workbook_grid(workbook_path: str):
    """Read the workbook selected channel grid from the Results by plate sheet."""
    workbook = pd.ExcelFile(workbook_path)
    if 'Results by plate' not in workbook.sheet_names:
        raise ValueError('The workbook does not contain a Results by plate sheet.')

    plate_df = pd.read_excel(workbook_path, sheet_name='Results by plate', header=None)

    wanted_idx = None
    for idx, row in plate_df.iterrows():
        candidate = row.iloc[0] if len(row) > 0 else None
        if pd.notna(candidate) and str(candidate).strip().lower() == 'od(450) - od(570)':
            wanted_idx = idx
            break

    if wanted_idx is None:
        raise ValueError('The requested OD(450) - OD(570) channel grid was not found in the workbook.')

    # The selected channel matrix occupies rows idx+2 through idx+9 with columns 0..12.
    selected_df = plate_df.iloc[wanted_idx + 2: wanted_idx + 10, 0:13].copy()
    selected_df = selected_df.reset_index(drop=True)

    # The first row includes the A/B/C row label; the second row includes the column labels.
    # Rebuild a notebook-shaped flat frame after the original grid is detected.
    selected_df.columns = [f'Unnamed: {i}' for i in range(13)]
    return selected_df


def main():
    parser = argparse.ArgumentParser(description='Automated 4PL ELISA analysis from the notebook workflow.')
    parser.add_argument('excel_path', nargs='?', default=None, help='Optional Excel workbook path (.xlsx/.xls) that contains the Results by plate grid.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for synthetic ELISA data generation.')
    args = parser.parse_args()

    if args.excel_path:
        df = load_workbook_grid(args.excel_path)
        STD = STANDARD_CONCENTRATIONS
        label_data = pd.DataFrame()
    else:
        df, STD = simulate_elisa_data(seed=args.seed)

    corrected_df, label_data, blank_avg = preprocess_elisa_data(df)

    x = np.array(STD)
    y = np.concatenate([
        corrected_df["Sample_1"].iloc[0:8].to_numpy(),
        corrected_df["Sample_2"].iloc[0:8].to_numpy()
    ])

    A, B, logEC50, D, r2, x_fit, y_fit = fit_4pl(x, y)
    C = 10 ** logEC50

    print('Top Asymptote:', A)
    print('Slope:', B)
    print('Inflection point:', C)
    print('Bottom Asymptote:', D)
    print('Standard Curve Coefficient:', r2)

    results_df = calculate_results(df, corrected_df, label_data, A, B, logEC50, D, C)
    print('\nCalculated sample concentration results:')
    print(results_df.to_string(index=False))

    save_results_to_csv(results_df, output_path='sample_concentration_results.csv')

    plot_fit(x_fit, y_fit, A, B, logEC50, D, r2)


if __name__ == '__main__':
    main()
