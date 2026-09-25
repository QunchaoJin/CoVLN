"""Aggregate per-episode prediction JSONs into one spreadsheet and print mean metrics.

Usage:
    python tools/aggregate_results.py --pred_dir datasets/exprs_map/<exp>/preds --output <exp>.xlsx
"""
import os
import json
import argparse

import pandas as pd


def aggregate_json_to_excel(folder_path, output_name="evaluation_results.xlsx"):
    data_list = []

    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    if not json_files:
        print(f"No JSON files found under {folder_path}.")
        return

    print(f"Processing {folder_path}: {len(json_files)} files...")

    for filename in json_files:
        file_path = os.path.join(folder_path, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)

                # 1. evaluation metrics, stored as [value] lists
                eval_data = content.get('evaluation', {})
                row = {k: v[0] if isinstance(v, list) and len(v) > 0 else v
                       for k, v in eval_data.items()}

                # 2. bookkeeping columns
                row['file_name'] = filename
                row['scan'] = content.get('scan', '')
                row['instr_id'] = content.get('instr_id', '')

                # 3. main vs aux agent, inferred from the file name
                row['agent_type'] = 'aux_agent' if 'aux' in filename else 'main_agent'

                data_list.append(row)
        except Exception as e:
            print(f"Failed to read {filename}: {e}")

    df = pd.DataFrame(data_list)

    for col in ['success', 'spl']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    cols = ['file_name', 'agent_type', 'instr_id', 'scan']
    existing_cols = [c for c in cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + other_cols]

    print("\n" + "=" * 30)
    print("Metrics:")

    target_metrics = ['success', 'spl', 'oracle_success', 'nav_error']
    available_metrics = [m for m in target_metrics if m in df.columns]

    if available_metrics:
        overall_means = df[available_metrics].mean()
        print("[Overall]")
        for metric in available_metrics:
            print(f"  {metric.upper()}: {overall_means[metric]:.4f}")

        if 'agent_type' in df.columns:
            print("\n[By agent type]")
            grouped_means = df.groupby('agent_type')[available_metrics].mean()
            print(grouped_means)
    else:
        print("No success/spl metrics found in the JSON files.")
    print("=" * 30 + "\n")

    df.to_excel(output_name, index=False)
    print(f"Saved to {output_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pred_dir', required=True, help='directory containing case_*.json prediction files')
    parser.add_argument('--output', default='evaluation_results.xlsx', help='output .xlsx path')
    args = parser.parse_args()
    aggregate_json_to_excel(args.pred_dir, args.output)
