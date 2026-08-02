#!/usr/bin/env python3
"""
Fix script to recompute objective summary files (CSV/JSON) from individual result files.
Useful when summaries were overwritten or incomplete due to sequential runs with filtering.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.append(os.getcwd())

from src.vibe_testing.reporting.passk import PassKReporter
from src.vibe_testing.utils import load_json

def recompute_summaries(
    results_dir: str, 
    prompt_type: Optional[str] = None,
    filename_prefix: str = "function_eval"
):
    path = Path(results_dir)
    if not path.is_dir():
        print(f"Error: {results_dir} is not a directory.")
        return

    # Pattern for individual objective result files
    # function_eval_objective_sample-<id>_v00.json
    pattern = f"{filename_prefix}_objective_sample-*.json"
    result_files = sorted(path.glob(pattern))
    
    if not result_files:
        print(f"No result files matching {pattern} found in {results_dir}")
        return

    print(f"Found {len(result_files)} individual result files.")

    records = []
    model_name = None
    model_config_path = None
    ks = set()
    run_name = "recomputed"

    for fpath in result_files:
        try:
            data = load_json(str(fpath))
            metrics = data.get("metrics")
            if not metrics:
                continue

            sample_id = metrics.get("sample_id", "")
            
            # Simple heuristic for prompt type filtering if requested
            is_variation = "::variation::" in sample_id
            if prompt_type == "personalized" and not is_variation:
                continue
            if prompt_type == "original" and is_variation:
                continue
            
            records.append(metrics)
            
            # Extract metadata from the first valid file
            if model_name is None:
                model_name = data.get("model_name")
            if model_config_path is None:
                model_config_path = data.get("model_config_path")
            
            # Collect all k values seen
            if "base" in metrics and "pass_at_k" in metrics["base"]:
                for k in metrics["base"]["pass_at_k"].keys():
                    ks.add(int(k))
        except Exception as e:
            print(f"Warning: Failed to process {fpath}: {e}")

    if not records:
        print("No records matched the criteria.")
        return

    print(f"Processing {len(records)} records...")
    
    if not ks:
        ks = {1}
    
    # Reconstruct reporter
    reporter = PassKReporter(
        ks=list(ks), 
        run_name=run_name, 
        model_name=model_name, 
        model_config_path=model_config_path
    )
    reporter._records = records
    
    # Export new summary files
    # Note: PassKReporter.export now supports prompt_type argument (after our previous edit)
    output_info = reporter.export(str(path), prompt_type=prompt_type)
    
    print("\nSuccess!")
    print(f"JSON summary: {output_info['json']}")
    print(f"CSV summary:  {output_info['csv']}")
    print(f"Aggregated Pass@1 (Base): {output_info['summary']['base'].get('1', 'N/A')}")
    print(f"Aggregated Pass@1 (Plus): {output_info['summary']['plus'].get('1', 'N/A')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recompute objective summaries.")
    parser.add_argument("results_dir", help="Directory containing individual JSON result files.")
    parser.add_argument("--prompt-type", choices=["original", "personalized"], help="Filter to specific prompt type.")
    parser.add_argument("--prefix", default="function_eval", help="Filename prefix (default: function_eval).")
    
    args = parser.parse_args()
    recompute_summaries(args.results_dir, args.prompt_type, args.prefix)

