"""
Task 2 Execution Script: Canonical Split Integration, Retraining, and Prediction
Robustly resolves file paths, handles column name variations, maps ID columns, and ensures reproducibility.
"""

import pickle
from pathlib import Path
import pandas as pd
from classifier import TicketRoutingClassifier

def find_file(filename_candidates, search_dirs):
    """Helper to find a file across multiple possible names and directories."""
    for d in search_dirs:
        for name in filename_candidates:
            p = d / name
            if p.exists():
                return p
    raise FileNotFoundError(f"Could not find any of {filename_candidates} in search directories: {search_dirs}")

def main():
    print("Loading source data and canonical split manifest...")
    
    # Use pathlib to locate paths relative to this script's location automatically
    script_dir = Path(__file__).resolve().parent  # points to Task2/
    root_dir = script_dir.parent                  # points to AI_Eng_Assignment_3/
    
    # Search directories for data files
    search_paths = [root_dir, script_dir, root_dir.parent, root_dir / "data"]
    
    # 1. Locate and load source dataset (accounts for naming variations)
    ticket_candidates = ["support_tickets.csv", "IT Support Ticket Data.csv", "cleaned_support_tickets.csv"]
    ticket_path = find_file(ticket_candidates, search_paths)
    print(f"Found ticket dataset at: {ticket_path}")
    source_df = pd.read_csv(ticket_path)
    
    # 2. Locate and load split manifest
    manifest_candidates = ["split_manifest.csv"]
    manifest_path = find_file(manifest_candidates, search_paths)
    print(f"Found split manifest at: {manifest_path}")
    split_manifest = pd.read_csv(manifest_path)
    
    # Standardize column names to lowercase and strip whitespace
    source_df.columns = [str(c).strip().lower() for c in source_df.columns]
    split_manifest.columns = [str(c).strip().lower() for c in split_manifest.columns]
    
    print(f"Source columns found: {list(source_df.columns)}")
    print(f"Manifest columns found: {list(split_manifest.columns)}")
    
    # Map common column name variations for source dataset
    source_col_map = {}
    for col in source_df.columns:
        if 'ticket' in col and 'id' in col:
            source_col_map[col] = 'ticket_id'
        elif col in ['id', 'ticketid'] and 'ticket_id' not in source_col_map.values():
            source_col_map[col] = 'ticket_id'
        elif col in ['body', 'text', 'content']:
            source_col_map[col] = 'body'
        elif col in ['department', 'label', 'category']:
            source_col_map[col] = 'department'
    source_df = source_df.rename(columns=source_col_map)
    
    # Map common column name variations for split manifest
    manifest_col_map = {}
    for col in split_manifest.columns:
        if 'ticket' in col and 'id' in col:
            manifest_col_map[col] = 'ticket_id'
        elif col in ['id', 'ticketid'] and 'ticket_id' not in manifest_col_map.values():
            manifest_col_map[col] = 'ticket_id'
        elif 'split' in col:
            manifest_col_map[col] = 'split'
    split_manifest = split_manifest.rename(columns=manifest_col_map)
    
    # Fallback if 'ticket_id' is still missing from either dataframe
    for name, df in [("source_df", source_df), ("split_manifest", split_manifest)]:
        if 'ticket_id' not in df.columns:
            print(f"Warning: 'ticket_id' column not found in {name}. Checking fallback index columns...")
            for candidate in ['unnamed: 0', 'index', 'id']:
                if candidate in df.columns:
                    df.rename(columns={candidate: 'ticket_id'}, inplace=True)
                    break
            else:
                df['ticket_id'] = df.index.astype(str)

    # Ensure data types match for merging (convert to string and strip whitespace)
    source_df['ticket_id'] = source_df['ticket_id'].astype(str).str.strip()
    split_manifest['ticket_id'] = split_manifest['ticket_id'].astype(str).str.strip()

    # 3. Resolve ticket_ids against the immutable source dataset
    full_df = pd.merge(source_df, split_manifest, on="ticket_id", how="inner")
    
    # 4. Separate into training and validation splits (Do NOT use test split)
    train_df = full_df[full_df['split'] == 'train'].copy()
    val_df = full_df[full_df['split'] == 'validation'].copy()
    
    print(f"Training Samples: {len(train_df)} | Validation Samples: {len(val_df)}")
    
    # 5. Instantiate and fit classifier on canonical training split
    print("Retraining classifier on canonical training split...")
    clf = TicketRoutingClassifier(random_state=42)
    clf.fit(raw_texts=train_df['body'], labels=train_df['department'])
    
    # 6. Save the retrained model artifact inside Task2/
    model_path = script_dir / "retrained_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model successfully saved to: {model_path}")
    
    # 7. Predict on the validation set
    print("Generating predictions for validation set...")
    val_preds = clf.predict(val_df['body'])
    
    # 8. Construct the required 4-column output CSV
    output_df = pd.DataFrame({
        'ticket_id': val_df['ticket_id'].values,
        'true_label': val_df['department'].values,
        'recommended_department': val_preds['recommended_department'].values,
        'class_scores': val_preds['class_scores'].values
    })
    
    output_path = script_dir / "validation_predictions.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Validation predictions successfully saved to: {output_path}")

if __name__ == "__main__":
    main()