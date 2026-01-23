"""
Analyze the current intel data distribution.
"""
import pandas as pd
import os

# Load the data
data_file = 'data/all_intel_2020_2026.csv'

if os.path.exists(data_file):
    df = pd.read_csv(data_file)
    
    print("=" * 60)
    print("CURRENT INTEL DATA ANALYSIS")
    print("=" * 60)
    
    print(f"\nTotal Records: {len(df):,}")
    
    # Date range
    if 'Date' in df.columns:
        print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Check daily count
    if 'Date' in df.columns:
        daily = df.groupby('Date').size()
        print(f"\nDaily Intel Statistics:")
        print(f"  Average per day: {daily.mean():.1f}")
        print(f"  Min per day: {daily.min()}")
        print(f"  Max per day: {daily.max()}")
    
    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION (CURRENT)")
    print("=" * 60)
    
    # Label distribution
    if 'Label' in df.columns:
        label_counts = df['Label'].value_counts()
        total = len(df)
        
        print("\n  Label             Count      Percentage")
        print("  " + "-" * 45)
        for label in ['TRUE_SIGNAL', 'NOISE', 'DECEPTION']:
            count = (df['Label'] == label).sum()
            pct = 100 * count / total
            print(f"  {label:15s}  {count:10,}    {pct:5.1f}%")
    
    print("\n" + "=" * 60)
    print("REQUIRED DISTRIBUTION (PROBLEM STATEMENT)")
    print("=" * 60)
    print("\n  TRUE_SIGNAL:    50%")
    print("  NOISE:          40%")
    print("  DECEPTION:      10%")
    
    print("\n" + "=" * 60)
    print("GAP ANALYSIS")
    print("=" * 60)
    
    # Compare
    current = {
        'TRUE_SIGNAL': (df['Label'] == 'TRUE_SIGNAL').sum() / total * 100,
        'NOISE': (df['Label'] == 'NOISE').sum() / total * 100,
        'DECEPTION': (df['Label'] == 'DECEPTION').sum() / total * 100,
    }
    required = {'TRUE_SIGNAL': 50, 'NOISE': 40, 'DECEPTION': 10}
    
    print("\n  Label         Current    Required    Gap")
    print("  " + "-" * 45)
    for label in ['TRUE_SIGNAL', 'NOISE', 'DECEPTION']:
        gap = current[label] - required[label]
        print(f"  {label:12s}  {current[label]:6.1f}%    {required[label]:6.1f}%   {gap:+5.1f}%")
    
    # Check by year
    if 'Year' in df.columns:
        print("\n" + "=" * 60)
        print("DISTRIBUTION BY YEAR")
        print("=" * 60)
        
        for year in sorted(df['Year'].unique()):
            year_df = df[df['Year'] == year]
            year_total = len(year_df)
            true_pct = (year_df['Label'] == 'TRUE_SIGNAL').sum() / year_total * 100
            noise_pct = (year_df['Label'] == 'NOISE').sum() / year_total * 100
            dec_pct = (year_df['Label'] == 'DECEPTION').sum() / year_total * 100
            print(f"  {year}: {year_total:,} records | TRUE: {true_pct:.1f}% | NOISE: {noise_pct:.1f}% | DECEPTION: {dec_pct:.1f}%")

else:
    print(f"Data file not found: {data_file}")
