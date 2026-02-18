# Author: Annie Needs
# Date: December 2025
# Objective: make a csv to compare the best training loss iterations for each scenario to appropriately compare the different scenarios' model performance. 

import os 
from pathlib import Path
import pandas as pd 

parent_dir = os.getcwd()

output_csv = 'scenario_comparisons_18Feb2026.csv'

rows = []

for i in range(1, 56):
    if i == 36:
        continue

    scenario_name = f'scenario{i}'
    print(scenario_name)
    scenario_dir = os.path.join(parent_dir,scenario_name)
    report_path = os.path.join(scenario_dir, 'report.csv')
    
    if Path(report_path).exists():
        
        df = pd.read_csv(report_path)

        if df.shape[0] > 1:
            best_row = df.loc[df['Cost_Val'].idxmin()].copy()
            best_row['Scenario'] = scenario_name
        
            rows.append(best_row)
        else:
            print(f"{scenario_name} did not run.")
            continue
    else:
        print(f"{scenario_name} has not been submitted.")
        continue


if rows: 
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(os.path.join(parent_dir, output_csv), index=False)
    print(f'saved comparison file to {output_csv}')
else:
    print('no valid scenarios found')