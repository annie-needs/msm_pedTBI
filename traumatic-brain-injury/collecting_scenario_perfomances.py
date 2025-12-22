# Author: Annie Needs
# Date: December 2025
# Objective: make a csv to compare the best training loss iterations for each scenario to appropriately compare the different scenarios' model performance. 

import os 
import pandas as pd 

parent_dir = os.getcwd()
n_scenarios = 21

output_csv = 'scenario_comparison.csv'

rows = []

for i in range(1, n_scenarios+1):
    if i != 13: # skip scenario 13 bc it had an error & didn't run properly
        scenario_name = f'scenario{i}'
        print(scenario_name)
        scenario_dir = os.path.join(parent_dir,scenario_name)
        report_path = os.path.join(scenario_dir, 'report.csv')

        df = pd.read_csv(report_path)

        best_row = df.loc[df['Cost_train'].idxmin()].copy()
        best_row['Scenario'] = scenario_name
        
        rows.append(best_row)

if rows: 
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(os.path.join(parent_dir, output_csv), index=False)
    print(f'saved comparison file to {output_csv}')
else:
    print('no valid scenarios found')