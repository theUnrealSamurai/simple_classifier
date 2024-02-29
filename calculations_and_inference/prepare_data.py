import os
import pandas as pd
from glob import glob
from tqdm import tqdm


pd.set_option('display.max_columns', None)


def rename_columns(files_list, new_first_line, output_folder):
    # in the raw csv files, the first row doesn't have column names instead has just one word that's v1.2
    # while reading that in pandas, that's being read as column names and causes problems further fixing that in this function
 
    os.makedirs(output_folder, exist_ok=True)
    
    for file_path in files_list:
        file_name = os.path.basename(file_path)
        output_file_path = os.path.join(output_folder, file_name)
        
        with open(file_path, 'r') as file:
            content = file.readlines()
        content[0] = new_first_line + '\n'
        
        with open(output_file_path, 'w') as output_file:
            output_file.writelines(content)


csv_files = glob("/Users/mits-mac-001/Code/simple_classifier/calculations_and_inference/data/set2/*.csv")
output_path = "/Users/mits-mac-001/Code/simple_classifier/calculations_and_inference/data/temporary"

files = glob(os.path.join(output_path, "*"))
for f in files:
    os.remove(f)

column_names = "Bonus_history,Coins_Inserted_Cumulative,Coins_Acquired_Cumulative,No_of_Rotation_(excluding_bonus),No_of_Rotation_(including_bonus),Rotations_Till_bonus,Junk1,No_of_Regular_Bonus,No_of_Big_Bonus,Junk2,Junk3,Junk4,Junk5,Junk6,Junk7"

rename_columns(csv_files, column_names, output_folder=output_path)

csv_files = glob(os.path.join(output_path, "*.csv"))

print("\n\n")


def prepare_file(file_path):
    file_name = file_path.split('/')[-1].split('.')[0]
    setting = file_name.split('_')[-1]

    # getting the file name to find the setting of the machine. The setting of the machine is written in the file after the underscore

    df = pd.read_csv(file_path)

    df = df[['No_of_Rotation_(excluding_bonus)', 'Coins_Inserted_Cumulative', 'Coins_Acquired_Cumulative', 'Bonus_history', 'Rotations_Till_bonus', 'No_of_Regular_Bonus', 'No_of_Big_Bonus']]
    
    df["Profit_Loss"] = df["Coins_Acquired_Cumulative"] - df["Coins_Inserted_Cumulative"]

    # df['No_of_Rotation_(including_bonus)'] = df['No_of_Rotation_(including_bonus)'].shift(1).fillna(0)  # Replace NaN with 0
    df['No_of_Rotation_(excluding_bonus)'] = df['No_of_Rotation_(excluding_bonus)'].shift(1).fillna(0)  # Replace NaN with 0

    df['Bonus_history'] = df['Bonus_history'].fillna(0)

    indices_to_remove = []
    for i in range(len(df) - 1):
        if df.iloc[i]['No_of_Rotation_(excluding_bonus)'] == df.iloc[i + 1]['No_of_Rotation_(excluding_bonus)']:
            indices_to_remove.append(i + 1)

    # Remove rows where the next row has the same value
    df = df.drop(indices_to_remove)

    # Reset index after dropping rows
    df = df.reset_index(drop=True)

    df['Hit'] = [0] + list(df['Coins_Acquired_Cumulative'].shift(-2) - df['Coins_Acquired_Cumulative'].shift(-1))[:-1]
    df['Hit'].fillna(0, inplace=True)

    df['No_of_grape_till_now'] = (df['Hit'] == 8).cumsum()
    df['No_of_rhino_till_now'] = (df['Hit'] == 3).cumsum()
    df['No_of_cherry_corner_till_now'] = (df['Hit'] == 2).cumsum()
    df['No_of_clown_till_now'] = (df['Hit'] == 10).cumsum()
    df['No_of_cherry_middle_till_now'] = (df['Hit'] == 1).cumsum()
    
    df["RB_probaility"] = df['No_of_Regular_Bonus'] / df["No_of_Rotation_(excluding_bonus)"]
    df["BB_probaility"] = df['No_of_Big_Bonus'] / df["No_of_Rotation_(excluding_bonus)"]

    df["grape_probaility"] = df['No_of_grape_till_now'] / df["No_of_Rotation_(excluding_bonus)"]
    df["rhino_probaility"] = df['No_of_rhino_till_now'] / df["No_of_Rotation_(excluding_bonus)"]
    df["clown_probaility"] = df['No_of_clown_till_now'] / df["No_of_Rotation_(excluding_bonus)"]
    df["cherry_probaility"] = (df['No_of_cherry_corner_till_now'] + df["No_of_cherry_middle_till_now"] )/ df["No_of_Rotation_(excluding_bonus)"]
    

    df.drop(columns=["Hit"], inplace=True)

    df["setting"] = setting

    # df = df[["Bonus_history", "Coins_Inserted", "Coins_Acquired_Cumulative", "Hit"]]

    df = df.drop(df.index[0])
    df = df.reset_index(drop=True)

    df.drop(columns=["Bonus_history", "Rotations_Till_bonus"], inplace=True)

    df.to_csv("/Users/mits-mac-001/Code/simple_classifier/calculations_and_inference/data/temporary/" + file_name + ".csv", index=False)

for file in tqdm(csv_files):
    prepare_file(file)

csv_files = glob("/Users/mits-mac-001/Code/simple_classifier/calculations_and_inference/data/temporary/*.csv")

dfs = []

# Read each CSV file into a DataFrame and append to the list
for file_path in csv_files:
    df = pd.read_csv(file_path)
    print(len(df))
    dfs.append(df)

merged_df = pd.concat(dfs, ignore_index=True)


print(len(merged_df))
print(merged_df.tail())


merged_df.to_csv("/Users/mits-mac-001/Code/simple_classifier/calculations_and_inference/data/processed_data/set2.csv", index=False)


