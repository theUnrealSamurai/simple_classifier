import pandas as pd
import tempfile

def clean_file(file_input_path, file_output_path):
    file_name = file_input_path.split('/')[-1].split('.')[0]
    setting = file_name.split('_')[-1]

    
    # column_names = "Bonus_history,Coins_Inserted_Cumulative,Coins_Acquired_Cumulative,No_of_Rotation_(excluding_bonus),No_of_Rotation_(including_bonus),Rotations_Till_bonus,Junk1,No_of_Regular_Bonus,No_of_Big_Bonus,Junk2,Junk3,Junk4,Junk5,Junk6,Junk7"

    column_names = ["Bonus_history", "Coins_Inserted_Cumulative", "Coins_Acquired_Cumulative", "No_of_Rotation_(excluding_bonus)", "No_of_Rotation_(including_bonus)", "Rotations_Till_bonus", "Junk1", "No_of_Regular_Bonus", "No_of_Big_Bonus", "Junk2", "Junk3", "Junk4", "Junk5", "Junk6", "Junk7"]
    df = pd.read_csv(file_input_path, header=None)
    df.columns = column_names

    df.head()

    return

    with open(file_input_path, 'r+') as file:
        content = file.readlines()
        content[0] = column_names + '\n'
        file.seek(0)
        file.writelines(content)
        file.truncate()


    print(df.head())

    return

    df = df[['No_of_Rotation_(excluding_bonus)', 'Coins_Inserted_Cumulative', 'Coins_Acquired_Cumulative', 'Bonus_history', 'Rotations_Till_bonus', 'No_of_Regular_Bonus', 'No_of_Big_Bonus']]
    
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
    
    df.drop(columns=["Hit"], inplace=True)

    df["setting"] = setting

    # df = df[["Bonus_history", "Coins_Inserted", "Coins_Acquired_Cumulative", "Hit"]]

    df = df.drop(df.index[0])
    df = df.reset_index(drop=True)

    df.drop(columns=["Bonus_history", "Rotations_Till_bonus"], inplace=True)

    df.to_csv("/Users/mits-mac-001/Code/simple_classifier/calculations_and_inference/data/temporary2/" + file_name + ".csv", index=False)


clean_file("/Users/mits-mac-001/Code/simple_classifier/calculations_and_inference/data/raw/20240215_4.csv", "/Users/mits-mac-001/Code/simple_classifier/calculations_and_inference/data/temporary2/")