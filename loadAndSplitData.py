import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path

# Gets the absolute path to the file currently running
script_dir = Path(__file__).resolve().parent

"""
All structureddata loaded here from: https://huggingface.co/collections/marcelbinz/psych-101-csvs
Maggies Farm loaded from: https://github.com/marcelbinz/Llama-3.1-Centaur-70B/blob/f8038fb57e052614a1d0aae3e58bf5cca93da883/generalization/dubois2022value
"""
# load data
TwoBanditExp1 = pd.read_parquet("hf://datasets/marcelbinz/gershman2018deconstructing/exp1/train-00000-of-00001.parquet")
TwoBanditExp2 = pd.read_parquet("hf://datasets/marcelbinz/gershman2018deconstructing/exp2/train-00000-of-00001.parquet")
TwoChanging = pd.read_parquet("hf://datasets/marcelbinz/xiong2023neural/exp1/train-00000-of-00001.parquet")
FourDrifting = pd.read_parquet("hf://datasets/marcelbinz/bahrami2020four/exp/train-00000-of-00001.parquet")
HorizonSade = pd.read_parquet("hf://datasets/marcelbinz/sadeghiyeh2020temporal/exp1/train-00000-of-00001.parquet")
HorizonSomer = pd.read_parquet("hf://datasets/marcelbinz/somerville2017charting/exp1/train-00000-of-00001.parquet")
HorizonFeng = pd.read_parquet("hf://datasets/marcelbinz/feng2021dynamics/exp1/train-00000-of-00001.parquet")
HorizonWaltz = pd.read_parquet("hf://datasets/marcelbinz/waltz2020differential/exp1/train-00000-of-00001.parquet")

# maggies farm is saved elsewhere
MaggiesFarm_struc_path = "https://github.com/marcelbinz/Llama-3.1-Centaur-70B/raw/main/generalization/dubois2022value/exp1.csv"
MaggiesFarm_struc = pd.read_csv(MaggiesFarm_struc_path)

# load text data
Psych101 = pd.DataFrame(load_dataset('marcelbinz/Psych-101')['train'])
Psych101test = pd.DataFrame(load_dataset('marcelbinz/Psych-101-test') ['test'])

MaggiesFarm_path = "https://github.com/marcelbinz/Llama-3.1-Centaur-70B/raw/main/generalization/dubois2022value/prompts.jsonl"
MaggiesFarm = pd.read_json(MaggiesFarm_path, lines=True)

# Include 101Name mapping in the master registry to simplify text retrieval
# 'dir' is the data directory of each experiment, as read by OpenEvolve/evaluator.py, LLMFineTuning and LLMInference
experiments = pd.DataFrame([
    {'name': 'TwoBandit',      'dir': 'TB',  'experiment': 'exp1', 'split': 'Train', '101Name': 'gershman2018deconstructing/exp1.csv', 'struc_data': TwoBanditExp1},
    {'name': 'TwoBandit',      'dir': 'TB',  'experiment': 'exp2', 'split': 'Train', '101Name': 'gershman2018deconstructing/exp2.csv', 'struc_data': TwoBanditExp2},
    {'name': 'HorizonSomer',   'dir': 'HSo', 'experiment': 'exp0', 'split': 'Train', '101Name': 'somerville2017charting/exp1.csv',      'struc_data': HorizonSomer},
    {'name': 'HorizonWaltz',   'dir': 'HW',  'experiment': 'exp0', 'split': 'Train', '101Name': 'waltz2020differential/exp1.csv',       'struc_data': HorizonWaltz},
    {'name': 'DriftingBandit', 'dir': 'DB',  'experiment': 'exp0', 'split': 'Train', '101Name': 'bahrami2020four/exp.csv',              'struc_data': FourDrifting},
    {'name': 'ChangingBandit', 'dir': 'CB',  'experiment': 'exp0', 'split': 'OOD',   '101Name': 'xiong2023neural/exp1.csv',             'struc_data': TwoChanging},
    {'name': 'HorizonSade',    'dir': 'HSa', 'experiment': 'exp0', 'split': 'OOD',   '101Name': 'sadeghiyeh2020temporal/exp1.csv',      'struc_data': HorizonSade},
    {'name': 'HorizonFeng',    'dir': 'HF',  'experiment': 'exp0', 'split': 'OOD',   '101Name': 'feng2021dynamics/exp1.csv',            'struc_data': HorizonFeng},
    {'name': 'MaggiesFarm',    'dir': 'MF',  'experiment': 'exp0', 'split': 'OOD',   '101Name': None,                                   'struc_data': MaggiesFarm_struc},
])

# Match and map text datasets onto the registry
experiments['text_data'] = None
for idx, row in experiments.iterrows():
    if row['name'] == 'MaggiesFarm':
        experiments.at[idx, 'text_data'] = MaggiesFarm.copy()
    else:
        train_df = Psych101[Psych101["experiment"] == row['101Name']]
        test_df = Psych101test[Psych101test["experiment"] == row['101Name']]
        experiments.at[idx, 'text_data'] = pd.concat([train_df, test_df])

# target columns in exact order
TARGET_COLS = [
    'participant', 'game', 'horizon', 'trial', 'forced', 'human_choice', 'reward', 'hazard_rate'
]

exps_with_nans = []
split_results = {}

# loop over all data
for _, exp in experiments.iterrows():

    name = exp['name']
    exp_id = f"{name}_{exp['experiment']}"
    
    # get data
    text = exp['text_data'].copy()
    raw_struc = exp['struc_data'].copy()


    # Clean the nested structural data frame
    if 'RT' in raw_struc.columns:
        raw_struc = raw_struc.drop(['RT'], axis=1)
        
    struc = raw_struc.dropna().copy()
    
    exps_with_nans.append({
        "name": name,
        "removed_rows": len(raw_struc) - len(struc),
        "total_rows": len(struc),
        "nan_fraction": ((len(raw_struc) - len(struc)) / len(raw_struc)) if len(raw_struc) > 0 else 0
    })

    """
    All experiments contain the data columns:
    participant:    int32   - The participant ID
    game:           int32   - The number of the game, starts at 1
    horizon:        int32   - The number of trials per game, -1 if not known by the participant
    trial:          int32   - The trial number, resets every game and starts at 1
    forced:         int32   - Either 0 or 1, with 1 representing a non-trial, which won't be taken into account during computation of nll
    human_choice:   int32   - The human choice that has to be modelled
    reward:         int32   - The reward as a consequence of the human_choice
    hazard_rate:    int32   - Indicates the degree of abrupt expected point change, it ranges from 0-10, with 1 representing a 10% change
    """

    # fix trial index
    struc['trial'] = struc.groupby(['participant', 'task']).cumcount()
    
    # 2. Create clean dataframe with defaults
    clean_struc = pd.DataFrame({
        'participant':  struc['participant'].astype('int32'),
        'game':         (1 + struc['task']).astype('int32'),
        'horizon':      struc['horizon'].astype('int32') if name in ['HorizonSade', 'HorizonSomer', 'HorizonFeng', 'HorizonWaltz'] else -1,
        'trial':        (1 + struc['trial']).astype('int32'),
        'forced':       struc['forced'].astype('int32') if 'forced' in struc.columns else 0,
        'human_choice': struc['choice'].astype('int32') if 'choice' in struc.columns else 0,
        'reward':       struc['reward'].astype('int32') if 'reward' in struc.columns else 0,
        'hazard_rate':  (struc['hazard_rate'] * 10).round().astype('int32') if name == 'ChangingBandit' else 0,
    })

    # Dynamically resolve max horizon lengths for contextual tasks
    if name not in ['HorizonSade', 'HorizonSomer', 'HorizonFeng', 'HorizonWaltz', 'DriftingBandit']:
        clean_struc['horizon'] = clean_struc.groupby(['participant', 'game'])['trial'].transform('max')

    # retain only relevant columns in the exact target order
    clean_struc = clean_struc[TARGET_COLS]

    # Split allocations based on shuffled independent participants
    participants = clean_struc['participant'].unique()
    np.random.RandomState(42).shuffle(participants)
    n = len(participants)
    
    if exp['split'] == 'Train':
        # get the cut points for train, val, and test splits
        c1, c2 = (int(0.6 * n), int(0.7 * n)) if name == 'DriftingBandit' else (int(0.7 * n), int(0.8 * n))
        
        splits = {
            'Train': participants[:c1],
            'Val':   participants[c1:c2],
            'Test':  participants[c2:]
        }
    elif exp['split'] == 'OOD':
        c1 = int(0.8 * n)
        splits = {
            'Train': participants[:c1],
            'Test':  participants[c1:]
        }
    else:
        splits = None
        raise ValueError(f"Unknown split type: {exp['split']} for experiment {name}")
       
    # create data folders for each experiment 
    Path(f"{script_dir}/Data/{exp['dir']}").mkdir(parents=True, exist_ok=True)

    # save the split results for reporting
    for split_label, split_participants in splits.items():
        # Mask matching structural matrices and text segments
        sub_data = clean_struc[clean_struc['participant'].isin(split_participants)]
        sub_text = text[text['participant'].isin(split_participants)]
        
        # Export natural language prompts to CSV
        sub_text.to_csv(f"{script_dir}/Data/{exp['dir']}/{split_label}_text_{exp['experiment']}.csv", index=False)

        # Export clean tensor states to NumPy, as integers so that choices can be used as indices
        np.save(f"{script_dir}/Data/{exp['dir']}/struc_{split_label}_{exp['experiment']}.npy", sub_data.to_numpy(dtype=np.int64))


# --- Summary Report ---
if exps_with_nans:
    print("\n Experiments containing NaNs:")
    for entry in exps_with_nans:
        print(f"- {entry['name']}: {entry['removed_rows']} rows removed")
else:
    print("\n No NaNs found in any experiments.")

# Result:
"""
 Experiments containing NaNs:
- TwoBandit: 0 rows removed
- TwoBandit: 0 rows removed
- ChangingBandit: 0 rows removed
- DriftingBandit: 4934 rows removed
- HorizonSade: 0 rows removed
- HorizonSomer: 0 rows removed
- HorizonFeng: 0 rows removed
- HorizonWaltz: 0 rows removed
- MaggiesFarm: 0 rows removed
"""
