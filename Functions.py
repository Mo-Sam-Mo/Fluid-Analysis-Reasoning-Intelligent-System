import numpy as np


FINAL_FEATURE_NAMES = [
    'Cu', 'Fe', 'Cr', 'Al', 'Si', 'Pb', 'Sn', 'Ni', 'Na', 'B', 'P', 'Zn',
    'Mo', 'Ca', 'Mg', 'TBN', 'V100', 'V40', 'OXI', 'TAN',
    'delta_visc_40', 'metal_sum', 'iron_to_copper_ratio',
    'water_flag', 'antifreeze_flag'
]

VISCOSITY_SPEC_40 = 150
METAL_COLS = ['Cu', 'Fe', 'Cr', 'Al', 'Si', 'Pb', 'Sn', 'Ni', 'Na', 'B', 'P', 'Zn', 'Mo', 'Ca', 'Mg']

def enrich_and_reorder_features(X):
    # Extract values
    cu = X.at[X.index[0], 'Cu']
    fe = X.at[X.index[0], 'Fe']
    v40 = X.at[X.index[0], 'V40']

    # Compute additional features
    delta_visc_40 = v40 - VISCOSITY_SPEC_40
    fe_cu_ratio = fe / cu if cu != 0 else 0
    metal_sum = X.loc[:, METAL_COLS].sum(axis=1).iloc[0]

    # Add new features
    X['delta_visc_40'] = delta_visc_40
    X['metal_sum'] = metal_sum
    X['iron_to_copper_ratio'] = fe_cu_ratio

    return X[FINAL_FEATURE_NAMES]


def sample_conversion(X):
    sample = '['
    for i, feat in X.items():
        if i == 'label':
            continue
        sample += f'{i}= {feat}, '
    sample = sample[:-1] + ']'
    return sample


def organize_report(raw_text):
    sections = {
        "The Observation": "",
        "Investigation Points": "",
        "The Solution": "",
        "Story": ""
    }
    
    current_section = None

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("**") and line.endswith("**"):
            current_section = line.strip("*").strip()
        elif current_section:
            sections[current_section] += line + " "

    # Clean up whitespace
    for key in sections:
        sections[key] = sections[key].strip()

    # Prepare main report (excluding story)
    main_report = ""
    for key in ["The Observation", "Investigation Points", "The Solution"]:
        main_report += f"**{key}**\n{sections[key]}\n\n"

    story = sections["Story"]
    return main_report.strip(), story