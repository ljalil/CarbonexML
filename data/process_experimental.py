import pandas as pd
from sklearn.model_selection import train_test_split

T_crit = 304.13  # K
P_crit = 7.3773  # MPa

def run():
    exp = pd.read_excel('data/experimental_data/CO2_solubility_in_brines.ods', engine='odf')
    bad_authors = ['Spycher et al. (2003)', 'Liu et al. (2011)',  'Yan et al. (2011)', 'dos Santos et al. (2003)', 'Bermejo et al. (2005)', 'Kamps et al. (2007)']
    exp = exp[~exp['Authors'].isin(bad_authors)]
    # Compute distance from critical point
    exp['T_critical_distance'] = (exp['Temperature (K)'] - T_crit).round(2)
    exp['P_critical_distance'] = (exp['Pressure (MPa)'] - P_crit).round(2)

    ionic_composition = {
        "Na+": {"NaCl": 1, "Na2SO4": 2},
        "K+": {"KCl": 1},
        "Mg+2": {"MgCl2": 1},
        "Ca+2": {"CaCl2": 1},
        "SO4-2": {"Na2SO4": 1},
        "Cl-": {"NaCl": 1, "KCl": 1, "MgCl2": 2, "CaCl2": 2},
    }

    for ion in ionic_composition.keys():
        exp[ion] = 0

    for ion, salts_dict in ionic_composition.items():
        for salt, multiplier in salts_dict.items():
            if salt in exp.columns:
                exp[ion] += exp[salt] * multiplier

    exp = exp[['Temperature (K)','Pressure (MPa)','T_critical_distance', 'P_critical_distance', 'Na+','K+','Mg+2','Ca+2','SO4-2','Cl-','Dissolved CO2 (mol/kg)']]
    exp.to_csv('data/processed/experimental.csv', index=False)

def split_experimental_data(split=0.2):
    exp = pd.read_csv('data/processed/experimental.csv')
    x_train, x_test, y_train, y_test = train_test_split(
        exp.drop(columns=['Dissolved CO2 (mol/kg)']),
        exp['Dissolved CO2 (mol/kg)'],
        test_size=split,
        random_state=42
    )
    train_df = x_train.copy()
    train_df['Dissolved CO2 (mol/kg)'] = y_train
    test_df = x_test.copy()
    test_df['Dissolved CO2 (mol/kg)'] = y_test

    train_df.to_csv('data/processed/experimental_train.csv', index=False)
    test_df.to_csv('data/processed/experimental_test.csv', index=False)


run()
split_experimental_data()