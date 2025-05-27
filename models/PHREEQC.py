import os
import yaml
import subprocess
import numpy as np
import pandas as pd

#cfg = yaml.safe_load(open("configs/data_params.yaml"))
database_folder = '/usr/local/share/doc/phreeqc/database'
temp_files_path = '.'

experimental_test = '/home/jalil/OneDrive/Papers/CCUS/CarbonexML/data/processed/experimental_test.csv'

def simulate_single_state(
    temperature, pressure, ion_moles, database = 'phreeqc', pqi_file_path="."
):

    
    database_path = os.path.join(database_folder, f'{database}.dat')

    pressure_atm = pressure * 9.86923
    temperature_c = round(temperature - 273.15, 2)
    p_co2 = pressure_atm * 0.95
    p_h2o = pressure_atm * 0.05


    phreeqc_code = f"DATABASE {database_path}\n"
    phreeqc_code += f"SOLUTION 1\n\ttemp\t{temperature_c}\n\tpH\t7.0\n\t"
    phreeqc_code += f"units\tmol/kgw\n"

    # Get ion concentrations from the ion_moles dictionary
    phreeqc_code += f'\tNa\t{ion_moles.get("Na+", 0)}\n'
    phreeqc_code += f'\tCl\t{ion_moles.get("Cl-", 0)}\n'
    phreeqc_code += f'\tCa\t{ion_moles.get("Ca+2", 0)}\n'
    phreeqc_code += f'\tMg\t{ion_moles.get("Mg+2", 0)}\n'
    phreeqc_code += f'\tK\t{ion_moles.get("K+", 0)}\n'
    phreeqc_code += f'\tS(6)\t{ion_moles.get("SO4-2", 0)}\n'

    phreeqc_code += f"GAS_PHASE 1\n\t-fixed_pressure\n\t-pressure {pressure_atm}\n"
    phreeqc_code += f"\t-volume 1.0\n\t CO2(g) {p_co2}\n\tH2O(g) {p_h2o}\n"
    phreeqc_code += f"SELECTED_OUTPUT\n\t-file out.tsv\n\t-totals C(4)\n\t-solution True\n\t-gases CO2(g)\n\t-saturation_indices CO2(g)\nEND"
    filename = "out.pqi"


    output_file = os.path.join(pqi_file_path, filename)
    with open(output_file, "w") as pqi:
        pqi.write(phreeqc_code)


    subprocess.run(
        [
            "phreeqc",
            f"{os.path.join(pqi_file_path, filename)}",
            f"{os.path.join(pqi_file_path, filename).replace('.pqi', '.pqo')}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )



    simulation = pd.read_csv("out.tsv", sep="\t")
    simulation.columns = simulation.columns.str.replace(" ", "")
    simulation.pressure = simulation.pressure * 0.101325

    os.remove("out.pqi")
    os.remove("out.tsv")
    os.remove("out.pqo")
    #os.remove("phreeqc.log")
    #os.remove("error.inp")

    try:
        output = float(simulation.iloc[-1]["C(4)"]), float(simulation.iloc[-1]["pH"])
    except IndexError:
        output = (np.nan, np.nan)
    return output


def simulate_varying_pressure(temperature, ion_moles, database, print_code=False, pqi_file_path="."):
    database_path = os.path.join(database_folder, f'{database}.dat')

    temperature_c = temperature - 273.15

    phreeqc_code = f"DATABASE {database_path}\n\n"

    phreeqc_code += "SOLUTION 1\n"
    phreeqc_code += f"\ttemperature\t{temperature_c}\n"
    phreeqc_code += "\tunits\tmol/kgw\n"
    phreeqc_code += f'\tNa\t{ion_moles.get("Na+", 0)}\n'
    phreeqc_code += f'\tCl\t{ion_moles.get("Cl-", 0)}\n'
    phreeqc_code += f'\tCa\t{ion_moles.get("Ca+2", 0)}\n'
    phreeqc_code += f'\tMg\t{ion_moles.get("Mg+2", 0)}\n'
    phreeqc_code += f'\tK\t{ion_moles.get("K+", 0)}\n'
    phreeqc_code += f'\tS(6)\t{ion_moles.get("SO4-2", 0)}\n'
    
    #if fix_ph != None:
    #    phreeqc_code += 'PHASES\n\tFix_H+\n\tH+ = H+\n\tlog_k  0.0\nEND\n'
    #    phreeqc_code += f'EQUILIBRIUM_PHASES 1\n\tFix_H+\t-{fix_ph}\tNaOH\t10.0\n'
    phreeqc_code  += f'GAS_PHASE 1\n\t-fixed_volume\n\tCO2(g)\t0\n\tH2O(g)\t0\n'
    phreeqc_code  += f'REACTION 1\n\tCO2 1;\t  0 100*0.5\n'
    phreeqc_code  += 'INCREMENTAL_REACTIONS true\n'
    temperature = str(temperature).replace('.', '_')
    phreeqc_code += f'SELECTED_OUTPUT\n\t-file {os.path.join(temp_files_path, "out.tsv")}\n\t-totals\tC(4)\n\t-solution True\n\t-gases\tCO2(g)\n\t-saturation_indices CO2(g)\nEND\n'

    filename = 'out.pqi'
    if print_code:
        print(phreeqc_code)
    pqi = open(os.path.join(temp_files_path, filename), 'w')
    pqi.write(phreeqc_code)

    pqi.close()

    subprocess.run(['phreeqc', f"{os.path.join(temp_files_path, filename)}", f"{os.path.join(temp_files_path, filename).replace('.pqi', '.pqo')}"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    #simulation = pd.read_csv(f'simulation_data/{salt}-{moles}m@{temperature}K.tsv', sep='\t')
    simulation = pd.read_csv(os.path.join(temp_files_path, "out.tsv"), sep='\t')
    simulation.columns = simulation.columns.str.replace(' ', '')
    simulation.pressure = simulation.pressure * 0.101325
    simulation.rename(columns = {'C(4)': 'Dissolved CO2 (mol/kg)', 'pressure': 'Pressure (MPa)'}, inplace=True)

    if os.path.exists("error.inp"):
        os.remove("error.inp")
        
    if os.path.exists("phreeqc.log"):
        os.remove("phreeqc.log")

    return simulation[['Pressure (MPa)', 'Dissolved CO2 (mol/kg)']]

if __name__ == "__main__":
    # Main script to calculate R2 score using experimental_test.csv
    import pandas as pd
    from sklearn.metrics import r2_score

    # Load experimental test data
    df = pd.read_csv(experimental_test)
    predictions = []
    # Run PHREEQC simulations for each experimental condition
    for _, row in df.iterrows():
        temperature = row['Temperature (K)']
        pressure = row['Pressure (MPa)']
        ion_moles = {
            "Na+": row['Na+'],
            "Cl-": row['Cl-'],
            "Ca+2": row['Ca+2'],
            "Mg+2": row['Mg+2'],
            "K+": row['K+'],
            "SO4-2": row['SO4-2'],
        }
        pred_co2, _ = simulate_single_state(temperature, pressure, ion_moles)
        predictions.append(pred_co2)
    # Compute R^2 score between predicted and experimental dissolved CO2
    df['Predicted CO2'] = predictions
    # exclude cases where prediction overshoots experimental by more than 2 mol/kg
    df['diff'] = df['Predicted CO2'] - df['Dissolved CO2 (mol/kg)']
    df_filtered = df[df['diff'] <= 2]
    # compute R2 on filtered data
    r2 = r2_score(df_filtered['Dissolved CO2 (mol/kg)'], df_filtered['Predicted CO2'])
    print(f"Excluded {len(df) - len(df_filtered)} cases where pred-act > 2")
    print(df_filtered[['Dissolved CO2 (mol/kg)', 'Predicted CO2']])
    print(f"PHREEQC prediction R2 score: {r2:.4f}")

    # Plot predicted vs actual dissolved CO2 (filtered)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(df_filtered['Dissolved CO2 (mol/kg)'], df_filtered['Predicted CO2'], color='blue', label='Filtered Data')
    # plot unity line
    # compute axis limits on filtered data
    min_val = min(df_filtered['Dissolved CO2 (mol/kg)'].min(), df_filtered['Predicted CO2'].min())
    max_val = max(df_filtered['Dissolved CO2 (mol/kg)'].max(), df_filtered['Predicted CO2'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal')
    plt.xlabel('Experimental Dissolved CO2 (mol/kg)')
    plt.ylabel('Predicted Dissolved CO2 (mol/kg)')
    plt.title(f'Predicted vs Actual PHREEQC (R² = {r2:.4f})')
    plt.legend()
    plt.grid(True)
    plt.show()