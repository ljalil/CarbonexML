import os
import yaml
import subprocess
import numpy as np
import pandas as pd

cfg = yaml.safe_load(open("configs/data_params.yaml"))
database_folder = cfg["phreeqc_database_folder"]
temp_files_path = cfg["temp_files_path"]

def simulate_single_state(
    temperature, pressure, ion_moles, database = 'pitzer', pqi_file_path="."
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
    import matplotlib.pyplot as plt
    # Example usage
    temperature = 298.15  # Temperature in Kelvin
    pressure = 1.0  # Pressure in MPa
    ion_moles = {
        "Na+": 0.1,
        "Cl-": 0.1,
        "Ca+2": 0.01,
        "Mg+2": 0.01,
        "K+": 0.01,
        "SO4-2": 0.01,
    }

    results = simulate_varying_pressure(temperature, ion_moles, database='pitzer')
    plt.plot(results["Pressure (MPa)"], results["Dissolved CO2 (mol/kg)"])

    pressures = np.linspace(1, 100, 5)
    results = []
    for pressure in pressures:
        result = simulate_single_state(temperature, pressure, ion_moles)
        results.append(result[0])

    plt.scatter(pressures, results, color='red', label='Single State Simulation')

    
    plt.xlabel("Pressure (MPa)")
    plt.ylabel("Dissolved CO2 (mol/kg)")
    plt.title("Dissolved CO2 vs Pressure")
    plt.grid()
    plt.show()