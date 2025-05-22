import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc
from src.PHREEQC.simulation import *

def run(output_file = "data/simulation_clean.csv", n = 100, debug = False):
    if debug: print('Generating simulation data using single salts...')
    salt_list = ["NaCl", "KCl", "MgCl2", "CaCl2", "Na2SO4"]
    all_rows = []

    for salt in salt_list:
        if debug: print('Generating data for salt:', salt)
        for _ in range(n):
            conc = np.random.uniform(0, 5)
            temp = np.random.uniform(273.15, 573.15)
            conc = np.round(conc, 4)
            temp = np.round(temp, 2)

            Na = K = Mg = Ca = SO4 = Cl = 0.0
            if salt == "NaCl":
                Na = conc
                Cl = conc
            elif salt == "KCl":
                K = conc
                Cl = conc
            elif salt == "MgCl2":
                Mg = conc
                Cl = 2 * conc
            elif salt == "CaCl2":
                Ca = conc
                Cl = 2 * conc
            elif salt == "Na2SO4":
                Na = 2 * conc
                SO4 = conc

            # Run the simulation for the current salt; the simulation returns multiple rows,
            # each corresponding to a different (Pressure, Dissolved CO2) pair.
            sim_df = simulation.simulate_varying_pressure(temp, salts=[salt], moles=[conc])

            # For each row in the simulation output, create a final row with constant
            # temperature and salt composition.
            for _, sim_row in sim_df.iterrows():
                row = {
                    "Temperature (K)": temp,
                    "Pressure (MPa)": sim_row["Pressure (MPa)"],
                    "Na+": Na,
                    "K+": K,
                    "Mg+2": Mg,
                    "Ca+2": Ca,
                    "SO4-2": SO4,
                    "Cl-": Cl,
                    "Dissolved CO2 (mol/kg)": sim_row["Dissolved CO2 (mol/kg)"],
                }
                all_rows.append(row)

    sim = pd.DataFrame(all_rows)
    sim = sim.query("`Dissolved CO2 (mol/kg)` < 4")
    sim["Pressure (MPa)"] = sim["Pressure (MPa)"].round(2)
    sim["Temperature (K)"] = sim["Temperature (K)"].round(2)
    sim["Na+"] = sim["Na+"].round(4)
    sim["K+"] = sim["K+"].round(4)
    sim["Mg+2"] = sim["Mg+2"].round(4)
    sim["Ca+2"] = sim["Ca+2"].round(4)
    sim["SO4-2"] = sim["SO4-2"].round(4)
    sim["Cl-"] = sim["Cl-"].round(4)
    sim["Dissolved CO2 (mol/kg)"] = sim["Dissolved CO2 (mol/kg)"].round(4)
    sim.to_csv(output_file, index=False)
