import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple
from DuanSun2006 import DuanSun2006
from SpanWagner import vaporization_curve
from sklearn.model_selection import train_test_split

def generate_dataset(
    pressure_range: List[float],
    temperature_range: List[float],
    salt_concentrations: Dict[str, List[float]],
    include_mixed_salts: bool = True,
    output_file: str = "co2_solubility_dataset.csv"
) -> pd.DataFrame:
    model = DuanSun2006()
    
    data = []
    
    # Function to calculate ion molalities from salt concentrations
    def calculate_ion_molalities(salt_combo: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate individual ion molalities from salt concentrations
        ensuring charge balance.
        """
        # Initialize all ion molalities to zero
        molalities = {
            "Na+": 0.0,
            "K+": 0.0,
            "Ca+2": 0.0,
            "Mg+2": 0.0,
            "SO4-2": 0.0,
            "Cl-": 0.0
        }
        
        # Add contributions from each salt
        if "NaCl" in salt_combo:
            molalities["Na+"] += salt_combo["NaCl"]
            molalities["Cl-"] += salt_combo["NaCl"]
            
        if "KCl" in salt_combo:
            molalities["K+"] += salt_combo["KCl"]
            molalities["Cl-"] += salt_combo["KCl"]
            
        if "CaCl2" in salt_combo:
            molalities["Ca+2"] += salt_combo["CaCl2"]
            molalities["Cl-"] += 2 * salt_combo["CaCl2"]
            
        if "MgCl2" in salt_combo:
            molalities["Mg+2"] += salt_combo["MgCl2"]
            molalities["Cl-"] += 2 * salt_combo["MgCl2"]
            
        if "Na2SO4" in salt_combo:
            molalities["Na+"] += 2 * salt_combo["Na2SO4"]
            molalities["SO4-2"] += salt_combo["Na2SO4"]
            
        return molalities
    
    # Generate single salt solutions (all combinations where just one salt is present)
    for P in pressure_range:
        for T in temperature_range:
            
            # First, handle pure water (no salts)
            try:
                pure_water_molalities = calculate_ion_molalities({})
                
                P_bar = P * 10  # Convert MPa to bar for intermediate calculations

                # Calculate log activity using DuanSun2006 method
                log_activity = model.calculate_log_activity(P_bar, T, pure_water_molalities)
                
                # Calculate CO2 solubility
                co2_solubility = model.calculate_CO2_solubility(
                    P, T, pure_water_molalities, model="DuanSun"
                )
                
                # Store pure water data
                data_entry = {
                    "Pressure_MPa": P,
                    "Temperature_K": T,
                    "CO2_fugacity": co2_fugacity,
                    "CO2_vap_mol_frac": co2_vapor_mole_frac,
                    "CO2_mu_liquid": co2_mu_liquid,
                    "Log_activity": log_activity,
                    "Dissolved_CO2_mol_kg": co2_solubility
                }
                
                # Add salt concentrations (all zeros for pure water)
                for salt in salt_concentrations:
                    data_entry[f"{salt}_molality"] = 0.0
                
                # Add ion molalities
                for ion, molality in pure_water_molalities.items():
                    ion_name = ion.replace("+", "").replace("-", "")
                    data_entry[f"{ion_name}_molality"] = molality
                
                data.append(data_entry)
            except Exception as e:
                print(f"Error with pure water at P={P} MPa, T={T} K: {str(e)}")
            
            # Now handle single salt solutions
            for salt_name, concentrations in salt_concentrations.items():
                for salt_conc in concentrations:
                    if salt_conc == 0.0:
                        continue  # Skip zero concentration (already covered by pure water)

                    salt_combo = {salt_name: salt_conc}
                    molalities = calculate_ion_molalities(salt_combo)
                    
                    try:
                        model._load_DuanSun_parameters()  # Ensure we're using DuanSun parameters
                        
                        P_bar = P * 10  # Convert MPa to bar for intermediate calculations
                        
                        # Calculate fugacity
                        co2_fugacity = model.co2_fugacity(P_bar, T)
                        
                        # Calculate CO2 vapor mole fraction
                        co2_vapor_mole_frac = model.calculate_CO2_vap_mol_frac(P_bar, T)
                        
                        # Calculate chemical potential
                        co2_mu_liquid = model.calculate_CO2_mu_liquid(P_bar, T)
                        
                        # Calculate log activity using DuanSun2006 method
                        log_activity = model.calculate_log_activity(P_bar, T, molalities)
                        
                        # Calculate CO2 solubility
                        co2_solubility = model.calculate_CO2_solubility(
                            P, T, molalities, model="DuanSun"
                        )
                        
                        # Store single salt solution data with selected fields
                        data_entry = {
                            "Pressure_MPa": P,
                            "Temperature_K": T,
                            "CO2_fugacity": co2_fugacity,
                            "CO2_vap_mol_frac": co2_vapor_mole_frac,
                            "CO2_mu_liquid": co2_mu_liquid,
                            "Log_activity": log_activity,
                            "Dissolved_CO2_mol_kg": co2_solubility
                        }
                        
                        # Add salt concentrations
                        for salt in salt_concentrations:
                            data_entry[f"{salt}_molality"] = salt_combo.get(salt, 0.0)
                        
                        # Add ion molalities
                        for ion, molality in molalities.items():
                            ion_name = ion.replace("+", "").replace("-", "")
                            data_entry[f"{ion_name}_molality"] = molality
                        
                        data.append(data_entry)
                    except Exception as e:
                        print(f"Error with {salt_name}={salt_conc} at P={P} MPa, T={T} K: {str(e)}")
    
    # Generate mixed salt solutions if requested
    if include_mixed_salts:
        # Select a subset of combinations to avoid explosion of combinations
        # Here we'll generate some binary and ternary mixtures
        binary_pairs = [
            ("NaCl", "KCl"),
            ("NaCl", "CaCl2"),
            ("NaCl", "MgCl2"),
            ("NaCl", "Na2SO4"),
            ("KCl", "CaCl2"),
            ("KCl", "MgCl2")
        ]
        
        ternary_sets = [
            ("NaCl", "KCl", "CaCl2"),
            ("NaCl", "KCl", "MgCl2"),
            ("NaCl", "CaCl2", "MgCl2")
        ]
        
        # Process binary mixtures
        for P in pressure_range:
            for T in temperature_range:
                # Binary mixtures
                for salt1, salt2 in binary_pairs:
                    # Take a few combinations of concentrations
                    for conc1 in salt_concentrations[salt1][1:3]:  # Skip 0 and take next 2
                        for conc2 in salt_concentrations[salt2][1:3]:  # Skip 0 and take next 2
                            salt_combo = {salt1: conc1, salt2: conc2}
                            molalities = calculate_ion_molalities(salt_combo)
                            
                            try:
                                model._load_DuanSun_parameters()
                                
                                P_bar = P * 10  # Convert MPa to bar
                                
                                # Calculate properties
                                co2_fugacity = model.co2_fugacity(P_bar, T)
                                co2_vapor_mole_frac = model.calculate_CO2_vap_mol_frac(P_bar, T)
                                co2_mu_liquid = model.calculate_CO2_mu_liquid(P_bar, T)
                                # Calculate log activity using DuanSun2006 method
                                log_activity = model.calculate_log_activity(P_bar, T, molalities)

                                # Calculate CO2 solubility
                                co2_solubility = model.calculate_CO2_solubility(
                                    P, T, molalities, model="DuanSun"
                                )
                                
                                # Store binary mixture data with selected fields
                                data_entry = {
                                    "Pressure_MPa": P,
                                    "Temperature_K": T,
                                    "CO2_fugacity": co2_fugacity,
                                    "CO2_vap_mol_frac": co2_vapor_mole_frac,
                                    "CO2_mu_liquid": co2_mu_liquid,
                                    "Log_activity": log_activity,
                                    "Dissolved_CO2_mol_kg": co2_solubility
                                }
                                
                                # Add salt concentrations
                                for salt in salt_concentrations:
                                    data_entry[f"{salt}_molality"] = salt_combo.get(salt, 0.0)
                                
                                # Add ion molalities
                                for ion, molality in molalities.items():
                                    ion_name = ion.replace("+", "").replace("-", "")
                                    data_entry[f"{ion_name}_molality"] = molality
                                
                                data.append(data_entry)
                            except Exception as e:
                                print(f"Error with binary mixture {salt_combo} at P={P} MPa, T={T} K: {str(e)}")
                
                # Ternary mixtures - more limited to avoid too many combinations
                for salt1, salt2, salt3 in ternary_sets:
                    # Take only one concentration value for each salt
                    conc1 = salt_concentrations[salt1][1]  # Skip 0 and take first non-zero
                    conc2 = salt_concentrations[salt2][1]  # Skip 0 and take first non-zero
                    conc3 = salt_concentrations[salt3][1]  # Skip 0 and take first non-zero
                    
                    salt_combo = {salt1: conc1, salt2: conc2, salt3: conc3}
                    molalities = calculate_ion_molalities(salt_combo)
                    
                    try:
                        model._load_DuanSun_parameters()
                        
                        P_bar = P * 10  # Convert MPa to bar
                        
                        # Calculate properties
                        co2_fugacity = model.co2_fugacity(P_bar, T)
                        co2_vapor_mole_frac = model.calculate_CO2_vap_mol_frac(P_bar, T)
                        co2_mu_liquid = model.calculate_CO2_mu_liquid(P_bar, T)
                        # Calculate log activity using DuanSun2006 method
                        log_activity = model.calculate_log_activity(P_bar, T, molalities)

                        # Calculate CO2 solubility
                        co2_solubility = model.calculate_CO2_solubility(
                            P, T, molalities, model="DuanSun"
                        )
                        
                        # Store ternary mixture data with selected fields
                        data_entry = {
                            "Pressure_MPa": P,
                            "Temperature_K": T,
                            "CO2_fugacity": co2_fugacity,
                            "CO2_vap_mol_frac": co2_vapor_mole_frac,
                            "CO2_mu_liquid": co2_mu_liquid,
                            "Log_activity": log_activity,
                            "Dissolved_CO2_mol_kg": co2_solubility
                        }
                        
                        # Add salt concentrations
                        for salt in salt_concentrations:
                            data_entry[f"{salt}_molality"] = salt_combo.get(salt, 0.0)
                        
                        # Add ion molalities
                        for ion, molality in molalities.items():
                            ion_name = ion.replace("+", "").replace("-", "")
                            data_entry[f"{ion_name}_molality"] = molality
                        
                        data.append(data_entry)
                    except Exception as e:
                        print(f"Error with ternary mixture {salt_combo} at P={P} MPa, T={T} K: {str(e)}")
    
    df = pd.DataFrame(data)
    # Round numeric columns: Pressure_MPa and Temperature_K to 2 decimals, others to 4
    num_cols = df.select_dtypes(include='number').columns.tolist()
    other_cols = [c for c in num_cols if c not in ['Pressure_MPa', 'Temperature_K']]
    df[other_cols] = df[other_cols].round(4)
    df[['Pressure_MPa', 'Temperature_K']] = df[['Pressure_MPa', 'Temperature_K']].round(2)
    
    # Rename columns according to new nomenclature
    column_mapping = {
        'Temperature_K': 'Temperature (K)',
        'Pressure_MPa': 'Pressure (MPa)',
        'Na_molality': 'Na+',
        'K_molality': 'K+',
        'Mg2_molality': 'Mg+2',
        'Ca2_molality': 'Ca+2',
        'SO42_molality': 'SO4-2',
        'Cl_molality': 'Cl-',
        'CO2_fugacity': 'Fugacity',
        'Log_activity': 'Activity',
        'CO2_mu_liquid': 'ChemicalPotential',
        'CO2_vap_mol_frac': 'MolarFraction',
        'Dissolved_CO2_mol_kg': 'Dissolved CO2 (mol/kg)'
    }

    #df = df[['Temperature_K', ']]
    df = df[column_mapping.keys()]  # Reorder columns to match the mapping
    
    # Apply the column renaming
    df = df.rename(columns=column_mapping)

    
    
    df.to_csv(output_file, index=False)
    print(f"Dataset generated and saved to {output_file}")
    return df

if __name__ == "__main__":
    pressure_range = np.linspace(0.5, 100.0, 50)  # MPa, 10 evenly spaced points
    temperature_range = np.linspace(273.15, 500.15, 50)  # K, 6 temperatures
    
    # Define salt concentrations instead of individual ion molalities
    salt_concentrations = {
        "NaCl": np.linspace(0.0, 6, 7).tolist(),
        "KCl": np.linspace(0.0, 4, 5).tolist(),
        "CaCl2": np.linspace(0.0, 4, 5).tolist(),
        "MgCl2": np.linspace(0.0, 5, 6).tolist(),
        "Na2SO4": np.linspace(0.0, 5, 6).tolist()
    }
    
    # Generate the dataset with both single salt and mixed salt solutions
    df = generate_dataset(pressure_range, temperature_range, salt_concentrations, include_mixed_salts=True, output_file="/home/jalil/OneDrive/Papers/CCUS/MachineLearning/data/processed/DuanSunData.csv")
    
    print(f"Generated {len(df)} data points")
    print("First few rows of the dataset:")
    print(df.head())
    print("\nDataset statistics:")
    print(df.describe())
    
    # Split dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_file = "/home/jalil/OneDrive/Papers/CCUS/MachineLearning/data/processed/DuanSun_train.csv"
    test_file = "/home/jalil/OneDrive/Papers/CCUS/MachineLearning/data/processed/DuanSun_test.csv"
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    print(f"Training set saved to {train_file}")
    print(f"Testing set saved to {test_file}")