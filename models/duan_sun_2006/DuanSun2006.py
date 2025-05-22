import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
#from SpanWagner import vaporization_curve

P_triple = 0.51795
T_triple = 216.592
P_critical = 7.3773
T_critical = 304.1282

def vaporization_curve(T: float) -> float:
    if T < T_triple or T > T_critical:
        raise ValueError(
            f"Temperature for vaporization curve should be between triple point and critical point temperatures (T_triple = {T_triple}), T_critical = {T_critical}"
        )

    a = [-7.0602087, 1.9391218, -1.6463597, -3.2995634]
    t = [1, 1.5, 2, 4]
    sum_value = 0
    for a_i, t_i in zip(a, t):
        sum_value += a_i * ((1 - T / T_critical) ** t_i)
    return np.exp((T_critical / T) * sum_value) * P_critical


class DuanSun2006:
    def __init__(self):
        self._load_EOS_parameters()
        self._load_DuanSun_parameters()
        self._load_Guo_parameters()

        self.params = None

    def _load_EOS_parameters(self) -> None:
        self.EOS_PARAMS = [
            # Range 1
            [
                1.0,
                4.7586835e-3,
                -3.3569963e-6,
                0.0,
                -1.3179396,
                -3.8389101e-6,
                0.0,
                2.2815104e-3,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            # Range 2
            [
                -7.1734882e-1,
                1.5985379e-4,
                -4.9286471e-7,
                0.0,
                0.0,
                -2.7855285e-7,
                1.1877015e-9,
                0.0,
                0.0,
                0.0,
                0.0,
                -96.539512,
                4.4774938e-1,
                101.81078,
                5.3783879e-6,
            ],
            # Range 3
            [
                -6.5129019e-2,
                -2.1429977e-4,
                -1.1444930e-6,
                0.0,
                0.0,
                -1.1558081e-7,
                1.1952370e-9,
                0.0,
                0.0,
                0.0,
                0.0,
                -221.34306,
                0.0,
                71.820393,
                6.6089246e-6,
            ],
            # Range 4
            [
                5.0383896,
                -4.4257744e-3,
                0.0,
                1.9572733,
                0.0,
                2.4223436e-6,
                0.0,
                -9.3796135e-4,
                -1.502603,
                3.0272240e-3,
                -31.377342,
                -12.847063,
                0.0,
                0.0,
                -1.5056648e-5,
            ],
            # Range 5
            [
                -16.063152,
                -2.7057990e-3,
                0.0,
                1.4119239e-1,
                0.0,
                8.1132965e-7,
                0.0,
                -1.1453082e-4,
                2.3895671,
                5.0527457e-4,
                -17.76346,
                985.92232,
                0.0,
                0.0,
                -5.4965256e-7,
            ],
            # Range 6
            [
                -1.5693490e-1,
                4.4621407e-4,
                -9.1080591e-7,
                0.0,
                0.0,
                1.0647399e-7,
                2.4273357e-10,
                0.0,
                3.5874255e-1,
                6.3319710e-5,
                -249.89661,
                0.0,
                0.0,
                888.768,
                -6.6348003e-7,
            ],
        ]
    
    def _load_DuanSun_parameters(self) -> None:
        self.CO2_MU_LIQUID_COEFFS = [
            28.9447706,
            -0.0354581768,
            -4770.67077,
            1.02782768e-5,
            33.8126098,
            9.04037140e-3,
            -1.14934031e-3,
            -0.307405726,
            -0.0907301486,
            9.32713393e-4,
            0,
        ]

        self.LAMBDA_CO2_NA = [
            -0.411370585,
            6.07632013e-4,
            97.5347708,
            0,
            0,
            0,
            0,
            -0.0237622469,
            0.0170656236,
            0,
            1.41335834e-5,
        ]

        self.ZETA_CO2_NA_CL = [
            3.36389723e-4,
            -1.98298980e-5,
            0,
            0,
            0,
            0,
            0,
            2.12220830e-3,
            -5.24873303e-3,
            0,
            0,
        ]
    
    def _load_Guo_parameters(self) -> None:
        self.CO2_MU_LIQUID_COEFFS = [
            2.52671156e1,
            -2.99024399e-2,
            -4.11129437e3,
            1.23091891e-5,
            -4.86804783e1,
            9.66527036e-2,
            -1.43035525e-2,
            -4.28379454,
            2.70920374e-1,
            -1.64011109e-2,
            -1.24611227e-4,
        ]

        self.LAMBDA_CO2_NA = [
            2.32329297,
            -5.52304993e-3,
            -3.21472657e2,
            1.82754454e-6,
            8.16653987e1,
            -4.06006390e-2,
            6.10232321e-3,
            1.88150995,
            -2.50830982e-1,
            2.48768009e-2,
            1.01658267e-4,
        ]

        self.ZETA_CO2_NA_CL = [
            -1.10067716,
            2.58535943e-3,
            1.61555536e2,
            -9.67677864e-7,
            -3.24768654e1,
            1.30813929e-2,
            -1.97407284e-3,
            -6.02020260e-1,
            9.35464931e-2,
            -9.06376267e-3,
            -3.63798082e-5,
        ]

    def _determine_equation_range(self, T: float, P: float) -> int:
        if T < 304.1282:
            P1 = vaporization_curve(T) * 10  # convert from MPa to bar
        elif 304.1282 <= T < 405.0:
            P1 = 75.0 + (T - 305.0) * 1.25  # in bar
        elif T >= 405.0:
            P1 = 200.0  # in bar

        # Determine the T-P range
        if 273.0 < T < 573.0 and P < P1:
            range_idx = 0  # Range 1
        elif 273.0 < T < 340.0 and P1 <= P < 1000.0:
            range_idx = 1  # Range 2
        elif 273.0 < T < 340.0 and P >= 1000.0:
            range_idx = 2  # Range 3
        elif 340.0 <= T < 435.0 and P1 <= P <= 1000.0:
            range_idx = 3  # Range 4
        elif 340.0 <= T < 435.0 and P >= 1000.0:
            range_idx = 4  # Range 5
        elif T >= 435.0:  # and P > P1:
            range_idx = 5  # Range 6
        else:
            raise ValueError(
                f"Input conditions do not fall within defined T-P ranges {T=} {P=}"
            )

        return range_idx

    def calculate_CO2_mu_liquid(self,P: float, T: float) -> float:
        """
        calculates the chemical potential (mu) of CO2 in the liquid phase
        """
        return (
            self.CO2_MU_LIQUID_COEFFS[0]
            + self.CO2_MU_LIQUID_COEFFS[1] * T
            + self.CO2_MU_LIQUID_COEFFS[2] / T
            + self.CO2_MU_LIQUID_COEFFS[3] * T**2
            + self.CO2_MU_LIQUID_COEFFS[4] / (630 - T)
            + self.CO2_MU_LIQUID_COEFFS[5] * P
            + self.CO2_MU_LIQUID_COEFFS[6] * P * np.log(T)
            + self.CO2_MU_LIQUID_COEFFS[7] * P / T
            + self.CO2_MU_LIQUID_COEFFS[8] * P / (630 - T)
            + self.CO2_MU_LIQUID_COEFFS[9] * P**2 / (630 - T) ** 2
            + self.CO2_MU_LIQUID_COEFFS[10] * T * np.log(P)
        )

    def calculate_lambda_CO2_Na(self, P: float, T: float) -> float:
        """
        Calculate the binary interaction parameter lambda for CO2 and Na+.
        """
        return (
            self.LAMBDA_CO2_NA[0]
            + self.LAMBDA_CO2_NA[1] * T
            + self.LAMBDA_CO2_NA[2] / T
            + self.LAMBDA_CO2_NA[3] * T**2
            + self.LAMBDA_CO2_NA[4] / (630 - T)
            + self.LAMBDA_CO2_NA[5] * P
            + self.LAMBDA_CO2_NA[6] * P * np.log(T)
            + self.LAMBDA_CO2_NA[7] * P / T
            + self.LAMBDA_CO2_NA[8] * P / (630 - T)
            + self.LAMBDA_CO2_NA[9] * P**2 / (630 - T) ** 2
            + self.LAMBDA_CO2_NA[10] * T * np.log(P)
        )

    def calculate_zeta_CO2_Na_Cl(self, P: float, T: float) -> float:
        """
        Calculate the ternary interaction parameter zeta for CO2, Na+, and Cl-.
        """
        return (
            self.ZETA_CO2_NA_CL[0]
            + self.ZETA_CO2_NA_CL[1] * T
            + self.ZETA_CO2_NA_CL[2] / T
            + self.ZETA_CO2_NA_CL[3] * T**2
            + self.ZETA_CO2_NA_CL[4] / (630 - T)
            + self.ZETA_CO2_NA_CL[5] * P
            + self.ZETA_CO2_NA_CL[6] * P * np.log(T)
            + self.ZETA_CO2_NA_CL[7] * P / T
            + self.ZETA_CO2_NA_CL[8] * P / (630 - T)
            + self.ZETA_CO2_NA_CL[9] * P**2 / (630 - T) ** 2
            + self.ZETA_CO2_NA_CL[10] * T * np.log(P)
        )

    def calculate_CO2_vap_mol_frac(self, P: float, T: float) -> float:
        """
        Calculate the mole fraction of CO2 in the vapor phase.
        """
        Tc = 647.29
        Pc = 220.85
        t = (T - Tc) / Tc
        P_water = (Pc * T / Tc) * (
            1
            - 38.640844 * (-t) ** 1.9
            + 5.8948420 * t
            + 59.876516 * t**2
            + 26.654627 * t**3
            + 10.637097 * t**4
        )

        mole_fraction = (P - P_water) / P
        if mole_fraction < 0:
            return 1e-6
        return mole_fraction

    def co2_fugacity(self, P: float, T: float) -> float:
        """
        Calculate the fugacity of CO2 using the Duan and Sun (2006) equation of state.
        """
        range_idx = self._determine_equation_range(T, P)

        # Extract coefficients
        c1 = self.EOS_PARAMS[range_idx][0]
        c2 = self.EOS_PARAMS[range_idx][1]
        c3 = self.EOS_PARAMS[range_idx][2]
        c4 = self.EOS_PARAMS[range_idx][3]
        c5 = self.EOS_PARAMS[range_idx][4]
        c6 = self.EOS_PARAMS[range_idx][5]
        c7 = self.EOS_PARAMS[range_idx][6]
        c8 = self.EOS_PARAMS[range_idx][7]
        c9 = self.EOS_PARAMS[range_idx][8]
        c10 = self.EOS_PARAMS[range_idx][9]
        c11 = self.EOS_PARAMS[range_idx][10]
        c12 = self.EOS_PARAMS[range_idx][11]
        c13 = self.EOS_PARAMS[range_idx][12]
        c14 = self.EOS_PARAMS[range_idx][13]
        c15 = self.EOS_PARAMS[range_idx][14]

        # Calculate fugacity using the provided equation
        try:
            fugacity = (c1 + 
                (c2 + c3 * T + c4 / T + c5 / (T - 150.0)) * P + 
                (c6 + c7 * T + c8 / T) * P**2 + 
                (c9 + c10 * T + c11 / T) * np.log(P) + 
                (c12 + c13 * T) / P + 
                c14 / T + 
                c15 * T**2
            )
            
            return fugacity

        except ZeroDivisionError:
            raise ZeroDivisionError(
                "Division by zero encountered in fugacity calculation."
            )
        except ValueError as e:
            raise ValueError(f"Math domain error during fugacity calculation: {e}")

    def calculate_log_activity(self, P: float, T: float, molalities: Dict[str, float]) -> float:
        """
        Calculate the logarithm of the activity coefficient using Pitzer equations.
        
        Parameters:
            P: Pressure in bar
            T: Temperature in Kelvin
            molalities: Dictionary of ion molalities
            
        Returns:
            float: The logarithm of the activity coefficient
        """
        # Extract ion molalities with defaults of 0 for any missing ions
        m_na = molalities.get("Na+", 0.0)
        m_cl = molalities.get("Cl-", 0.0)
        m_k = molalities.get("K+", 0.0)
        m_ca = molalities.get("Ca+2", 0.0)
        m_mg = molalities.get("Mg+2", 0.0)
        m_so4 = molalities.get("SO4-2", 0.0)
        
        # Calculate activity coefficient using Pitzer equations
        log_activity = (
            -2
            * self.calculate_lambda_CO2_Na(P, T)
            * (m_na + m_k + 2 * m_ca + 2 * m_mg)
            - self.calculate_zeta_CO2_Na_Cl(P, T) * m_cl * (m_na + m_k + m_ca + m_mg)
            + 0.07 * m_so4
        )
        
        return log_activity

    def calculate_CO2_solubility(
        self,
        P: float,
        T: float,
        molalities: Optional[Dict[str, float]] = None,
        model: str = "DuanSun",
    ) -> float:
        """
        Calculate CO2 solubility in water or brine.

        Parameters:
            P: P in MPa (converted to bar in the function)
            T: T in Kelvin
            molalities: Dictionary of ion molalities. Expected keys are 'Na', 'Cl', 'K',
                    'Ca', 'Mg', 'SO4'. Missing ions are assumed to have zero molality.

        Returns:
            float: CO2 solubility in mol/kg water

        Examples:
            >>> model = DuanSun2006()
            >>> # Water at 323.15 K and 100 bar
            >>> model.calculate_CO2_solubility(100.0, 323.15)
            >>> # NaCl brine at 323.15 K and 100 bar
            >>> model.calculate_CO2_solubility(100.0, 323.15, {'Na': 1.0, 'Cl': 1.0})
        """
        #print(f"Calculating CO2 solubility at {T} K and {P} bar")
        #print(f"Ion molalities: {molalities}")
        if model == "DuanSun":
            self._load_DuanSun_parameters()
        elif model == "Guo":
            self._load_Guo_parameters()
        else:
            raise ValueError(f"Model {model} not recognized.")

        P = P * 10  # Convert from MPa to bar
        # Set default empty dictionary if None provided
        if molalities is None:
            molalities = {}

        # Calculate log activity using the extracted method
        log_activity = self.calculate_log_activity(P, T, molalities)

        try:
            # Calculate CO2 solubility
            prod = self.calculate_CO2_vap_mol_frac(P, T) * P * self.co2_fugacity(P, T)
            if prod <= 0:
                print(f"Negative product: {prod} (T={T} K, P={P/10} MPa, molalities={molalities})")
                print(f"CO2 fugacity: {self.co2_fugacity(P, T)}")
                print(f"CO2 mole fraction: {self.calculate_CO2_vap_mol_frac(P, T)}")

            log_CO2_molality = (
                np.log(prod)
                - self.calculate_CO2_mu_liquid(P, T)
                + log_activity
            )

            return np.exp(log_CO2_molality)
        except (ValueError, ZeroDivisionError) as e:
            print("Error calculating CO2 solubility: {e} (T={T} K, P={P} bar)")


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    # Load processed experimental data from CSV instead of ODS
    experimental = pd.read_csv(
        "/home/jalil/OneDrive/Papers/CCUS/MachineLearning/data/processed/experimental.csv"
    )

    # Create figure with 1x5 subplots for different salt types
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    # Summary table for AAD%
    summary_results = []
    
    # Define salt types and their corresponding filters
    salt_types = ["NaCl", "KCl", "CaCl2", "MgCl2", "Na2SO4"]
    salt_filters = [
        # NaCl: Only Na+ and Cl- present, equal molality
        lambda df: (df["Na+"] > 0) & (df["K+"] == 0) & (df["Mg+2"] == 0) & 
                  (df["Ca+2"] == 0) & (df["SO4-2"] == 0) & (df["Na+"] == df["Cl-"]/1),
        # KCl: Only K+ and Cl- present, equal molality
        lambda df: (df["K+"] > 0) & (df["Na+"] == 0) & (df["Mg+2"] == 0) & 
                  (df["Ca+2"] == 0) & (df["SO4-2"] == 0) & (df["K+"] == df["Cl-"]/1),
        # CaCl2: Only Ca+2 and Cl- present, 2:1 ratio
        lambda df: (df["Ca+2"] > 0) & (df["Na+"] == 0) & (df["K+"] == 0) & 
                  (df["Mg+2"] == 0) & (df["SO4-2"] == 0) & (abs(df["Cl-"] - 2*df["Ca+2"]) < 0.1),
        # MgCl2: Only Mg+2 and Cl- present, 2:1 ratio
        lambda df: (df["Mg+2"] > 0) & (df["Na+"] == 0) & (df["K+"] == 0) & 
                  (df["Ca+2"] == 0) & (df["SO4-2"] == 0) & (abs(df["Cl-"] - 2*df["Mg+2"]) < 0.1),
        # Na2SO4: Only Na+ and SO4-2 present, 2:1 ratio
        lambda df: (df["SO4-2"] > 0) & (df["K+"] == 0) & (df["Mg+2"] == 0) & 
                  (df["Ca+2"] == 0) & (df["Cl-"] == 0) & (abs(df["Na+"] - 2*df["SO4-2"]) < 0.1)
    ]
    
    # Loop through all salt types
    for i, (salt, filter_func) in enumerate(zip(salt_types, salt_filters)):
        salt_data = experimental[filter_func(experimental)]
        
        if len(salt_data) == 0:
            print(f"No experimental data found for {salt}")
            summary_results.append({"Salt": salt, "AAD%": "No data"})
            axes[i].text(0.5, 0.5, f"No data for {salt}", ha='center', va='center')
            axes[i].set_title(f"{salt}")
            continue
        
        results = []
        model = DuanSun2006()
        
        for _, row in salt_data.iterrows():
            pressure = row['Pressure (MPa)']
            temperature = row['Temperature (K)']
            
            # Extract molalities directly from the dataframe
            molalities = {
                "Na+": row["Na+"],
                "K+": row["K+"],
                "Ca+2": row["Ca+2"],
                "Mg+2": row["Mg+2"],
                "SO4-2": row["SO4-2"],
                "Cl-": row["Cl-"]
            }
            
            # Determine the salt molality based on the salt type
            if salt == "NaCl":
                salt_molality = row["Na+"]
            elif salt == "KCl":
                salt_molality = row["K+"]
            elif salt == "CaCl2":
                salt_molality = row["Ca+2"]
            elif salt == "MgCl2":
                salt_molality = row["Mg+2"]
            else:  # Na2SO4
                salt_molality = row["SO4-2"]
            
            calculated_solubility = model.calculate_CO2_solubility(pressure, temperature, molalities)
            
            results.append({
                'Pressure (MPa)': pressure,
                'Temperature (K)': temperature,
                f'{salt} (molality)': salt_molality,
                'Experimental CO2 (mol/kg)': row['Dissolved CO2 (mol/kg)'],
                'Calculated CO2 (mol/kg)': calculated_solubility,
                'Difference (%)': (calculated_solubility - row['Dissolved CO2 (mol/kg)']) / row['Dissolved CO2 (mol/kg)'] * 100
            })

        results_df = pd.DataFrame(results)
        # replace infinite differences with NaN and compute AAD% ignoring NaNs
        diffs = results_df['Difference (%)'].replace([np.inf, -np.inf], np.nan)
        aad_percent = diffs.abs().mean()
        
        # Store AAD% for summary
        summary_results.append({"Salt": salt, "AAD%": f"{aad_percent:.2f}%", "Count": len(results_df)})
        
        # Plot in the corresponding subplot
        sc = axes[i].scatter(results_df['Experimental CO2 (mol/kg)'], results_df['Calculated CO2 (mol/kg)'], c=results_df[f'{salt} (molality)'], cmap='viridis')
        max_val = max(results_df['Experimental CO2 (mol/kg)'].max(), results_df['Calculated CO2 (mol/kg)'].max())
        axes[i].plot([0, max_val], [0, max_val], 'r--')
        axes[i].set_xlabel('Experimental CO2 (mol/kg)')
        axes[i].set_ylabel('Calculated CO2 (mol/kg)')
        axes[i].set_title(f'{salt} (AAD%: {aad_percent:.2f}%)')
        axes[i].grid(True)
        # Add colorbar for molality
        cbar = fig.colorbar(sc, ax=axes[i])
        cbar.set_label(f'{salt} molality (mol/kg)')

    # Print summary table of AAD%
    summary_df = pd.DataFrame(summary_results)
    print("\nSummary of Average Absolute Deviation (AAD%) for all salts:")
    print(summary_df.to_string(index=False))
    
    plt.tight_layout()
    plt.savefig('co2_solubility_all_salts_comparison.png', dpi=300)
    plt.show()
