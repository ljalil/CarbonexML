# A New Equation of State for Carbon Dioxide Covering the Fluid Region from the Triple‐Point Temperature to 1100 K at Pressures up to 800 MPa
# Roland Span; Wolfgang Wagner
# Journal of Physical and Chemical Reference Data 25, 1509 (1996); https://doi.org/10.1063/1.555991

import numpy as np

P_triple = 0.51795
T_triple = 216.592
P_critical = 7.3773
T_critical = 304.1282


def melting_curve(T: float) -> float:
    if T < T_triple:
        raise ValueError(
            f"Temperature for sublimation curve should be above the triple point temperature (T_triple = {T_triple})"
        )

    a = [1955.5390, 2055.4593]
    P_m = P_triple * (1 + a[0] * (T / T_triple - 1) + a[1] * (T / T_triple - 1) ** 2)
    return P_m


def sublimation_curve(T: float) -> float:
    if T < 0 or T > T_triple:
        raise ValueError(
            f"Temperature for sublimation curve should be between 0 and triple point temperature (T_triple = {T_triple})"
        )

    a = [-14.740846, 2.4327015, -5.3061778]
    P_sub = (
        np.exp(
            T_triple
            / T
            * (
                a[0] * (1 - T / T_triple)
                + a[1] * (1 - T / T_triple) ** 1.9
                + a[2] * (1 - T / T_triple) ** 2.9
            )
        )
        * P_triple
    )

    return P_sub


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
