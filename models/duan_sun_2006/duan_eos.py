
import numpy as np
from scipy.optimize import fsolve, newton
import matplotlib.pyplot as plt

a1 = 8.99288497e-2
a2 = -4.94783127e-1
a3 = 4.77922245e-2
a4 = 1.03808883e-2
a5 = -2.82516861e-2
a6 = 9.49887563e-2
a7 = 5.20600880e-4
a8 = -2.93540971e-4
a9 = -1.77265112e-3
a10 = -2.51101973e-5
a11 = 8.93353441e-5
a12 = 7.88998563e-5
a13 = -1.66727022e-2
alpha = -1.66727022e-2
a14 = 1.39800000
beta = 1.39800000
a15 = 2.96000000e-2
gamma = 2.96000000e-2


Pc = 7.38e6 #Pa
Tc = 304.13 #K
R = 8.314467 # Pa m^3 / mol K

def Z_func(Vr, params):
    Tr = params['Tr']
    Pr = params['Pr']
    return 1 + (a1 + a2/Tr**2 + a3/Tr**3)/Vr + (a4+a5/Tr**2+a6/Tr**3)/Vr**2 + (a7+a8/Tr**2+a9/Tr**3)/Vr**4 + (a10+a11/Tr**2+a12/Tr**3)/Vr**5 + (a13/(Tr**3*Vr**2))*(a14+a15/Vr**2)*np.exp(-a15/Vr**2) - Pr*Vr/Tr

'''def Z_func(Vr, params):
    Tr = params['Tr']
    Pr = params['Pr']
    B = a1 + a2/ Tr ** 2 + a3/ Tr ** 3
    C = a4 + a5/ Tr ** 2 + a6/ Tr ** 3
    D = a7 + a8/ Tr ** 2 + a9/ Tr ** 3
    E = a10 + a11/ Tr ** 2 + a12/ Tr ** 3
    F = alpha / Tr ** 3

    Z = 1 + B / Vr + C / Vr ** 2 + D / Vr ** 4 + E / Vr ** 5 + F / Vr ** 2 * (beta + gamma / Vr ** 2) * np.exp(-gamma / Vr ** 2)

    return Z'''

#Vc = R * Tc / Pc
def co2_fugacity(P, T):
    Pr = P / Pc
    Tr = T / Tc
    Vc = R * Tc / Pc
    Vr = fsolve(Z_func, 1, args={'Tr': Tr, 'Pr': Pr})[0]
    #Vr = newton(Z_func, 1, args={'Tr': Tr, 'Pr': Pr})[0]

    Z = Pr*Vr/Tr

    term1 = (a1 + a2/Tr**2 + a3/ Tr**3)/Vr
    term2 = (a4+a5/Tr**2+a6/Tr**3)/(2*Vr**2) 
    term3 = (a7+a8/Tr**2 + a9/Tr**3)/(4*Vr**4)
    term4 = (a10+a11/Tr**2+a12/Tr**3)/(5*Vr**5)
    term5 = a13/(2*(Tr**3)*a15) * (a14 + 1 - (a14+1+a15/Vr**2)*np.exp(-a15/Vr**2))
    ln_co2_fugacity = Z - 1 - np.log(Z) + term1 + term2 + term3 + term4 + term5
    #B = a1 + a2/ Tr ** 2 + a3/ Tr ** 3
    #C = a4 + a5/ Tr ** 2 + a6/ Tr ** 3
    #D = a7 + a8/ Tr ** 2 + a9/ Tr ** 3
    #E = a10 + a11/ Tr ** 2 + a12/ Tr ** 3
    #F = alpha / Tr ** 3

    #G = F/(2*gamma) * (beta+1-(beta+1+gamma/Vr**2)*np.exp(-gamma/Vr**2))

    #ln_co2_fugacity = Z - 1 - np.log(Z)+ B/Vr + C/(2*Vr**2) + D / (4 * Vr ** 4) + E / (5 * Vr ** 5) + G
    co2_fugacity = np.exp(ln_co2_fugacity)

    return co2_fugacity

pressures = np.linspace(0, 60e6, 200)

for temperature in [280, 290, 300, 320, 340, 380]:

    fugacities = [co2_fugacity(pressure, temperature) for pressure in pressures]

    plt.plot(pressures, fugacities, label=f'{temperature}')


plt.legend()
plt.ylim([0, 1])
plt.show()