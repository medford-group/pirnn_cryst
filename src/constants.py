"""Physical constants for paracetamol crystallization kinetics."""

kv = 0.797          # volume shape factor [-]
rho = 1.263          # crystal density [g/cm^3]
MM = 151.163         # molar mass [g/mol]
R = 8.314            # gas constant [J/mol.K]

# Ground-truth kinetic parameters
GT_PARAMS = {
    'kb2': 6e3,       # nucleation rate constant [#/(min.g solv)]
    'alfa': 2.08,     # nucleation supersaturation exponent [-]
    'beta': 0.713,    # nucleation mass exponent [-]
    'kg': 2.73e5,     # growth rate constant [(um/min)(g/g)^(-gama_g)]
    'Eag': 4.13e4,    # growth activation energy [J/mol]
    'gama_g': 1.24,   # growth supersaturation exponent [-]
}
