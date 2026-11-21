from __future__ import annotations

import numpy as np

# JDFTx's defaults for a minimal input (no fluid). JDFTXInfile's comparison methods fill in missing tags and
# subtags from these, adjusted for the defaults that depend on other tags (see inputs._get_default_value).
default_inputs = {
    "basis": "kpoint-dependent",
    "coords-type": "Lattice",
    "core-overlap-check": "vector",
    "coulomb-interaction": "Periodic",
    "davidson-band-ratio": 1.1,
    "dump": [{"End": {"State": True}}],
    "dump-name": "$INPUT.$VAR",
    "elec-cutoff": 20.0,
    "elec-eigen-algo": "Davidson",
    "elec-ex-corr": "gga-PBE",
    "electronic-minimize": {
        "dirUpdateScheme": "FletcherReeves",
        "linminMethod": "DirUpdateRecommended",
        "nIterations": 100,
        "history": 15,
        "knormThreshold": 0,
        "maxThreshold": False,
        "energyDiffThreshold": 1e-08,
        "nEnergyDiff": 2,
        "convergeAll": False,
        "alphaTstart": 1,
        "alphaTmin": 1e-10,
        "updateTestStepSize": True,
        "alphaTreduceFactor": 0.1,
        "alphaTincreaseFactor": 3,
        "nAlphaAdjustMax": 3,
        "wolfeEnergy": 0.0001,
        "wolfeGradient": 0.9,
        "abortOnFailedStep": False,
        "fdTest": False,
    },
    "exchange-regularization": "WignerSeitzTruncated",
    "fluid": {"type": "None", "Temperature": 298.0, "Pressure": 1.01325},
    "fluid-ex-corr": {
        "kinetic": "lda-TF",
        "exchange-correlation": "lda-PZ",
    },
    "fluid-gummel-loop": {"maxIterations": 10, "Atol": 1e-05},
    "fluid-minimize": {
        "dirUpdateScheme": "PolakRibiere",
        "linminMethod": "DirUpdateRecommended",
        "nIterations": 100,
        "history": 15,
        "knormThreshold": 0,
        "maxThreshold": False,
        "energyDiffThreshold": 0,
        "nEnergyDiff": 2,
        "convergeAll": False,
        "alphaTstart": 1,
        "alphaTmin": 1e-10,
        "updateTestStepSize": True,
        "alphaTreduceFactor": 0.1,
        "alphaTincreaseFactor": 3,
        "nAlphaAdjustMax": 3,
        "wolfeEnergy": 0.0001,
        "wolfeGradient": 0.9,
        "abortOnFailedStep": False,
        "fdTest": False,
    },
    "fluid-solvent": [
        {
            "name": "H2O",
            "concentration": 55.338,
            "functional": "ScalarEOS",
            "epsBulk": 78.4,
            "pMol": 0.92466,
            "epsInf": 1.77,
            "Pvap": 1.06736e-10,
            "sigmaBulk": 4.62e-05,
            "Rvdw": 2.61727,
            "Res": 1.42,
            "tauNuc": 343133.0,  # Dumped as an int but should be a float
            "poleEl": [
                {
                    "omega0": 15.0,
                    "gamma0": 7.0,
                    "A0": 1.0,
                },
            ],
        },
    ],
    "forces-output-coords": "Positions",
    "ion-width": 0.0,  # Dumped as an int but should be a float
    "ionic-minimize": {
        "dirUpdateScheme": "L-BFGS",
        "linminMethod": "DirUpdateRecommended",
        "nIterations": 0,
        "history": 15,
        "knormThreshold": 0.0001,
        "maxThreshold": False,
        "energyDiffThreshold": 1e-06,
        "nEnergyDiff": 2,
        "convergeAll": False,
        "alphaTstart": 1,
        "alphaTmin": 1e-10,
        "updateTestStepSize": True,
        "alphaTreduceFactor": 0.1,
        "alphaTincreaseFactor": 3,
        "nAlphaAdjustMax": 3,
        "wolfeEnergy": 0.0001,
        "wolfeGradient": 0.9,
        "abortOnFailedStep": False,
        "fdTest": False,
    },
    "kpoint": {
        "k0": 0.0,
        "k1": 0.0,
        "k2": 0.0,
        "weight": 1.0,
    },
    "kpoint-folding": {
        "n0": 1,
        "n1": 1,
        "n2": 1,
    },
    "latt-move-scale": {
        "s0": 1,
        "s1": 1,
        "s2": 1,
    },
    "latt-scale": {
        "s0": 1,
        "s1": 1,
        "s2": 1,
    },
    "lattice-minimize": {
        "dirUpdateScheme": "L-BFGS",
        "linminMethod": "DirUpdateRecommended",
        "nIterations": 0,
        "history": 15,
        "knormThreshold": 0,
        "maxThreshold": False,
        "energyDiffThreshold": 1e-06,
        "nEnergyDiff": 2,
        "convergeAll": False,
        "alphaTstart": 1,
        "alphaTmin": 1e-10,
        "updateTestStepSize": True,
        "alphaTreduceFactor": 0.1,
        "alphaTincreaseFactor": 3,
        "nAlphaAdjustMax": 3,
        "wolfeEnergy": 0.0001,
        "wolfeGradient": 0.9,
        "abortOnFailedStep": False,
        "fdTest": False,
    },
    "lcao-params": {
        "nIter": -1,
        "Ediff": 1e-06,
        "smearingWidth": 0.001,
    },
    "pcm-variant": "GLSSA13",
    "perturb-minimize": {
        "nIterations": 0,
        "algorithm": "MINRES",
        "residualTol": 0.0001,
        "residualDiffThreshold": 0.0001,
        "CGBypass": False,
        "recomputeResidual": False,
    },
    "spintype": "no-spin",
    "subspace-rotation-factor": {
        "factor": 1,
        "adjust": True,
    },
    "symmetries": "automatic",
    "symmetry-threshold": 0.0001,
}


# JDFTx's units (core/Units.h), in which it echoes the fluid component parameters:
Angstrom = 1 / 0.5291772  # Angstrom in bohrs
meter = 1e10 * Angstrom  # meter in bohrs
liter = 1e-3 * pow(meter, 3)  # liter in cubic bohrs
mol = 6.0221367e23  # mole in number (i.e. Avogadro number)
Joule = 1 / 4.35974434e-18  # Joule in Hartrees
kg = 1.0 / 9.10938291e-31  # kilogram in electron masses
Newton = Joule / meter  # Newton in Hartree/bohr
Pascal = Newton / (meter * meter)  # Pascal in Hartree/bohr^3
KPascal = 1000 * Pascal  # KPa in Hartree/bohr^3
mmHg = 133.322387415 * Pascal  # mm Hg in Hartree/bohr^3
sec = np.sqrt((kg * meter) / Newton)  # second in inverse Hartrees
fs = sec * 1.0e-15  # femtosecond in inverse Hartrees


def antoinePvap(T: float, A: float, B: float, C: float) -> float:
    """Calculate vapor pressure using Antoine equation.

    Calculate vapor pressure using Antoine equation. Used for setting default Pvap of solvents.

    Parameters:
    ----------
    T: float
        Temperature in Kelvin.
    A: float
        Antoine coefficient A.
    B: float
        Antoine coefficient B.
    C: float
        Antoine coefficient C.

    Returns:
    -------
    float
        Vapor pressure in Hartree/bohr^3.
    """
    return KPascal * 10 ** (A - (B / (T + C)))


pureNbulk = {
    "H2O": 4.9383e-3,
    "CHCl3": 1.109e-3,
    "CCl4": 9.205e-4,
    "CH3CN": 1.709e-3,
    "DMC": 1.059e-3,
    "EC": 1.339e-3,
    "PC": 1.039e-3,
    "DMF": 1.153e-3,
    "THF": 1.100e-3,
    "EthylEther": 8.5e-4,
    "Isobutanol": 9.668e-4,
    "Chlorobenzene": 8.74e-4,
    "CarbonDisulfide": 1.48e-3,
    "DMSO": 1.256e-3,
    "CH2Cl2": 1.392e-3,
    "Ethanol": 1.528e-3,
    "Methanol": 2.203e-3,
    "Octanol": 5.646e-4,
    "Glyme": 8.586e-4,
    "EthyleneGlycol": 1.60e-3,
}


def get_default_fluid_params(name: str, T: float) -> dict:
    """Get JDFTx's default parameters for a fluid component (commands/fluid.cpp, fluid/FluidComponent.cpp).

    Parameters:
    ----------
    name: str
        Name of the fluid.
    T: float
        Temperature in Kelvin.

    Returns:
    -------
    dict
        Default fluid parameters, in the units JDFTx echoes them in (fluid-solvent, -cation and -anion subtags).
    """
    default_fluid_params = {
        "H2O": {
            "concentration": pureNbulk["H2O"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 78.4,
            "pMol": 0.92466,
            "epsInf": 1.77,
            "Pvap": antoinePvap(T, 7.31549, 1794.88, -34.764),
            "sigmaBulk": 4.62e-05,
            "Rvdw": 1.385 * Angstrom,
            "Res": 1.42,
            "poleEl": [
                {
                    "omega0": 15.0,
                    "gamma0": 7.0,
                    "A0": 1.0,
                },
            ],
        },
        "CHCl3": {
            "concentration": pureNbulk["CHCl3"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 4.8069,
            "pMol": 0.49091,
            "epsInf": 2.09,
            "Pvap": antoinePvap(T, 5.96288, 1106.94, -54.598),
            "sigmaBulk": 1.71e-5,
            "Rvdw": 2.53 * Angstrom,
            "Res": 2.22,
        },
        "CCl4": {
            "concentration": pureNbulk["CCl4"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 2.238,
            "pMol": 0.0,
            "epsInf": 2.13,
            "Pvap": antoinePvap(T, 6.10445, 1265.63, -41.002),
            "sigmaBulk": 1.68e-5,
            "Rvdw": 2.69 * Angstrom,
            "Res": 1.90,
        },
        "CH3CN": {
            "concentration": pureNbulk["CH3CN"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 38.8,
            "pMol": 1.89,
            "epsInf": 1.81,
            "Pvap": antoinePvap(T, 6.52111, 1492.375, -24.208),
            "sigmaBulk": 1.88e-5,
            "Rvdw": 2.12 * Angstrom,
            "Res": 2.6,
        },
        "CH2Cl2": {
            "concentration": pureNbulk["CH2Cl2"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 9.08,
            "pMol": 0.89,
            "epsInf": 1.424,
            "sigmaBulk": 1.70e-5,
        },
        "Ethanol": {
            "concentration": pureNbulk["Ethanol"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 24.3,
            "pMol": 0.76,
            "epsInf": 1.361,
            "sigmaBulk": 1.44e-5,
        },
        "Methanol": {
            "concentration": pureNbulk["Methanol"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 32.66,
            "pMol": 0.791,
            "epsInf": 1.328,
            "sigmaBulk": 1.445e-5,
        },
        "Octanol": {
            "concentration": pureNbulk["Octanol"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 10.30,
            "pMol": 0.661,
            "epsInf": 2.036,
            "Pvap": antoinePvap(T, 8.47682, 2603.359, -48.799),
            "sigmaBulk": 1.766e-5,
            "Rvdw": 3.348 * Angstrom,
        },
        "DMC": {
            "concentration": pureNbulk["DMC"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 3.1,
            "pMol": 0.16,
            "epsInf": 1.87,
            "Pvap": 18 * mmHg,
            "sigmaBulk": 2.05e-5,
        },
        "EC": {
            "concentration": pureNbulk["EC"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 90.5,
            "pMol": 2.88,
            "epsInf": 2.00,
            "Pvap": antoinePvap(T, 6.05764, 1705.267, -102.261),
            "sigmaBulk": 3.51e-5,
        },
        "PC": {
            "concentration": pureNbulk["PC"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 64.0,
            "pMol": 2.95,
            "epsInf": 2.02,
            "Pvap": antoinePvap(T, 6.20181, 1788.900, -88.715),
            "sigmaBulk": 2.88e-5,
        },
        "DMF": {
            "concentration": pureNbulk["DMF"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 38.0,
            "pMol": 2.19,
            "epsInf": 2.05,
            "Pvap": antoinePvap(T, 6.05286, 1400.86, -76.716),
            "sigmaBulk": 2.26e-5,
        },
        "THF": {
            "concentration": pureNbulk["THF"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 7.6,
            "pMol": 0.90,
            "epsInf": 1.98,
            "Pvap": antoinePvap(T, 6.12142, 1203.11, -46.795),
            "sigmaBulk": 1.78e-5,
        },
        "DMSO": {
            "concentration": pureNbulk["DMSO"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 48.0,
            "pMol": 1.56,
            "epsInf": 2.19,
            "sigmaBulk": 2.80e-5,
            "Pvap": antoinePvap(T, 7.23039, 2239.161, -29.215),
            "Rvdw": 2.378 * Angstrom,
        },
        "EthylEther": {
            "concentration": pureNbulk["EthylEther"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 4.34,
            "pMol": 0.487,
            "epsInf": 1.82,
            "Pvap": antoinePvap(T, 6.96559, 1071.54, 227.774),
            "sigmaBulk": 1.092e-5,
        },
        "Chlorobenzene": {
            "concentration": pureNbulk["Chlorobenzene"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 5.69,
            "pMol": 0.72,
            "epsInf": 2.32,
            "Pvap": antoinePvap(T, 4.11083, 1435.675, -55.124),
            "sigmaBulk": 2.1e-5,
        },
        "Isobutanol": {
            "concentration": pureNbulk["Isobutanol"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 17.93,
            "pMol": 0.646,
            "epsInf": 1.949,
            "sigmaBulk": 1.445e-5,
        },
        "CarbonDisulfide": {
            "concentration": pureNbulk["CarbonDisulfide"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 2.641,
            "epsInf": 2.641,
            "pMol": 0.0,
        },
        "Glyme": {
            "concentration": pureNbulk["Glyme"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 7.20,
            "epsInf": 1.90,
            "pMol": 0.0,
        },
        "EthyleneGlycol": {
            "concentration": pureNbulk["EthyleneGlycol"] / (mol / liter),  # in mol/L
            "functional": "ScalarEOS",
            "epsBulk": 41.4,
            "epsInf": 1.43,
            "pMol": 0.0,
        },
        "Na+": {
            "functional": "MeanFieldLJ",
            "concentration": 1.0,
            "Rvdw": 1.16 * Angstrom,
        },
        "K+": {
            "functional": "MeanFieldLJ",
            "concentration": 1.0,
            "Rvdw": 1.51 * Angstrom,
        },
        "Cl-": {
            "functional": "MeanFieldLJ",
            "concentration": 1.0,
            "Rvdw": 1.67 * Angstrom,
        },
        "F-": {
            "functional": "MeanFieldLJ",
            "concentration": 1.0,
            "Rvdw": 1.19 * Angstrom,
        },
        "ClO4-": {
            "functional": "MeanFieldLJ",
            "concentration": 1.0,
            "Rvdw": 2.41 * Angstrom,
        },
    }
    empty_fluid_component = {
        "epsBulk": 1.0,
        "epsInf": 1.0,
        "epsLJ": 0.0,
        "Nnorm": 0,
        "pMol": 0.0,
        "Pvap": 0.0,
        "quad_nAlpha": 0,
        "quad_nBeta": 0,
        "quad_nGamma": 0,
        "representation": "MuEps",
        "Res": 0.0,
        "Rvdw": 0.0,
        "s2quadType": "7design24",
        "sigmaBulk": 0.0,
        "tauNuc": float(int(8.3e3 * fs)),
        "translation": "LinearSpline",
    }
    return {"name": name} | empty_fluid_component | default_fluid_params.get(name, {})
