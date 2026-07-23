"""
DFI / DCI calculator
====================

Description
------------
Calculate Dynamic Flexibility Index, DFI, or Dynamic Coupling Index, DCI,
from a single PDB structure or a provided covariance / inverse Hessian matrix.

This script is modified based on:

Original repository:
https://github.com/SBOZKAN/DFI-DCI

Original author:
SBOZKAN


Usage
-----
DFI only:
    python calc_dfi_dci.py --pdb 1btl.pdb

DCI only:
    python calc_dfi_dci.py --pdb 1btl.pdb --dci A96

DCI with multiple residues:
    python calc_dfi_dci.py --pdb 1btl.pdb --dci A96 A120 B35

With chain selection:
    python calc_dfi_dci.py --pdb 1btl.pdb --chain A
    python calc_dfi_dci.py --pdb 1btl.pdb --chain A --dci A96

With external covariance / inverse Hessian:
    python calc_dfi_dci.py --pdb 1btl.pdb --hess invhessian.txt
    python calc_dfi_dci.py --pdb 1btl.pdb --hess invhessian.txt --dci A96

Output
------
DFI mode:
    prefix_dfi.csv

DCI mode:
    prefix_dci_A96.csv
    prefix_dci_A96_A120.csv
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import linalg as LA
from scipy import stats


if __name__ == "__main__" and len(sys.argv) < 2:
    print(__doc__)
    sys.exit(0)


class ATOM:
    def __init__(self, record, atom_index, atom_name, alc, res_name, chainID,
                 res_index, insert_code, x, y, z, occupancy,
                 temp_factor, atom_type):
        self.record = str(record)
        self.atom_index = int(atom_index)
        self.atom_name = str(atom_name)
        self.alc = str(alc)
        self.res_name = str(res_name)
        self.chainID = str(chainID)
        self.res_index = str(res_index).strip(" ")
        self.insert_code = str(insert_code)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

        self.occupancy = 1.0

        if temp_factor == "    ":
            self.temp_factor = 0.0
        else:
            self.temp_factor = float(temp_factor)

        self.atom_type = str(atom_type)


def pdb_reader(filename, CAonly=False, noalc=True, chainA=False,
               chain_name="A", Verbose=False):
    """
    Read ATOM entries from a PDB file.
    If CAonly=True, only C-alpha atoms are read.
    If chainA=True, only the specified chain is read.
    """

    ATOMS = []
    readatoms = 0

    with open(filename) as pdb:
        for line in pdb:
            if line.startswith("ENDMDL"):
                if Verbose:
                    print("MULTIPLE MODELS...USING MODEL 1")
                return ATOMS

            if not line.startswith("ATOM"):
                continue

            atom_name = line[13:16]

            if CAonly and atom_name != "CA ":
                continue

            alc = line[16]
            if noalc and not (alc == " " or alc == "A"):
                continue

            chainID = line[21]
            if chainA and chainID != chain_name:
                continue

            ATOMS.append(
                ATOM(
                    line[:6],
                    line[7:11],
                    line[13:16],
                    line[16],
                    line[17:20],
                    line[21],
                    line[22:27],
                    line[26],
                    line[30:38],
                    line[38:46],
                    line[46:54],
                    line[55:60],
                    line[60:66],
                    line[77] if len(line) > 77 else ""
                )
            )

            readatoms += 1

    print(f"Read {readatoms} atoms from {filename}")
    return ATOMS


def getcoords(ATOMS):
    """
    Return x, y, z coordinate arrays from C-alpha ATOM objects.
    """

    x = np.array([atom.x for atom in ATOMS if atom.atom_name == "CA "], dtype=float)
    y = np.array([atom.y for atom in ATOMS if atom.atom_name == "CA "], dtype=float)
    z = np.array([atom.z for atom in ATOMS if atom.atom_name == "CA "], dtype=float)

    return x, y, z


def calchessian(resnum, x, y, z, gamma=100.0, cutoff=None, Verbose=False):
    """
    Calculate the 3N x 3N Hessian matrix from C-alpha coordinates.

    Parameters
    ----------
    resnum : int
        Number of residues.
    x, y, z : numpy arrays
        C-alpha coordinates.
    gamma : float
        Spring constant.
    cutoff : float or None
        If provided, residue pairs farther than cutoff are ignored.
    """

    numresthree = 3 * resnum
    hess = np.zeros((numresthree, numresthree), dtype=float)

    if Verbose:
        print("i,j,x1,y1,z1,x2,y2,z2,x_ij,y_ij,z_ij,r2,k,gamma,cutoff")

    for i in range(resnum):
        for j in range(resnum):
            if i == j:
                continue

            x_i, y_i, z_i = x[i], y[i], z[i]
            x_j, y_j, z_j = x[j], y[j], z[j]

            x_ij = x_i - x_j
            y_ij = y_i - y_j
            z_ij = z_i - z_j

            r2 = x_ij * x_ij + y_ij * y_ij + z_ij * z_ij

            if r2 <= 1e-12:
                continue

            if cutoff is not None and np.sqrt(r2) > cutoff:
                sprngcnst = 0.0
            else:
                sprngcnst = (gamma ** 3) / (r2 ** 3)

            if Verbose:
                print(
                    ",".join(
                        np.array(
                            [
                                i, j,
                                x_i, y_i, z_i,
                                x_j, y_j, z_j,
                                x_ij, y_ij, z_ij,
                                r2, sprngcnst, gamma, cutoff
                            ],
                            dtype=str
                        )
                    )
                )

            # Hii block
            hess[3 * i, 3 * i] += sprngcnst * (x_ij * x_ij / r2)
            hess[3 * i + 1, 3 * i + 1] += sprngcnst * (y_ij * y_ij / r2)
            hess[3 * i + 2, 3 * i + 2] += sprngcnst * (z_ij * z_ij / r2)

            hess[3 * i, 3 * i + 1] += sprngcnst * (x_ij * y_ij / r2)
            hess[3 * i, 3 * i + 2] += sprngcnst * (x_ij * z_ij / r2)
            hess[3 * i + 1, 3 * i] += sprngcnst * (y_ij * x_ij / r2)

            hess[3 * i + 1, 3 * i + 2] += sprngcnst * (y_ij * z_ij / r2)
            hess[3 * i + 2, 3 * i] += sprngcnst * (z_ij * x_ij / r2)
            hess[3 * i + 2, 3 * i + 1] += sprngcnst * (z_ij * y_ij / r2)

            # Hij block
            hess[3 * i, 3 * j] -= sprngcnst * (x_ij * x_ij / r2)
            hess[3 * i + 1, 3 * j + 1] -= sprngcnst * (y_ij * y_ij / r2)
            hess[3 * i + 2, 3 * j + 2] -= sprngcnst * (z_ij * z_ij / r2)

            hess[3 * i, 3 * j + 1] -= sprngcnst * (x_ij * y_ij / r2)
            hess[3 * i, 3 * j + 2] -= sprngcnst * (x_ij * z_ij / r2)
            hess[3 * i + 1, 3 * j] -= sprngcnst * (y_ij * x_ij / r2)

            hess[3 * i + 1, 3 * j + 2] -= sprngcnst * (y_ij * z_ij / r2)
            hess[3 * i + 2, 3 * j] -= sprngcnst * (z_ij * x_ij / r2)
            hess[3 * i + 2, 3 * j + 1] -= sprngcnst * (z_ij * y_ij / r2)

    return hess


def flatandwrite(matrix, outfile):
    """
    Flatten a matrix and write it to a file.
    """

    np.savetxt(outfile, matrix.flatten())


def calc_covariance(numres, x, y, z, invhessfile=None, Verbose=False):
    """
    Calculate covariance / inverse Hessian matrix from coordinates.
    """

    gamma = 100.0

    hess = calchessian(
        numres,
        x,
        y,
        z,
        gamma=gamma,
        cutoff=None,
        Verbose=Verbose
    )

    U, w, Vt = LA.svd(hess, full_matrices=False)

    tol = 1e-6
    singular = w < tol

    invw = np.zeros_like(w)
    invw[~singular] = 1.0 / w[~singular]

    invHrs = np.dot(np.dot(U, np.diag(invw)), Vt)

    if Verbose and invhessfile is not None:
        flatandwrite(invHrs, invhessfile)

    n_singular = np.sum(singular)

    if n_singular != 6:
        print(
            f"WARNING: Expected 6 near-zero modes, but found {n_singular}. "
            "This may happen for disconnected chains, missing residues, or unusual structures."
        )

    return invHrs


def pctrank(dfi, inverse=False):
    """
    Calculate percentile rank of DFI or DCI values.
    """

    if type(dfi).__module__ != "numpy":
        raise ValueError("Input needs to be a numpy array")

    dfiperc = []
    lendfi = float(len(dfi))

    for m in dfi:
        if inverse:
            amt = np.sum(dfi >= m)
        else:
            amt = np.sum(dfi <= m)

        dfiperc.append(amt / lendfi)

    return np.array(dfiperc, dtype=float)


def dfianal(fname, Array=False):
    """
    Calculate raw, relative, percentile, and z-score values.
    """

    if not Array:
        with open(fname, "r") as infile:
            dfi = np.array([x.strip("\n") for x in infile], dtype=float)
    else:
        dfi = fname

    dfirel = dfi / np.mean(dfi)
    dfizscore = stats.zscore(dfi)
    dfiperc = pctrank(dfi)

    return dfi, dfirel, dfiperc, dfizscore


def calcperturbMat(invHrs, direct, resnum, Normalize=True):
    """
    Calculate perturbation matrix used for DFI / DCI calculation.

    perturbMat[i, j] = response of residue i when residue j is perturbed.
    """

    perturbMat = np.zeros((resnum, resnum), dtype=float)

    for k in range(len(direct)):
        peturbDir = direct[k, :]

        for j in range(int(resnum)):
            delforce = np.zeros(3 * resnum, dtype=float)
            delforce[3 * j:3 * j + 3] = peturbDir

            delXperbVex = np.dot(invHrs, delforce)
            delXperbMat = delXperbVex.reshape((resnum, 3))

            delRperbVec = np.sqrt(
                np.sum(delXperbMat * delXperbMat, axis=1)
            )

            perturbMat[:, j] += delRperbVec[:]

    perturbMat /= len(direct)

    if Normalize:
        total = np.sum(perturbMat)
        if total == 0:
            raise ValueError("Perturbation matrix sum is zero. Cannot normalize.")
        nrmlperturbMat = perturbMat / total
    else:
        print("WARNING: The perturbation matrix is not normalized.")
        nrmlperturbMat = perturbMat

    return nrmlperturbMat


def chainresmap(ATOMS, Verbose=False):
    """
    Return a dictionary mapping chain + residue number to atom index.

    Example:
        A96 -> index
        B120 -> index
    """

    table = {}

    for i in range(len(ATOMS)):
        if ATOMS[i].chainID == " ":
            entry = ATOMS[i].res_index
        else:
            entry = ATOMS[i].chainID + ATOMS[i].res_index

        table[entry] = i

    if Verbose:
        print(table)

    return table


def dciresf(ls_chain, table):
    """
    Convert DCI residue names such as A96, B120 to residue indices.
    """

    ls_ind = []

    for res in ls_chain:
        if res in table:
            ls_ind.append(table[res])
        else:
            print(f"WARNING: Can't find {res}")

    return np.array(ls_ind, dtype=int)


def dcires_cords(dcires, x, y, z):
    """
    Pull out DCI residue coordinates.
    """

    return np.column_stack((x[dcires], y[dcires], z[dcires]))


def rdist(r, fr):
    """
    Calculate distance from residue r to functional DCI residues fr.
    """

    r_ij = fr - r
    rr = r_ij * r_ij

    return np.sqrt(rr.sum(axis=1))


def output_dfi_df(ATOMS, dfi, reldfi, pctdfi, zscoredfi,
                  outfile=None, writetofile=False):
    """
    Output DFI result only.
    """

    mapres = {
        "ALA": "A",
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "PRO": "P",
        "ARG": "R",
        "GLN": "Q",
        "ASN": "N",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V"
    }

    df = pd.DataFrame()

    df["ChainID"] = [atom.chainID for atom in ATOMS]
    df["ResID"] = [atom.res_index.strip(" ") for atom in ATOMS]
    df["Res"] = [atom.res_name for atom in ATOMS]
    df["R"] = df["Res"].map(mapres)

    df["dfi"] = dfi
    df["reldfi"] = reldfi
    df["pctdfi"] = pctdfi
    df["zscoredfi"] = zscoredfi

    if writetofile:
        df.to_csv(outfile, index=False)
        print(f"Wrote out to {outfile}")

    return df


def output_dci_df(ATOMS, dci, reldci, pctdci, zscoredci,
                  ls_ravg=None, ls_rmin=None,
                  outfile=None, writetofile=False):
    """
    Output DCI result only.
    """

    mapres = {
        "ALA": "A",
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "PRO": "P",
        "ARG": "R",
        "GLN": "Q",
        "ASN": "N",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V"
    }

    df = pd.DataFrame()

    df["ChainID"] = [atom.chainID for atom in ATOMS]
    df["ResID"] = [atom.res_index.strip(" ") for atom in ATOMS]
    df["Res"] = [atom.res_name for atom in ATOMS]
    df["R"] = df["Res"].map(mapres)

    df["dci"] = dci
    df["reldci"] = reldci
    df["pctdci"] = pctdci
    df["zscoredci"] = zscoredci

    if ls_ravg is not None:
        df["ravg"] = ls_ravg

    if ls_rmin is not None:
        df["rmin"] = ls_rmin

    if ls_rmin is not None:
        mask = (df["rmin"] > 8.0) & (df["pctdci"] > 0.75)
        df["A"] = mask.map(lambda x: "A" if x else "NotA")

    if writetofile:
        df.to_csv(outfile, index=False)
        print(f"Wrote out to {outfile}")

    return df


def make_dci_suffix(ls_reschain):
    """
    Make DCI output filename suffix from input residue names.

    Example:
        ['A96'] -> A96
        ['A96', 'A120'] -> A96_A120
    """

    if len(ls_reschain) == 0:
        return ""

    # 保持输入顺序，同时去掉重复项
    seen = set()
    unique_res = []

    for res in list(ls_reschain):
        if res not in seen:
            unique_res.append(str(res))
            seen.add(res)

    return "_".join(unique_res)


def prepare_structure_and_covariance(pdbfile, covar=None, chain_name=None,
                                     Verbose=False):
    """
    Shared preparation step for DFI or DCI.
    """

    if chain_name is not None:
        ATOMS = pdb_reader(
            pdbfile,
            CAonly=True,
            noalc=True,
            chainA=True,
            chain_name=chain_name,
            Verbose=False
        )
    else:
        ATOMS = pdb_reader(
            pdbfile,
            CAonly=True,
            noalc=True,
            chainA=False,
            chain_name="A",
            Verbose=False
        )

    x, y, z = getcoords(ATOMS)
    numres = len(ATOMS)

    if numres == 0:
        raise ValueError("No C-alpha atoms found. Please check PDB file or chain ID.")

    if covar is None:
        invHrs = calc_covariance(
            numres,
            x,
            y,
            z,
            invhessfile=None,
            Verbose=Verbose
        )
    else:
        invHrs = np.loadtxt(covar)

        expected_shape = (3 * numres, 3 * numres)
        if invHrs.shape != expected_shape:
            raise ValueError(
                f"Loaded Hessian / covariance matrix shape is {invHrs.shape}, "
                f"but expected {expected_shape} for {numres} residues."
            )

    return ATOMS, x, y, z, numres, invHrs


def build_default_directions():
    """
    Build the seven perturbation directions used in the original DFI script.
    """

    directions = np.vstack(
        (
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        )
    )

    normL = np.linalg.norm(directions, axis=1)
    direct = directions / normL[:, None]

    return direct


def run_dfi(pdbfile, pdbid=None, covar=None, chain_name=None,
            Verbose=False, writetofile=False):
    """
    Calculate DFI only.

    Output:
        prefix_dfi.csv
    """

    if pdbid is None:
        pdbid = os.path.splitext(os.path.basename(pdbfile))[0]

    prefix = os.path.splitext(os.path.basename(pdbfile))[0]
    dfi_file = prefix + "_dfi.csv"

    ATOMS, x, y, z, numres, invHrs = prepare_structure_and_covariance(
        pdbfile,
        covar=covar,
        chain_name=chain_name,
        Verbose=Verbose
    )

    direct = build_default_directions()
    nrmlperturbMat = calcperturbMat(invHrs, direct, numres)

    dfi_raw = np.sum(nrmlperturbMat, axis=1)

    dfi, reldfi, pctdfi, zscoredfi = dfianal(
        dfi_raw,
        Array=True
    )

    df_dfi = output_dfi_df(
        ATOMS,
        dfi=dfi,
        reldfi=reldfi,
        pctdfi=pctdfi,
        zscoredfi=zscoredfi,
        outfile=dfi_file,
        writetofile=writetofile
    )

    return df_dfi


def run_dci(pdbfile, dci_residues, pdbid=None, covar=None,
            chain_name=None, Verbose=False, writetofile=False):
    """
    Calculate DCI only.

    Output:
        prefix_dci_A96.csv
        prefix_dci_A96_A120.csv
    """

    if pdbid is None:
        pdbid = os.path.splitext(os.path.basename(pdbfile))[0]

    if len(dci_residues) == 0:
        raise ValueError("DCI calculation requires --dci, e.g. --dci A96")

    prefix = os.path.splitext(os.path.basename(pdbfile))[0]
    dci_suffix = make_dci_suffix(dci_residues)
    dci_file = prefix + "_dci_" + dci_suffix + ".csv"

    ATOMS, x, y, z, numres, invHrs = prepare_structure_and_covariance(
        pdbfile,
        covar=covar,
        chain_name=chain_name,
        Verbose=Verbose
    )

    direct = build_default_directions()
    nrmlperturbMat = calcperturbMat(invHrs, direct, numres)

    # find the DCI residues
    dcires = np.sort(
        dciresf(
            dci_residues,
            chainresmap(ATOMS)
        )
    )

    if len(dcires) == 0:
        raise ValueError("No valid DCI residues found. Please check residue names such as A96.")

    # DCI calculation
    dcitop = np.sum(nrmlperturbMat[:, dcires], axis=1) / len(dcires)
    dcibot = np.sum(nrmlperturbMat, axis=1) / len(nrmlperturbMat)

    dci_raw = dcitop / dcibot

    dci, reldci, pctdci, zscoredci = dfianal(
        dci_raw,
        Array=True
    )

    rlist = np.column_stack((x, y, z))
    fr = dcires_cords(dcires, x, y, z)

    ls_ravg = np.array([rdist(r, fr).mean() for r in rlist])
    ls_rmin = np.array([rdist(r, fr).min() for r in rlist])

    df_dci = output_dci_df(
        ATOMS,
        dci=dci,
        reldci=reldci,
        pctdci=pctdci,
        zscoredci=zscoredci,
        ls_ravg=ls_ravg,
        ls_rmin=ls_rmin,
        outfile=dci_file,
        writetofile=writetofile
    )

    return df_dci


def CLdict(argv):
    """
    Return a dictionary of command line options from sys.argv.
    """

    comline_arg = {}

    for i, s in enumerate(argv):
        if s == "--pdb":
            if i + 1 >= len(argv):
                print("ERROR: --pdb requires a file name.")
                sys.exit(1)

            comline_arg[s] = argv[i + 1]

            if not os.path.isfile(argv[i + 1]):
                print(f"File {argv[i + 1]} not found.")
                sys.exit(1)

        elif s == "--hess":
            if i + 1 >= len(argv):
                print("ERROR: --hess requires a file name.")
                sys.exit(1)

            comline_arg[s] = argv[i + 1]

            if not os.path.isfile(argv[i + 1]):
                print(f"File {argv[i + 1]} not found.")
                sys.exit(1)

        elif s == "--chain":
            if i + 1 >= len(argv):
                print("ERROR: --chain requires a chain ID.")
                sys.exit(1)

            comline_arg[s] = argv[i + 1]

        elif s == "--dci":
            resvals = []

            for res in argv[i + 1:]:
                if res.startswith("--"):
                    break
                else:
                    resvals.append(res)

            if len(resvals) == 0:
                print("ERROR: --dci requires at least one residue, e.g. --dci A96")
                sys.exit(1)

            comline_arg[s] = np.array(resvals)

        elif s == "--help":
            print(__doc__)
            sys.exit(0)

    if "--pdb" not in argv:
        print("ERROR: No --pdb provided.")
        print(__doc__)
        sys.exit(1)

    return comline_arg


def parseCommandLine(argv):
    """
    Parse command line input.
    """

    comlinargs = CLdict(argv)

    pdbfile = comlinargs["--pdb"]
    pdbid = os.path.splitext(os.path.basename(pdbfile))[0]

    mdhess = comlinargs.get("--hess", None)
    ls_reschain = comlinargs.get("--dci", [])
    chain_name = comlinargs.get("--chain", None)

    if len(ls_reschain) > 0:
        mode = "dci"
    else:
        mode = "dfi"

    return pdbfile, pdbid, mdhess, ls_reschain, chain_name, mode


if __name__ == "__main__":

    pdbfile, pdbid, covar, ls_reschain, chain_name, mode = parseCommandLine(
        sys.argv
    )

    print(f"Processing {pdbfile}")

    if mode == "dfi":
        print("Mode: DFI only")

        run_dfi(
            pdbfile,
            pdbid=pdbid,
            covar=covar,
            chain_name=chain_name,
            Verbose=False,
            writetofile=True
        )

    elif mode == "dci":
        print("Mode: DCI only")
        print("DCI residues:", " ".join(ls_reschain))

        run_dci(
            pdbfile,
            dci_residues=ls_reschain,
            pdbid=pdbid,
            covar=covar,
            chain_name=chain_name,
            Verbose=False,
            writetofile=True
        )
