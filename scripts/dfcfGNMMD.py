import os
import argparse
import numpy as np

from Bio.PDB import PDBParser
from scipy.stats import pearsonr

# ============================================================
# dcfGNMMD Transfer Entropy Calculation
# ============================================================
#
# Origin of the implementation:
#
# This Python implementation was rewritten from the original
# MATLAB code provided by Dr. Zhongjie Han.
#
# Reference:
#
# https://doi.org/10.1021/acs.jpclett.3c00366
#
# The original methodology and MATLAB implementation were
# developed by Dr. Zhongjie Han.
#
# This Python version reorganizes the original workflow into
# a one-click calculation program with automatic parameter
# optimization and result generation.
#
# When using this program or results obtained from this program,
# please cite the original publication and acknowledge the
# original MATLAB implementation provided by Dr. Zhongjie Han.
#
# ============================================================

#
# This program performs the following steps:
#
# 1. Read the PDB structure.
# 2. Extract all C-alpha atoms.
# 3. Read the 3N x 3N MD covariance matrix.
# 4. Convert the Cartesian covariance matrix into an
#    N x N residue-level covariance matrix.
# 5. Calculate MD mean-square fluctuations.
# 6. Perform automatic optimization of rc and s.
# 7. Select the parameter combination with the highest
#    average Pearson correlation coefficient.
# 8. Construct the final dfcfGNM network using the optimal
#    parameters.
# 9. Calculate transfer entropy.
# 10. Calculate net transfer entropy.
# 11. Normalize the net transfer entropy matrix.
# 12. Calculate normalized net information flow.
# 13. Save all results automatically.
#
#
# usage:
#
#     python dcfGNMMD.py -- casp_dry.pdb --covar casp_covar_1.dat
#
# The optimization range is fixed automatically:
#
#     rc = 6 ~ 15
#     s  = 0 ~ 6
#
# These parameters do not need to be specified manually.
#
# ============================================================


# ============================================================
# Fixed calculation parameters
# ============================================================

# Distance cutoff optimization range.
RC_MIN = 6
RC_MAX = 15

# s parameter optimization range.
S_MIN = 0
S_MAX = 6

# Eigenvalue tolerance.
EIG_TOL = 1.0e-10

# Singular-value cutoff used in the transfer entropy
# mode decomposition.
SVD_CUTOFF = 0.01


# ============================================================
# Command-line arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "dfcfGNMMD covariance fitting and "
            "transfer entropy calculation."
        )
    )

    # Input PDB file.
    parser.add_argument(
        "--pdb",
        default="casp3_dry.pdb",
        help="Input PDB file. Default: casp3_dry.pdb"
    )

    # Input MD covariance matrix.
    parser.add_argument(
        "--covar",
        default="casp_covar_1.dat",
        help=(
            "Input 3N x 3N covariance matrix. "
            "Default: casp_covar_1.dat"
        )
    )

    # Transfer entropy time parameter.
    parser.add_argument(
        "--tau",
        type=float,
        default=5.0,
        help="Transfer entropy time parameter. Default: 5.0"
    )

    # Optional output directory.
    parser.add_argument(
        "--outdir",
        default=None,
        help=(
            "Output directory. "
            "If omitted, it is generated automatically."
        )
    )

    return parser.parse_args()


# ============================================================
# Generate output names from covariance filename
# ============================================================

def generate_output_names(covar_file):

    # Extract the covariance filename.
    covar_name = os.path.basename(
        covar_file
    )

    # Remove the filename extension.
    covar_stem = os.path.splitext(
        covar_name
    )[0]

    #
    # Example:
    #
    # casp_covar_1.dat
    #
    # becomes:
    #
    # casp_covar_1
    #

    # Try to extract the numerical suffix.
    if "_" in covar_stem:

        base_name, suffix = covar_stem.rsplit(
            "_",
            1
        )

    else:

        base_name = covar_stem
        suffix = ""

    # Only treat the last part as a suffix if it is numeric.
    if suffix.isdigit():

        output_suffix = "_" + suffix

        output_dir = (
            base_name
            + "_fcfgnmmd"
            + output_suffix
        )

    else:

        output_suffix = ""

        output_dir = (
            covar_stem
            + "_fcfgnmmd"
        )

    return (
        covar_stem,
        output_suffix,
        output_dir
    )


# ============================================================
# Read C-alpha coordinates from PDB
# ============================================================

def read_ca_coordinates(pdb_file):

    # Check whether the PDB file exists.
    if not os.path.isfile(
        pdb_file
    ):

        raise FileNotFoundError(
            "\nPDB file not found:\n"
            f"{os.path.abspath(pdb_file)}"
        )

    # Initialize the PDB parser.
    parser = PDBParser(
        QUIET=True
    )

    # Read the PDB structure.
    structure = parser.get_structure(
        "protein",
        pdb_file
    )

    # Use the first model.
    model = next(
        structure.get_models()
    )

    posall = []
    resall = []

    # Loop over all chains.
    for chain in model:

        # Loop over all residues.
        for residue in chain:

            # Keep only C-alpha atoms.
            if "CA" in residue:

                ca = residue["CA"]

                # Store C-alpha Cartesian coordinates.
                posall.append(
                    ca.get_coord()
                )

                # Store residue sequence number.
                resall.append(
                    residue.get_id()[1]
                )

    # Convert coordinates to NumPy array.
    posall = np.asarray(
        posall,
        dtype=float
    )

    # Convert residue numbers to NumPy array.
    resall = np.asarray(
        resall,
        dtype=int
    )

    return posall, resall


# ============================================================
# Convert 3N x 3N covariance matrix to N x N
# residue-level covariance matrix
# ============================================================

def convert_covariance(
    md_cof,
    N
):

    # Initialize the residue-level covariance matrix.
    md_cof_residue = np.zeros(
        (N, N),
        dtype=float
    )

    # Loop over all residue pairs.
    for i in range(N):

        i_start = i * 3
        i_end = (i + 1) * 3

        for j in range(N):

            j_start = j * 3
            j_end = (j + 1) * 3

            # Extract the corresponding 3 x 3 block.
            block = md_cof[
                i_start:i_end,
                j_start:j_end
            ]

            # Use the trace of the 3 x 3 block.
            md_cof_residue[i, j] = np.trace(
                block
            )

    return md_cof_residue


# ============================================================
# Main program
# ============================================================

def main():

    # Read command-line arguments.
    args = parse_args()


    # ========================================================
    # Generate output names
    # ========================================================

    (
        covar_stem,
        output_suffix,
        automatic_output_dir
    ) = generate_output_names(
        args.covar
    )


    # Use the automatically generated output directory
    # unless the user explicitly provides one.
    if args.outdir is None:

        output_dir = automatic_output_dir

    else:

        output_dir = args.outdir


    # Create the output directory.
    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # ========================================================
    # Print initial information
    # ========================================================

    print()
    print("=" * 70)
    print(
        "       dfcfGNMMD Transfer Entropy"
    )
    print(
        "             One-click Calculation"
    )
    print("=" * 70)

    print()

    print(
        f"PDB file        : {args.pdb}"
    )

    print(
        f"Covariance file : {args.covar}"
    )

    print(
        f"Tau             : {args.tau}"
    )

    print(
        f"Output directory: {output_dir}"
    )

    print()

    print(
        "Automatic parameter optimization:"
    )

    print(
        f"  rc = {RC_MIN} ~ {RC_MAX}"
    )

    print(
        f"  s  = {S_MIN} ~ {S_MAX}"
    )


    # ========================================================
    # Step 1: Read PDB
    # ========================================================

    print()
    print("=" * 70)
    print(
        "[1] Reading PDB structure"
    )
    print("=" * 70)


    posall, resall = read_ca_coordinates(
        args.pdb
    )


    # Number of C-alpha atoms.
    n = len(posall)


    if n == 0:

        raise ValueError(
            "No C-alpha atoms were found in the PDB file."
        )


    print(
        f"Number of C-alpha atoms = {n}"
    )


    # ========================================================
    # Step 2: Read MD covariance matrix
    # ========================================================

    print()
    print("=" * 70)
    print(
        "[2] Reading MD covariance matrix"
    )
    print("=" * 70)


    # Check whether the covariance file exists.
    if not os.path.isfile(
        args.covar
    ):

        raise FileNotFoundError(
            "\nCovariance file not found:\n"
            f"{os.path.abspath(args.covar)}"
        )


    # Read covariance matrix.
    md_cof_3n = np.loadtxt(
        args.covar
    )


    print(
        f"Covariance shape = {md_cof_3n.shape}"
    )


    # ========================================================
    # Step 3: Check covariance matrix
    # ========================================================

    if md_cof_3n.ndim != 2:

        raise ValueError(
            "Covariance matrix must be two-dimensional."
        )


    if (
        md_cof_3n.shape[0]
        != md_cof_3n.shape[1]
    ):

        raise ValueError(
            "Covariance matrix must be square."
        )


    if (
        md_cof_3n.shape[0] % 3
        != 0
    ):

        raise ValueError(
            "Covariance matrix size must be divisible by 3."
        )


    if not np.all(
        np.isfinite(md_cof_3n)
    ):

        raise ValueError(
            "Covariance matrix contains NaN or Inf."
        )


    # Calculate the number of residues.
    N = (
        md_cof_3n.shape[0]
        // 3
    )


    print(
        f"Number of residues from covariance = {N}"
    )


    # ========================================================
    # Step 4: Check PDB and covariance consistency
    # ========================================================

    if N != n:

        raise ValueError(
            "\n"
            "============================================\n"
            "ERROR: PDB and covariance size mismatch!\n"
            "============================================\n"
            f"PDB C-alpha atoms   = {n}\n"
            f"Covariance residues = {N}\n"
            "\n"
            "Please check whether the PDB and covariance "
            "matrix correspond to the same system."
        )


    # ========================================================
    # Step 5: Convert covariance matrix
    # ========================================================

    print()
    print("=" * 70)
    print(
        "[3] Converting 3N x 3N covariance"
    )
    print("=" * 70)


    md_cof_residue = convert_covariance(
        md_cof_3n,
        N
    )


    # Save residue-level covariance.
    residue_covariance_file = os.path.join(
        output_dir,
        f"casp_covariance{output_suffix}.dat"
    )


    np.savetxt(
        residue_covariance_file,
        md_cof_residue,
        delimiter="\t",
        fmt="%.6f"
    )


    print(
        "Residue-level covariance saved:"
    )

    print(
        f"  {residue_covariance_file}"
    )


    # ========================================================
    # Step 6: Calculate MD MSF
    # ========================================================

    md_msf = np.diag(
        md_cof_residue
    )


    # ========================================================
    # Step 7: MD covariance eigenvalue decomposition
    # ========================================================

    print()
    print("=" * 70)
    print(
        "[4] MD covariance eigenvalue decomposition"
    )
    print("=" * 70)


    # Use eigh because the covariance matrix is symmetric.
    M_values, U = np.linalg.eigh(
        md_cof_residue
    )


    # Transpose of the eigenvector matrix.
    U1 = U.T


    # Check eigenvalues.
    if not np.all(
        np.isfinite(M_values)
    ):

        raise ValueError(
            "MD covariance contains invalid eigenvalues."
        )


    print(
        f"Minimum eigenvalue = "
        f"{M_values.min():.6e}"
    )

    print(
        f"Maximum eigenvalue = "
        f"{M_values.max():.6e}"
    )


    # ========================================================
    # Step 8: Define dfcfGNM calculation
    # ========================================================

    def calculate_dfcfgnmmd(
        rc,
        s
    ):

        # ----------------------------------------------------
        # Calculate p.
        #
        # Original MATLAB expression:
        #
        # p = 10^-7 * (10^s)
        # ----------------------------------------------------

        p = (
            10.0 ** (-7)
            * 10.0 ** s
        )


        # ----------------------------------------------------
        # Calculate diagonal elements of mk.
        # ----------------------------------------------------

        sqrt_term = np.sqrt(
            M_values ** 2
            + 8.0 * p
        )


        denominator = (
            M_values
            + sqrt_term
        )


        # Check for zero denominators.
        if np.any(
            denominator == 0
        ):

            return None


        mk_diag = (
            2.0
            / denominator
        )


        # Check numerical values.
        if not np.all(
            np.isfinite(mk_diag)
        ):

            return None


        # ----------------------------------------------------
        # Calculate K = U * mk * U^T.
        # ----------------------------------------------------

        K = (
            U
            @ np.diag(
                mk_diag
            )
            @ U1
        )


        # ----------------------------------------------------
        # Calculate force constant matrix.
        # ----------------------------------------------------

        k = -K.copy()


        # ----------------------------------------------------
        # Keep only positive force constants.
        # ----------------------------------------------------

        k[k < 0] = 0.0


        # ----------------------------------------------------
        # Construct GNM network matrix.
        # ----------------------------------------------------

        netmat = np.zeros(
            (N, N),
            dtype=float
        )


        # Calculate CA-CA distances.
        for i in range(N):

            for j in range(
                i + 1,
                N
            ):

                dis = np.linalg.norm(
                    posall[i]
                    - posall[j]
                )


                if dis <= rc:

                    netmat[i, j] = (
                        -k[i, j]
                    )

                    netmat[j, i] = (
                        -k[i, j]
                    )


        # ----------------------------------------------------
        # Set diagonal elements.
        # ----------------------------------------------------

        for i in range(N):

            netmat[i, i] = (
                -np.sum(
                    netmat[i, :]
                )
            )


        # Check network matrix.
        if not np.all(
            np.isfinite(netmat)
        ):

            return None


        # ----------------------------------------------------
        # Eigenvalue decomposition.
        # ----------------------------------------------------

        D_values, V = np.linalg.eigh(
            netmat
        )


        # Check eigenvalues.
        if not np.all(
            np.isfinite(D_values)
        ):

            return None


        # ----------------------------------------------------
        # Count zero modes.
        # ----------------------------------------------------

        zero_modes = np.sum(
            np.abs(D_values)
            <= EIG_TOL
        )


        # ----------------------------------------------------
        # Keep positive non-zero modes.
        # ----------------------------------------------------

        valid = (
            D_values
            > EIG_TOL
        )


        D_valid = D_values[
            valid
        ]


        V_valid = V[
            :,
            valid
        ]


        if len(D_valid) == 0:

            return None


        # ----------------------------------------------------
        # Calculate dfcfGNM MSF.
        # ----------------------------------------------------

        dfcfgnmmd_msf = np.sum(
            (
                V_valid ** 2
            )
            /
            D_valid[
                np.newaxis,
                :
            ],
            axis=1
        )


        if not np.all(
            np.isfinite(
                dfcfgnmmd_msf
            )
        ):

            return None


        # ----------------------------------------------------
        # Calculate PCC for MSF.
        # ----------------------------------------------------

        try:

            PCC_msf = pearsonr(
                md_msf,
                dfcfgnmmd_msf
            )[0]

        except ValueError:

            return None


        # ----------------------------------------------------
        # Calculate GNM covariance fluctuation matrix.
        # ----------------------------------------------------

        flu = (
            V_valid
            @ np.diag(
                1.0 / D_valid
            )
            @ V_valid.T
        )


        if not np.all(
            np.isfinite(flu)
        ):

            return None


        # ----------------------------------------------------
        # Calculate PCC for covariance.
        # ----------------------------------------------------

        try:

            PCC_cof = pearsonr(
                md_cof_residue.ravel(),
                flu.ravel()
            )[0]

        except ValueError:

            return None


        # ----------------------------------------------------
        # Calculate average PCC.
        # ----------------------------------------------------

        PCC_avg = (
            PCC_msf
            + PCC_cof
        ) / 2.0


        if not np.isfinite(
            PCC_avg
        ):

            return None


        return {
            "rc": rc,
            "s": s,
            "p": p,
            "zero_modes": zero_modes,
            "PCC_msf": PCC_msf,
            "PCC_cof": PCC_cof,
            "PCC_avg": PCC_avg,
            "netmat": netmat
        }


    # ========================================================
    # Step 9: Automatic optimization
    # ========================================================

    print()
    print("=" * 70)
    print(
        "[5] Automatic parameter optimization"
    )
    print("=" * 70)


    total_combinations = (
        RC_MAX - RC_MIN + 1
    ) * (
        S_MAX - S_MIN + 1
    )


    print(
        f"Total parameter combinations = "
        f"{total_combinations}"
    )


    Pmax = -np.inf

    best_rc = None
    best_s = None
    best_result = None

    optimization_results = []


    # Loop over all rc values.
    for rc in range(
        RC_MIN,
        RC_MAX + 1
    ):

        # Loop over all s values.
        for s in range(
            S_MIN,
            S_MAX + 1
        ):

            result = calculate_dfcfgnmmd(
                rc,
                s
            )


            # Skip invalid combinations.
            if result is None:

                print(
                    f"rc={rc:2d}, "
                    f"s={s}: invalid"
                )

                continue


            optimization_results.append(
                result
            )


            print(
                f"rc={rc:2d}, "
                f"s={s}, "
                f"zero_modes="
                f"{result['zero_modes']}, "
                f"PCC_msf="
                f"{result['PCC_msf']:.6f}, "
                f"PCC_cof="
                f"{result['PCC_cof']:.6f}, "
                f"PCC_avg="
                f"{result['PCC_avg']:.6f}"
            )


            # Update optimal parameters.
            if (
                result["PCC_avg"]
                > Pmax
            ):

                Pmax = (
                    result["PCC_avg"]
                )

                best_rc = rc
                best_s = s
                best_result = result


    # Check optimization result.
    if best_result is None:

        raise RuntimeError(
            "No valid parameter combination was found."
        )


    # ========================================================
    # Step 10: Save optimization results
    # ========================================================

    optimization_file = os.path.join(
        output_dir,
        f"optimization{output_suffix}.dat"
    )


    with open(
        optimization_file,
        "w"
    ) as f:

        f.write(
            "# rc s p zero_modes "
            "PCC_msf PCC_cof PCC_avg\n"
        )


        for result in (
            optimization_results
        ):

            f.write(
                f"{result['rc']} "
                f"{result['s']} "
                f"{result['p']:.8e} "
                f"{result['zero_modes']} "
                f"{result['PCC_msf']:.10f} "
                f"{result['PCC_cof']:.10f} "
                f"{result['PCC_avg']:.10f}\n"
            )


    # ========================================================
    # Step 11: Print optimal parameters
    # ========================================================

    print()
    print("=" * 70)
    print(
        "             Optimal Parameters"
    )
    print("=" * 70)


    print(
        f"Best rc         = {best_rc}"
    )

    print(
        f"Best s          = {best_s}"
    )

    print(
        f"Best p          = "
        f"{best_result['p']:.8e}"
    )

    print(
        f"Best zero modes = "
        f"{best_result['zero_modes']}"
    )

    print(
        f"Best PCC_msf    = "
        f"{best_result['PCC_msf']:.10f}"
    )

    print(
        f"Best PCC_cof    = "
        f"{best_result['PCC_cof']:.10f}"
    )

    print(
        f"Best PCC_avg    = "
        f"{best_result['PCC_avg']:.10f}"
    )


    # ========================================================
    # Step 12: Build final dfcfGNM
    # ========================================================

    print()
    print("=" * 70)
    print(
        "[6] Building final dfcfGNM"
    )
    print("=" * 70)


    final_result = calculate_dfcfgnmmd(
        best_rc,
        best_s
    )


    if final_result is None:

        raise RuntimeError(
            "Failed to construct the final dfcfGNM."
        )


    netmat = final_result[
        "netmat"
    ]


    # ========================================================
    # Step 13: Singular value decomposition
    # ========================================================

    print(
        "Performing singular value decomposition..."
    )


    U_svd, singular_values, Vt_svd = (
        np.linalg.svd(
            netmat
        )
    )


    # ========================================================
    # Step 14: Calculate pseudoinverse
    # ========================================================

    print(
        "Calculating pseudoinverse..."
    )


    inv_s = np.zeros_like(
        singular_values
    )


    valid_svd = (
        singular_values
        > EIG_TOL
    )


    inv_s[
        valid_svd
    ] = (
        1.0
        / singular_values[
            valid_svd
        ]
    )


    # Calculate the pseudoinverse.
    InvKirchhoff = (
        U_svd
        @ np.diag(
            inv_s
        )
        @ U_svd.T
    )


    # ========================================================
    # Step 15: Construct mode-dependent covariance matrices
    # ========================================================

    print(
        "Constructing mode-dependent covariance matrices..."
    )


    CellAijk = []


    for kk in range(N):

        if (
            singular_values[kk]
            > SVD_CUTOFF
        ):

            mode_cov = (
                np.outer(
                    U_svd[:, kk],
                    U_svd[:, kk]
                )
                / singular_values[kk]
            )

        else:

            mode_cov = np.zeros(
                (N, N),
                dtype=float
            )


        CellAijk.append(
            mode_cov
        )


    # ========================================================
    # Step 16: Calculate exponential time weighting
    # ========================================================

    exp_weight = np.exp(
        -singular_values
        * args.tau
    )


    # ========================================================
    # Step 17: Calculate transfer entropy
    # ========================================================

    print()
    print(
        "Calculating transfer entropy..."
    )


    Ti = np.zeros(
        (N, N),
        dtype=float
    )


    for i in range(N):

        # Print progress.
        if (
            i % max(
                1,
                N // 10
            )
            == 0
        ):

            print(
                f"  Progress: "
                f"{i}/{N}"
            )


        for j in range(N):

            # ------------------------------------------------
            # Extract aEk.
            # ------------------------------------------------

            aEk = np.array(
                [
                    CellAijk[kk][j, j]
                    for kk in range(N)
                ]
            )


            # ------------------------------------------------
            # Extract bEk.
            # ------------------------------------------------

            bEk = np.array(
                [
                    CellAijk[kk][i, j]
                    for kk in range(N)
                ]
            )


            # ------------------------------------------------
            # Apply exponential time weighting.
            # ------------------------------------------------

            aEk = (
                aEk
                * exp_weight
            )


            bEk = (
                bEk
                * exp_weight
            )


            # Calculate mode sums.
            sum_aEk = np.sum(
                aEk
            )

            sum_bEk = np.sum(
                bEk
            )


            # ------------------------------------------------
            # Calculate Equation 8 terms.
            # ------------------------------------------------

            # Term a.
            a_term = (
                InvKirchhoff[j, j] ** 2
                - sum_aEk ** 2
            )


            # Term b.
            b_term = (
                InvKirchhoff[i, i]
                * InvKirchhoff[j, j] ** 2
            )


            # Term c.
            c_term = (
                2.0
                * InvKirchhoff[i, j]
                * sum_aEk
                * sum_bEk
            )


            # Term d.
            d_term = (
                -(
                    sum_bEk ** 2
                    + InvKirchhoff[i, j] ** 2
                )
                * InvKirchhoff[j, j]
                - (
                    sum_aEk ** 2
                    * InvKirchhoff[i, i]
                )
            )


            # Term f.
            f_term = (
                InvKirchhoff[j, j]
            )


            # Term g.
            g_term = (
                InvKirchhoff[i, i]
                * InvKirchhoff[j, j]
                - InvKirchhoff[i, j] ** 2
            )


            # Combine terms b, c, and d.
            denominator_term = (
                b_term
                + c_term
                + d_term
            )


            # ------------------------------------------------
            # Check logarithm arguments.
            # ------------------------------------------------

            if (
                a_term > 0
                and denominator_term > 0
                and f_term > 0
                and g_term > 0
            ):

                Ti[i, j] = (

                    0.5
                    * np.log(
                        a_term
                    )

                    - 0.5
                    * np.log(
                        denominator_term
                    )

                    - 0.5
                    * np.log(
                        f_term
                    )

                    + 0.5
                    * np.log(
                        g_term
                    )
                )

            else:

                Ti[i, j] = 0.0


    print(
        f"  Progress: {N}/{N}"
    )


    # ========================================================
    # Step 18: Remove self-information
    # ========================================================

    np.fill_diagonal(
        Ti,
        0.0
    )


    # ========================================================
    # Step 19: Remove negative transfer entropy
    # ========================================================

    Ti[Ti < 0] = 0.0


    # ========================================================
    # Step 20: Calculate net transfer entropy
    # ========================================================

    netTE = (
        Ti
        - Ti.T
    )


    # ========================================================
    # Step 21: Calculate total outgoing transfer entropy
    # ========================================================

    sums_TrMuts = np.sum(
        Ti,
        axis=1
    )


    # ========================================================
    # Step 22: Calculate net information flow
    # ========================================================

    Difference = np.sum(
        netTE,
        axis=1
    )


    # ========================================================
    # Step 23: Normalize net transfer entropy
    # ========================================================

    max_netTE = np.max(
        netTE
    )


    if max_netTE != 0:

        norm_netTE = (
            netTE
            / max_netTE
        )

    else:

        norm_netTE = np.zeros_like(
            netTE
        )


    # ========================================================
    # Step 24: Calculate normalized net information flow
    # ========================================================

    norm_difference = np.sum(
        norm_netTE,
        axis=1
    )


    # ========================================================
    # Step 25: Save transfer entropy matrix
    # ========================================================

    Ti_file = os.path.join(
        output_dir,
        f"transfer_entropy{output_suffix}.dat"
    )


    np.savetxt(
        Ti_file,
        Ti,
        delimiter=" ",
        fmt="%.10f"
    )


    # ========================================================
    # Step 26: Save net transfer entropy
    # ========================================================

    netTE_file = os.path.join(
        output_dir,
        f"netTE{output_suffix}.dat"
    )


    np.savetxt(
        netTE_file,
        netTE,
        delimiter=" ",
        fmt="%.10f"
    )


    # ========================================================
    # Step 27: Save normalized net transfer entropy
    # ========================================================

    norm_netTE_file = os.path.join(
        output_dir,
        f"norm_netTE{output_suffix}.dat"
    )


    np.savetxt(
        norm_netTE_file,
        norm_netTE,
        delimiter=" ",
        fmt="%.10f"
    )


    # ========================================================
    # Step 28: Save normalized net information flow
    # ========================================================

    norm_difference_file = os.path.join(
        output_dir,
        f"norm_difference{output_suffix}.dat"
    )


    np.savetxt(
        norm_difference_file,
        norm_difference,
        delimiter=" ",
        fmt="%.10f"
    )


    # ========================================================
    # Step 29: Save optimal parameters
    # ========================================================

    parameter_file = os.path.join(
        output_dir,
        f"parameters{output_suffix}.dat"
    )


    with open(
        parameter_file,
        "w"
    ) as f:

        f.write(
            "# dfcfGNMMD parameters\n"
        )

        f.write(
            f"PDB = {args.pdb}\n"
        )

        f.write(
            f"Covariance = {args.covar}\n"
        )

        f.write(
            f"Number_of_residues = {N}\n"
        )

        f.write(
            f"rc = {best_rc}\n"
        )

        f.write(
            f"s = {best_s}\n"
        )

        f.write(
            f"p = {best_result['p']:.10e}\n"
        )

        f.write(
            f"tau = {args.tau}\n"
        )

        f.write(
            f"PCC_msf = "
            f"{best_result['PCC_msf']:.10f}\n"
        )

        f.write(
            f"PCC_cof = "
            f"{best_result['PCC_cof']:.10f}\n"
        )

        f.write(
            f"PCC_avg = "
            f"{best_result['PCC_avg']:.10f}\n"
        )

        f.write(
            f"max_netTE = "
            f"{max_netTE:.10e}\n"
        )


    # ========================================================
    # Step 30: Save summary
    # ========================================================

    summary_file = os.path.join(
        output_dir,
        f"summary{output_suffix}.txt"
    )


    with open(
        summary_file,
        "w"
    ) as f:

        f.write(
            "dfcfGNMMD Transfer Entropy\n"
        )

        f.write(
            "============================================\n"
        )

        f.write(
            f"PDB                  : {args.pdb}\n"
        )

        f.write(
            f"Covariance           : {args.covar}\n"
        )

        f.write(
            f"Number of residues   : {N}\n"
        )

        f.write(
            f"Optimal rc            : {best_rc}\n"
        )

        f.write(
            f"Optimal s             : {best_s}\n"
        )

        f.write(
            f"Optimal p             : "
            f"{best_result['p']:.10e}\n"
        )

        f.write(
            f"Tau                   : {args.tau}\n"
        )

        f.write(
            f"PCC_msf               : "
            f"{best_result['PCC_msf']:.10f}\n"
        )

        f.write(
            f"PCC_cof               : "
            f"{best_result['PCC_cof']:.10f}\n"
        )

        f.write(
            f"PCC_avg               : "
            f"{best_result['PCC_avg']:.10f}\n"
        )

        f.write(
            f"Maximum netTE         : "
            f"{max_netTE:.10e}\n"
        )


    # ========================================================
    # Step 31: Print final results
    # ========================================================

    print()
    print("=" * 70)
    print(
        "              Calculation Finished"
    )
    print("=" * 70)


    print()
    print(
        "Optimal parameters:"
    )

    print(
        f"  rc       = {best_rc}"
    )

    print(
        f"  s        = {best_s}"
    )

    print(
        f"  p        = "
        f"{best_result['p']:.8e}"
    )

    print(
        f"  PCC_msf  = "
        f"{best_result['PCC_msf']:.6f}"
    )

    print(
        f"  PCC_cof  = "
        f"{best_result['PCC_cof']:.6f}"
    )

    print(
        f"  PCC_avg  = "
        f"{best_result['PCC_avg']:.6f}"
    )

    print()

    print(
        "Output files:"
    )

    print(
        f"  {residue_covariance_file}"
    )

    print(
        f"  {optimization_file}"
    )

    print(
        f"  {parameter_file}"
    )

    print(
        f"  {summary_file}"
    )

    print(
        f"  {Ti_file}"
    )

    print(
        f"  {netTE_file}"
    )

    print(
        f"  {norm_netTE_file}"
    )

    print(
        f"  {norm_difference_file}"
    )

    print()

    print(
        f"Output directory:"
    )

    print(
        f"  {os.path.abspath(output_dir)}"
    )

    print()
    print("=" * 70)
    print(
        "                    ALL DONE"
    )
    print("=" * 70)


# ============================================================
# Program entry point
# ============================================================

if __name__ == "__main__":

    main()