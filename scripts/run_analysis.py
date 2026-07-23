import sys
import os
ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, ROOT)

import argparse
import numpy as np
from src.utils import *


def run_analysis(
        sys,
        ligand_resname="resname LIG",
        cutoff=10,
        dccm_resid=None,
        workdir="."
):
     
    def p(*args):
        return os.path.join(workdir, *args)
     
    print("===== Start MD analysis =====")

    print("Step 0: Prepare")

    get_dms_and_gems_file(p("./", f"{sys}_score.csv"), p("./", f"{sys}_10A_score.csv"), far_resi_set)

    print("Step 1: DFI and DCI")

    for i in range(1,6):

        get_dfi_dci_rank_file(
            p("DFI_DCI", f"{sys}_md{i}_repc0_new_dfi.csv"),
            p("DFI_DCI", f"{sys}_md{i}_repc0_dfi_rank.csv"),
            "dfi"
        )


    for i in range(1,6):

        get_dfi_dci_rank_file(
            p("DFI_DCI", f"{sys}_md{i}_repc0_new_dci.csv"),
            p("DFI_DCI", f"{sys}_md{i}_repc0_dci_rank.csv"),
            "dci"
        )


    pdb = p("./", f"{sys}.pdb")


    _, far_resi_set = find_distal_residues(
        pdb,
        ligand_resname,
        cutoff
    )



    for i in range(1,6):

        get_far_from_10A_rank(
            p("DFI_DCI", f"{sys}_md{i}_repc0_dfi_rank.csv"),
            p("DFI_DCI", f"{sys}_md{i}_dfi_10A_rank.csv"),
            far_resi_set
        )


        get_far_from_10A_rank(
            p("DFI_DCI", f"{sys}_md{i}_repc0_dci_rank.csv"),
            p("DFI_DCI", f"{sys}_md{i}_dci_10A_rank.csv"),
            far_resi_set
        )



    dfi_files=[
        p("DFI_DCI", f"{sys}_md{i}_dfi_10A_rank.csv")
        for i in range(1,6)
    ]


    average_by_resi(
        dfi_files,
        p("DFI_DCI", f"{sys}_dfi_10A_avg.csv"),
        "dfi"
    )


    dci_files=[
        p("DFI_DCI", f"{sys}_md{i}_dci_10A_rank.csv")
        for i in range(1,6)
    ]


    average_by_resi(
        dci_files,
        p("DFI_DCI", f"{sys}_dci_10A_avg.csv"),
        "dci"
    )


    for n in range(10,101,10):


        get_rank(
            p("./", f"{sys}_10A_score.csv"),
            p("DFI_DCI", f"{sys}_dfi_10A_avg.csv"),
            p("DFI_DCI", f"{sys}_dfi_top{n}.csv"),
            "dfi",
            n
        )


        get_rank(
            p("./", f"{sys}_10A_score.csv"),
            p("DFI_DCI", f"{sys}_dci_10A_avg.csv"),
            p("DFI_DCI", f"{sys}_dci_top{n}.csv"),
            "dci",
            n
        )



    # ============================
    # 2.1 SPM
    # ============================

    print("Step 2: SPM")


    get_site_rank(
        p("spm", "shortest_path.pml"),
        p("spm", f"{sys}_sphere_rank.csv"),
        far_resi_set
    )

   
    for n in range(10,101,10):

        get_rank(
            p("./", f"{sys}_10A_score.csv"),
            p("spm", f"{sys}_sphere_rank.csv"),
            p("spm", f"{sys}_spm_top{n}.csv"),
            "Count",
            n
        )



    # ============================
    # 3. TE
    # ============================

    print("Step 3: TE")


    for i in range(1,6):

        get_rank_from_netTE(
            p("trans_entropy", f"norm_netTE_{i}.dat"),
            p("trans_entropy", f"norm_netTE_rank_{i}.dat")
        )
    for i in range(1,6):
        get_far_from_10A_rank(
            p("trans_entropy", f"norm_netTE_rank_{i}.dat"),
            p("trans_entropy", f"norm_netTE_10A_rank_{i}.dat"),
            far_resi_set
        )
       
    te_files=[
        p("trans_entropy", f"norm_netTE_10A_rank_{i}.dat")
        for i in range(1,6)
    ]
    
      
    average_by_resi(
        te_files,
        p("trans_entropy", f"{sys}_TE_avg.csv"),
        "TE_out"
    )

 
    for n in range(10,101,10):

        get_rank(
            p("./", f"{sys}_10A_score.csv"),
            p("trans_entropy", f"{sys}_TE_avg.csv"),
            p("trans_entropy", f"{sys}_TE_top{n}.csv"),
            "TE_out",
            n
        )



    # ============================
    # 4. DCCM
    # ============================

    print("Step 4: DCCM")

    for i in range(1,6):
        for resid in dccm_resid:

            get_residue_corr(
                resid,
                p("dccm", f"{sys}_corr_{i}.dat"),
                p("dccm", f"{sys}_md{i}_{resid}_dccm.csv")
            )


  
    for i in range(1,6):
        dccm_files_i = []
        for resid in dccm_resid:
            dccm_files_i.append(
                p("dccm", f"{sys}_md{i}_{resid}_dccm.csv")
            )


        outfile=p("dccm", f"{sys}_md{i}_dccm_avg_rank.csv")


        get_dccm_rank_file(
            dccm_files_i,
            outfile
        )

    dccm_files = []
    for i in range(1,6):
        outfile=p("dccm", f"{sys}_md{i}_dccm_avg_10A_rank.csv")
        get_far_from_10A_rank(p("dccm", f"{sys}_md{i}_dccm_avg_rank.csv"), 
                              outfile,
                              far_resi_set=far_resi_set)
        dccm_files.append(outfile)
    
    average_by_resi(dccm_files, p("dccm", f"{sys}_dccm_10A_rank_avg.csv"), "DCCM_avg")

    for n in range(10, 101, 10):
        get_rank(p("./", f"{sys}_10A_score.csv"), 
                 p("dccm", f"{sys}_dccm_10A_rank_avg.csv"), 
                 p("dccm", f"{sys}_dccm_10A_exp_rank_top{n}.csv"), 
                 "DCCM_avg", n)

    print(f"===== {sys} finished =====")



if __name__ == "__main__":


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--sys",
        required=True,
        help="protein system name"
    )


    parser.add_argument(
        "--ligand",
        default="resname LIG"
    )


    parser.add_argument(
        "--cutoff",
        default=10,
        type=float
    )

    parser.add_argument(
    "--dccm_resid",
    nargs="+",
    type=int,
    required=True,
    help="target residues for DCCM analysis"
    ) 

    parser.add_argument(
    "--workdir",
    default=os.getcwd(),
    help="working directory"
    )

    args = parser.parse_args()


    run_analysis(
        args.sys,
        args.ligand,
        args.cutoff,
        args.dccm_resid,
        args.workdir
    )