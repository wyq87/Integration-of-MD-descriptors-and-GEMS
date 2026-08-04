
import csv
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import os
import re
import numpy as np
import pandas as pd
from collections import defaultdict

def get_dfi_dci_rank_file(input_file, output_file, indicator):
    with open(input_file, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    rows_sorted = sorted(rows, key=lambda x: float(x[indicator]), reverse=True)

    with open(output_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows_sorted)



def find_distal_residues(file, ligand_resname="resname LIG", cutoff=10):

    u = mda.Universe(file)
    ligand = u.select_atoms(ligand_resname)
    lig_heavy = ligand.select_atoms("not name H*")

    resid2dist = {}

    for res in u.select_atoms("protein").residues:
      
        res_heavy = res.atoms.select_atoms("not name H*")

        if len(res_heavy) == 0 or len(lig_heavy) == 0:
            continue

        d = distances.distance_array(
            res_heavy.positions,
            lig_heavy.positions
        ).min()

        resid2dist[res.resid] = d

    far_resi_set = {r for r, d in resid2dist.items() if d >= cutoff}

    return resid2dist, far_resi_set
   
def average_by_resi(files, output_file, indicator):
    dfs = []

    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(df_all.columns)
    df_mean = (
        df_all
        .groupby("ResID", as_index=False)
        .agg({
            # "Chain": "first",      
            "ResID": "first",
            # "Res": "first",
            indicator: "mean",
           
        })
        .sort_values(indicator, ascending=False)
    )

    df_mean.to_csv(output_file, index=False)
  

def extract_resid(mut):
        m = re.search(r"\d+", mut)
        return int(m.group()) if m else None
        
def get_rank(dms_rank_file, md_descriptor_rank_file, output_file, indicator, num):
  
    df_dms = pd.read_csv(dms_rank_file)   
    df_md  = pd.read_csv(md_descriptor_rank_file)   

    df_dms["ResID"] = df_dms["mutant"].apply(extract_resid)
    df_dms["avg_rank"] = df_dms["avg_rank"].astype(int)
    df_dms["DMS_rank"] = df_dms["DMS_rank"].astype(int)
    
    col = "ResID" if "ResID" in df_md.columns else "ResI"
    top_residues = ( df_md.sort_values(indicator, ascending=False) .head(num)[col] .tolist() )
    
    results = []

    for resid in top_residues:
       sub = df_dms[df_dms["ResID"] == resid]

       if sub.empty:
           results.append({
               "ResID": resid,
               "Best_mutant": "N/A",
               "DMS_rank": "N/A",
               "avg_rank": "N/A",
               "DMS_score_bin": "N/A"
           })
       else:
          
           sub_sorted = sub.sort_values("avg_rank")

           for _, row in sub_sorted.iterrows():
               results.append({
                   "ResID": resid,
                   "Best_mutant": row["mutant"],
                   "DMS_rank": row["DMS_rank"],
                   "avg_rank": row["avg_rank"],
                   "DMS_score_bin": row["DMS_score_bin"]
               })
    out_df = pd.DataFrame(results) 

    out_df["avg_rank"] = pd.to_numeric(out_df["avg_rank"], errors="coerce")
    out_df = out_df.sort_values("avg_rank", na_position="last")
    
    out_df.to_csv(output_file, index=False)

def get_site_rank(inputfile="shortest_path.pml", outputfile="sphere_rank.csv", far_resi_set=None):
    sphere_dict = {}   # {resi: max_scale}

    current_resi = []
    with open(inputfile) as f:
        for line in f:
            if "sphere_scale" in line:
                scale = float(re.findall(r"sphere_scale,([0-9\.]+)", line)[0])
                resi = int(re.findall(r"resi (\d+)", line)[0])
                if resi not in sphere_dict or scale > sphere_dict[resi]:
                    sphere_dict[resi] = scale

                current_resi.append(resi)

    sphere_dict = {resi: scale for resi, scale in sphere_dict.items() if resi in far_resi_set}
  
    sphere_sorted = sorted(sphere_dict.items(), key=lambda x: x[1], reverse=True)
    sphere_df = pd.DataFrame(sphere_sorted, columns=["ResID", "Count"])

    sphere_df.to_csv(outputfile, index=False)

def get_rank_from_netTE(norm_netTE_file, output_file):
    
    netTE = np.loadtxt(norm_netTE_file)
    N = netTE.shape[0]
    te_out = np.sum(netTE, axis=1)

    candidates = []
    for i in range(N):
        res_id = i + 1  
        candidates.append((res_id, te_out[i]))

    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    header = "ResID  TE_out"
    np.savetxt(output_file, sorted_candidates, fmt="%-4d %.6f", header=header, comments='')


def get_far_from_10A_rank(file, outfile, far_resi_set):

    if file.endswith(".dat"):
        df = pd.read_csv(file, delim_whitespace=True)
    else:
        df = pd.read_csv(file)
 
    df_far = df[df["ResID"].isin(far_resi_set)]

    df_far.to_csv(outfile, index=False)

def get_dccm_rank_file(files, output_file):
    data = defaultdict(lambda: {
        "DCCM_sum": 0.0,
        "count": 0
    })

    for fname in files:
        with open(fname, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                resi = int(row["ResID"])
                data[resi]["DCCM_sum"] += float(row["DCCM_corr"])
                data[resi]["count"] += 1

    rows = []
    for resi, v in data.items():
        rows.append({
            "ResID": resi,
            "DCCM_avg": v["DCCM_sum"] / v["count"],
        })

    rows_sorted = sorted(rows, key=lambda x: x["DCCM_avg"], reverse=True)


    with open(output_file, "w", newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ResID", "DCCM_avg"]
        )
        writer.writeheader()
        writer.writerows(rows_sorted)

def get_residue_corr(target_resid, dccm_file, output_file):
    
    dccm = np.loadtxt(dccm_file)
    idx = target_resid - 1   # 0-based


    corr_row = dccm[idx]


    positive_corr = [
        (i + 1, corr)
        for i, corr in enumerate(corr_row)
        # if i != idx and corr > 0
    ]

    df_out = pd.DataFrame(
        positive_corr,
        columns=["ResID", "DCCM_corr"]
    )

    df_out.to_csv(output_file, index=False)


def get_dms_and_gems_file(file, outfile, far_resi_set):

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

   
    df["Residue"] = pd.to_numeric(
        df["mutant"].str.extract(r'(\d+)')[0],
        errors="coerce"
    )
    df_near = df[df["Residue"].isin(far_resi_set)].copy()


    df_near["DMS_rank"] = (
        df_near["DMS_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    df_near["avg_rank"] = (
        df_near["avg_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    df_near = df_near.sort_values("DMS_rank")

   
    df_near.to_csv(outfile, index=False)

    print(f"{outfile} has been generated with {len(df_near)} entries.")


