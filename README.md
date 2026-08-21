# Integration-of-MD-descriptors-and-GEMS
This repository provides a workflow for integrating molecular dynamics (MD)-derived descriptors with GEMS for identifying potential distal mutation sites.
The workflow combines multiple MD-based descriptors, including:
- Dynamic Flexibility Index (DFI)
- Dynamic Coupling Index (DCI)
- Shortest Path Map (SPM)
- Transfer Entropy (TE)
- Dynamic Cross-Correlation Matrix (DCCM)
  
for ranking candidate mutation sites and evaluating their enrichment using deep mutational scanning (DMS) data.

## Workflow
Benchmark workflow for integrating five MD-derived descriptors with GEMS. For seven enzymes, five independent 200 ns MD simulations were performed. After confirming structural stability, trajectories from 20-200 ns were used to calculate five MD-derived dynamic descriptors, including DCCM, DFI, DCI, TE, and SPM. For each descriptor, residues were ranked based on their average values across five trajectories, and the top 10-100 sites were selected. For each descriptor, all mutations at selected sites were ranked by GEMS score, and the top 100 were selected. Their union defined the hybrid mutation set. Performance was assessed by the number of variants ranked within the top 100 of the corresponding DMS landscape.

<img width="865" height="675" alt="image" src="https://github.com/user-attachments/assets/6b151cb3-c3ba-4640-9e49-134a08315390" />

## DMS and GEMS Data

The DMS datasets and GEMS-related input files are provided in directory `data`

These files are used for evaluating the prediction performance of MD-derived descriptors.

## Calculation of MD Descriptors

### 1. DFI and DCI

DFI and DCI are calculated from representative structures extracted from MD trajectories.

#### Step 1: Extract representative structures

Use:

```
scripts/cluster.in
```

to perform trajectory clustering and extract the representative structure (top-ranked cluster structure) from each MD trajectory.

Example:

```bash
cpptraj -i scripts/cluster.in
```

---

#### Step 2: Calculate DFI and DCI

Example:

```bash
python scripts/calc_dfi_dci.py --pdb bgl3.pdb
```
```bash
python scripts/calc_dfi_dci.py --pdb bgl3.pdb --dci A178 A383
```
`A178` and `A383` are functual residues in active center. 

The script calculates:

- Dynamic Flexibility Index (DFI)
- Dynamic Coupling Index (DCI)

The generated residue-level scores are used for ranking candidate mutation sites.

---

### 2. Shortest Path Map (SPM)

SPM analysis requires residue-residue distance and correlation matrices.

Generate the required files using:

```
scripts/correl_all.in
```

Example:

```bash
cpptraj -i scripts/correl_all.in
```

Upload the generated matrices together with the structure file to:

[SPMweb](https://spmosuna.com/)

The output file:

```
shortest_path.pml
```

is used for subsequent SPM-based residue ranking.

---

### 3. Transfer Entropy (TE)

#### Step 1
Generate covariance matrices from MD trajectories using:

```
scripts/covar.in
```

Example:

```bash
cpptraj -i scripts/covar.in
```

The covariance matrices are used for transfer entropy analysis to characterize residue communication.

---

#### Step 2
Generate `norm_differece`file using:
```
scripts/dfcfGNMMD.py --pdb bgl3.pdb --covar bgl3_covar_1.dat
```
`bgl3_covar_1.dat` is the covariance file generated in Step 1.

The output file will be named according to the input covariance file, e.g., norm_difference_1, corresponding to `bgl3_covar_1.dat`.

### 4. Dynamic Cross-Correlation Matrix (DCCM)

Generate residue correlation matrices using:

```
scripts/correl.in
```

Example:

```bash
cpptraj -i scripts/correl.in
```

The generated correlation matrices are used for DCCM-based residue ranking.

---

## Identification of Distal Candidate Sites

After calculating all MD descriptors, run:

```bash
python scripts/run_analysis.py --sys bgl3 --ligand "resname LIG" --cutoff 10 --dccm_resid 178 383 --workdir "/home/user/MD_GEMS"
```

- `--sys`

  Name of the protein system. The system name should be consistent with the corresponding PDB file name.

  Example:

  ```bash
  --sys bgl3
  ```

  corresponds to:

  ```
  bgl3.pdb
  ```

---

- `--ligand`

  Ligand name used for identifying the active site. The ligand selection follows the atom selection syntax of **MDAnalysis**.

  Example:

  ```bash
  --ligand "resname LIG"
  ```

  where `LIG` is the residue name of the ligand in the structure file.

---

- `--cutoff`

  Distance cutoff (Å) used to define distal candidate sites.

  Residues located farther than the specified cutoff distance from the active site are selected as distal mutation candidates.

  Example:

  ```bash
  --cutoff 10
  ```

  selects residues more than 10 Å away from the active site.

---

- `--dccm_resid`

  Functional residues used as reference sites for DCCM analysis.

  These residues should be identical to the functional residues used for DCI calculation.

  Multiple residues can be specified.

  Example:

  ```bash
  --dccm_resid 178 383
  ```

  indicates that residues 178 and 383 are used as functional communication centers.

---

- `--workdir`

  Absolute path of the working directory containing all input files and analysis folders.

  Example:

  ```bash
  --workdir "/home/user/GEMS_MD"
  ```

  The working directory should contain the required input files, including:

  ```
  PDB structure file
  MD descriptor files
  DMS data
  ```

---

The workflow performs:

1. Descriptor ranking
2. Averaging across MD replicates
3. Removal of residues within 10 Å of functional sites
4. Integration with DMS datasets
5. Identification of distal candidate mutation sites

## Output

All designed mutant sets are saved using the naming convention:

`xx_topn.csv`

where `xx` denotes the corresponding MD descriptor (e.g., dfi, dci, te, dccm, or spm), and `n` indicates the number of selected candidate sites, ranging from 10 to 100.


## Notes

This repository provides an experimental implementation of the MD descriptor and GEMS integration workflow.

## Citation
If you used this workflow, please cite these references:
- [Prediction of Distal Mutation Effects in Enzymes via Integration of Molecular Dynamics Descriptors and Zero-Shot Model](https://doi.org/10.1021/acs.jctc.6c00829) 
- DFI, DCI: [Perturbation-Response Scanning Reveals Ligand Entry-Exit Mechanisms of Ferric Binding Protein](https://doi.org/10.1371/journal.pcbi.1000544)\
            [Structural dynamics flexibility informs function and evolution at a proteome scale](https://doi.org/10.1111/eva.12052)\
            [Design of novel cyanovirin-N variants by modulation of binding dynamics through distal mutations]( https://doi.org/10.7554/eLife.67474)
- SPM: [The shortest path method (SPM) webserver for computational enzyme design](https://doi.org/10.1093/protein/gzae005)\
       [The challenge of predicting distal active site mutations in computational enzyme design](https://doi.org/10.1002/wcms)
- TE:  [Study of the Allosteric Mechanism of Human Mitochondrial Phenylalanyl-tRNA Synthetase by Transfer Entropy via an Improved Gaussian Network Model and Co-evolution Analyses](https://doi.org/10.1021/acs.jpclett.3c00366)

## License

This project is released under the MIT License.
