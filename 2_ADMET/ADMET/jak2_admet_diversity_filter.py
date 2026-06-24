import os
import sys
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, DataStructs, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from rdkit.Chem import RDConfig

# Suppress RDKit warnings for clean terminal output
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Import the SA Score module from RDKit Contrib
try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
except ImportError:
    raise ImportError("Could not import sascorer. Ensure RDKit contrib modules are properly installed.")

print("="*70)
print("JAK2 TARGETED ADMET, SYNTHESIS & DIVERSITY FUNNEL")
print("="*70)

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================
INPUT_FILE = "1L_JAK2_graph_RL_graph_trans_RL.tsv"
OUTPUT_FILE = "JAK2_Diverse_Elite_Ready_for_Docking.tsv"
SIMILARITY_THRESHOLD = 0.65  # Q1-standard for defining genuine scaffold diversity

try:
    df = pd.read_csv(INPUT_FILE, sep='\t')
    smiles_col = 'SMILES' if 'SMILES' in df.columns else df.columns[0]
    raw_smiles = df[smiles_col].dropna().tolist()
    print(f"Loaded {len(raw_smiles)} generated molecules from '{INPUT_FILE}'.")
except Exception as e:
    raise ValueError(f"Failed to load file. Error: {e}")

# ==========================================
# 2. TOXICITY & hERG INITIALIZATION
# ==========================================
params = FilterCatalogParams()
params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK) 
params.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
params.AddCatalog(FilterCatalogParams.FilterCatalogs.ZINC)
tox_catalog = FilterCatalog(params)

# Custom hERG Proxy: Aliphatic basic nitrogen + High Lipophilicity
basic_nitrogen = Chem.MolFromSmarts("[N;X3;+0;!$(N-C=[O,S,N]);!$(N-a)]")

# ==========================================
# 3. RING GEOMETRY VALIDATOR
# ==========================================
def is_realistic_ring_system(mol):
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) < 4 or len(ring) > 6:
            return False
        heteroatom_count = sum(1 for idx in ring if mol.GetAtomWithIdx(idx).GetAtomicNum() != 6)
        if heteroatom_count > 3:
            return False
    if rdMolDescriptors.CalcNumBridgeheadAtoms(mol) > 0:
        return False
    return True

# ==========================================
# 4. SCREENING FUNNEL EXECUTION
# ==========================================
elite_mols = []

fail_parse = 0
fail_physchem = 0
fail_pharmacophore = 0
fail_tox = 0
fail_herg = 0
fail_qed = 0
fail_rings = 0
fail_sa = 0

print("\nRunning Target-Specific ADMET & Synthesizability De-risking...")

for smi in tqdm(raw_smiles):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        fail_parse += 1
        continue
        
    # A. JAK2 Specific Physicochemical Bounds (Strict Oral + BBB Exclusion)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rotb = Descriptors.NumRotatableBonds(mol)
    
    # Tightened for JAK2 (MW: 300-480, LogP: 1.5-3.8, TPSA: 70-115, RotB <= 6)
    if not (300 <= mw <= 480) or not (1.5 <= logp <= 3.8) or not (1 <= hbd <= 3) or not (4 <= hba <= 8) or not (70 <= tpsa <= 115) or (rotb > 6):
        fail_physchem += 1
        continue

    # B. JAK2 Hinge-Binder Pharmacophore Constraint
    num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    num_aromatic_heterocycles = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
    
    # Must have 2-4 aromatic rings, and at least one MUST be a heteroaromatic ring for hinge binding
    if not (2 <= num_aromatic_rings <= 4) or (num_aromatic_heterocycles < 1):
        fail_pharmacophore += 1
        continue
        
    # C. Ring Geometry & Topology
    if not is_realistic_ring_system(mol):
        fail_rings += 1
        continue
        
    # D. Synthetic Accessibility (SA) Score (1 = easy, 10 = impossible)
    sa_score = sascorer.calculateScore(mol)
    if sa_score > 4.5:
        fail_sa += 1
        continue
        
    # E. Substructural Toxicity Alerts
    if tox_catalog.HasMatch(mol):
        fail_tox += 1
        continue
        
    # F. hERG Cardiotoxicity Proxy
    if logp > 3.5 and mol.HasSubstructMatch(basic_nitrogen):
        fail_herg += 1
        continue
        
    # G. Quantitative Estimate of Drug-likeness (QED)
    if QED.qed(mol) < 0.60:
        fail_qed += 1
        continue
        
    # Survived all strictly defined parameters
    elite_mols.append((smi, mol))

num_survivors = len(elite_mols)
print(f"\n=> {num_survivors} molecules survived strict JAK2 target profiling.")

# ==========================================
# 5. NATURAL DIVERSITY EXTRACTION & OBJECTIVE FILTERING
# ==========================================
# ==========================================
# 5. NATURAL DIVERSITY EXTRACTION & OBJECTIVE FILTERING
# ==========================================
MIN_CLUSTER_SIZE = 2   # Allow "consensus pairs" (RL model found the scaffold more than once)
STRICT_QED_CUTOFF = 0.70 # Optimized for the bulky reality of hinge-binding kinase inhibitors

if num_survivors == 0:
    print("No molecules survived the funnel.")
    final_smiles = []
else:
    print(f"\nClustering survivors to find natural distinct scaffolds (Max Similarity: {SIMILARITY_THRESHOLD})...")
    
    mfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [mfp_gen.GetFingerprint(mol) for smi, mol in elite_mols]
    
    dists = []
    nfps = len(fps)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - x for x in sims])
        
    dist_threshold = 1.0 - SIMILARITY_THRESHOLD
    clusters = Butina.ClusterData(dists, nfps, dist_threshold, isDistData=True)
    
    # 1. CLUSTER DENSITY FILTER: Keep only clusters where the generative model showed consensus
    dense_clusters = [cluster for cluster in clusters if len(cluster) >= MIN_CLUSTER_SIZE]
    
    print(f"Total initial clusters: {len(clusters)}")
    print(f"Dense consensus clusters (>= {MIN_CLUSTER_SIZE} members): {len(dense_clusters)}")
    
    # Extract the centroid (index 0) of each dense cluster
    centroid_data = [(elite_mols[cluster[0]][0], elite_mols[cluster[0]][1]) for cluster in dense_clusters]
    
    # 2. OBJECTIVE QED FILTER: Keep only the absolute highest-tier drug-like centroids
    final_smiles = []
    for smi, mol in centroid_data:
        if QED.qed(mol) >= STRICT_QED_CUTOFF:
            final_smiles.append(smi)
            
    print(f"Centroids passing strict QED limit (>= {STRICT_QED_CUTOFF}): {len(final_smiles)}")

# ==========================================
# 6. FINAL REPORT & EXPORT
# ==========================================
total = len(raw_smiles)
survival_rate = (len(final_smiles) / total) * 100 if total > 0 else 0

print("\n" + "="*70)
print("FINAL ATTRITION REPORT FOR MANUSCRIPT")
print("="*70)
print(f"Total Input Molecules:            {total}")
print(f"Failed SMILES Parsing:            {fail_parse}")
print(f"Failed General PhysChem Bounds:   {fail_physchem}")
print(f"Failed JAK2 Hinge Pharmacophore:  {fail_pharmacophore}")
print(f"Failed Ring Topology Limits:      {fail_rings}")
print(f"Failed Synthetic Accessibility:   {fail_sa}")
print(f"Failed Toxicity Alerts (PAINS+):  {fail_tox}")
print(f"Failed hERG Proxy Alert:          {fail_herg}")
print(f"Failed QED Strict Threshold:      {fail_qed}")
print("-" * 70)
print(f"Total Viable Survivors:           {num_survivors}")
print(f"Distinct Clusters Extracted:      {len(final_smiles)} (Butina threshold={SIMILARITY_THRESHOLD})")
print("-" * 70)
print(f"ELITE SURVIVORS FOR DOCKING:      {len(final_smiles)}")
print(f"Final Selection Rate:             {survival_rate:.2f}%")
print("="*70)

if len(final_smiles) > 0:
    df_out = pd.DataFrame({"Clean_SMILES": final_smiles})
    df_out.to_csv(OUTPUT_FILE, sep='\t', index=False)
    print(f"\nFiltered diverse SMILES saved successfully to: {OUTPUT_FILE}")