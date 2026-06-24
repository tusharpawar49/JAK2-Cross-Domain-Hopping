import os
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

# Suppress standard RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

print("="*70)
print("3D LIGAND PREPARATION: pH 7.4 & MMFF94s MINIMIZATION")
print("="*70)

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_FILE = "JAK2_Diverse_Elite_Ready_for_Docking.tsv"
OUTPUT_SDF = "JAK2_GNINA_Ready_Ligands_376.sdf"

try:
    df = pd.read_csv(INPUT_FILE, sep='\t')
    smiles_col = 'Clean_SMILES' if 'Clean_SMILES' in df.columns else df.columns[0]
    elite_smiles = df[smiles_col].dropna().tolist()
    print(f"Loaded {len(elite_smiles)} elite SMILES from '{INPUT_FILE}'.")
except Exception as e:
    raise ValueError(f"Failed to load file. Error: {e}")

# ==========================================
# 2. PHYSIOLOGICAL pH 7.4 IONIZER
# ==========================================
# SMARTS patterns for common physiological ionization
acidic_smarts = Chem.MolFromSmarts("[CX3](=O)[OX1H0-,OX2H1]") # Carboxylic acids
basic_smarts = Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(NC=O);!$(Na)]") # Aliphatic amines

def ionize_at_ph_74(mol):
    """Protonates basic amines and deprotonates acids to simulate pH 7.4"""
    mol.UpdatePropertyCache(strict=False)
    
    # Deprotonate acids -> [O-]
    acid_matches = mol.GetSubstructMatches(acidic_smarts)
    for match in acid_matches:
        idx = match[2] if len(match) > 2 else match[-1]
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 8 and atom.GetFormalCharge() == 0:
            atom.SetFormalCharge(-1)
            atom.SetNumExplicitHs(0)
            
    # Protonate amines -> [NH+]
    base_matches = mol.GetSubstructMatches(basic_smarts)
    for match in base_matches:
        idx = match[0]
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 0:
            atom.SetFormalCharge(1)
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
            
    Chem.SanitizeMol(mol)
    return mol

# ==========================================
# 3. STANDARDIZATION & 3D GENERATION
# ==========================================
# Initialize the standardizer
enumerator = rdMolStandardize.TautomerEnumerator()

prepared_mols = []
fail_standard = 0
fail_3d = 0
fail_minimize = 0

print("\nExecuting 3D Embedding and MMFF94s Minimization Funnel...")

# Open the SDWriter to stream molecules directly into the SDF file
writer = Chem.SDWriter(OUTPUT_SDF)

for i, smi in enumerate(tqdm(elite_smiles)):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: continue
    
    # Step A: Canonicalize Tautomer (forces the most stable tautomeric state)
    try:
        mol = enumerator.Canonicalize(mol)
    except:
        fail_standard += 1
        continue
        
    # Step B: Adjust Formal Charges for pH 7.4
    try:
        mol = ionize_at_ph_74(mol)
    except:
        fail_standard += 1
        continue
        
    # Step C: Add Explicit Hydrogens (Crucial for 3D routing and docking)
    mol = Chem.AddHs(mol)
    
    # Step D: Generate 3D Coordinates via ETKDGv3
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True # Helps resolve highly constrained macrocycles
    # RDKit dynamically handles embedding attempts natively in modern versions
    
    embed_status = AllChem.EmbedMolecule(mol, params)
    if embed_status == -1:
        fail_3d += 1
        continue
        
    # Step E: Force Field Minimization (MMFF94s for planar nitrogens)
    try:
        # maxIters = 2000 ensures thermodynamic convergence of the structure
        opt_status = AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94s', maxIters=2000)
        if opt_status != 0:
            # 1 indicates failure to converge within maxIters
            fail_minimize += 1
            continue
    except:
        fail_minimize += 1
        continue
        
    # Step F: Tag with ID and write to SDF
    mol.SetProp("_Name", f"JAK2_Elite_Ligand_{i+1}")
    writer.write(mol)
    prepared_mols.append(mol)

writer.close()

# ==========================================
# 4. FINAL REPORT
# ==========================================
print("\n" + "="*70)
print("3D PREPARATION ATTRITION REPORT")
print("="*70)
print(f"Total Input SMILES:              {len(elite_smiles)}")
print(f"Failed Standardization/Tautomer: {fail_standard}")
print(f"Failed ETKDGv3 3D Embedding:     {fail_3d}")
print(f"Failed MMFF94s Minimization:     {fail_minimize}")
print("-" * 70)
print(f"SUCCESSFULLY PREPARED FOR GNINA: {len(prepared_mols)}")
print("="*70)
print(f"Final 3D Coordinates Saved To:   {OUTPUT_SDF}")