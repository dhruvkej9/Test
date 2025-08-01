"""
Shared utilities for the drug discovery application
"""
import pubchempy as pcp
import re
import requests as pyrequests
from rdkit import Chem

# Common name to SMILES mapping
COMMON_NAME_TO_SMILES = {
    'paracetamol': 'CC(=O)NC1=CC=C(O)C=C1',
    'acetaminophen': 'CC(=O)NC1=CC=C(O)C=C1',
    'aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
    'caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
    'ibuprofen': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
    'nicotine': 'CN1CCCC1C2=CN=CC=C2',
    'sodium chloride': '[Na+].[Cl-]',
    'table salt': '[Na+].[Cl-]',
    'glucose': 'C(C1C(C(C(C(O1)O)O)O)O)O',
    'ethanol': 'CCO',
    'water': 'O',
    'acetone': 'CC(=O)C',
    'benzene': 'C1=CC=CC=C1',
    'chloroform': 'ClC(Cl)Cl',
    'methane': 'C',
    'carbon dioxide': 'O=C=O',
    'ammonia': 'N',
    'hydrochloric acid': 'Cl',
    'sulfuric acid': 'O=S(=O)(O)O',
    'nitric acid': 'O=N(=O)O',
    'sodium hydroxide': '[Na+].[OH-]',
    'potassium permanganate': 'O=[Mn](=O)(=O)=O.[K+]',
}

def name_to_smiles(query):
    """
    Convert compound name or SMILES string to canonical SMILES.
    
    Args:
        query (str): Compound name or SMILES string
        
    Returns:
        tuple: (smiles_string, error_message)
    """
    q_lower = query.strip().lower()
    
    # 1. Manual dictionary
    if q_lower in COMMON_NAME_TO_SMILES:
        return COMMON_NAME_TO_SMILES[q_lower], None
    
    # 2. Try PubChem by name
    name_error = None
    try:
        compounds = pcp.get_compounds(query, 'name')
        if compounds and hasattr(compounds[0], 'isomeric_smiles') and compounds[0].isomeric_smiles:
            return compounds[0].isomeric_smiles, None
    except Exception as e:
        name_error = f"PubChem lookup failed for '{query}' (name): {e}"
    
    # 3. Try PubChem by synonym, but ignore BadRequest
    try:
        compounds = pcp.get_compounds(query, 'synonym')
        if compounds and hasattr(compounds[0], 'isomeric_smiles') and compounds[0].isomeric_smiles:
            return compounds[0].isomeric_smiles, None
    except Exception as e:
        if "BadRequest" not in str(e):
            return None, f"PubChem lookup failed for '{query}' (synonym): {e}"
    
    # 4. Try NIH CACTUS resolver as fallback
    try:
        cactus_url = f"https://cactus.nci.nih.gov/chemical/structure/{pyrequests.utils.quote(query)}/smiles"
        resp = pyrequests.get(cactus_url, timeout=5)
        if resp.status_code == 200 and resp.text and 'Not Found' not in resp.text:
            smiles = resp.text.strip()
            # Validate with RDKit
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return smiles, None
    except Exception as e:
        pass
    
    # 5. Try as SMILES
    try:
        mol = Chem.MolFromSmiles(query)
        if mol and re.match(r'^[A-Za-z0-9@+\-=#$%\[\]()/\\.]+$', query):
            return query, None
    except Exception:
        pass
    
    # 6. Fail with clear error, but prefer name_error if available
    if name_error:
        return None, name_error
    
    return None, f"Could not resolve '{query}' to a SMILES string using local dictionary, PubChem, or NIH CACTUS. Please check the spelling, try a more specific name, or provide a valid SMILES."

def get_pubchem_cid(smiles):
    """Get PubChem CID for a SMILES string."""
    try:
        compounds = pcp.get_compounds(smiles, 'smiles')
        if compounds and compounds[0].cid:
            return compounds[0].cid
    except Exception:
        pass
    return None