import streamlit as st
import requests
from rdkit import Chem
from rdkit.Chem import Draw
from src.components.mol3d_viewer import show_molecule
import pubchempy as pcp
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
import numpy as np
import matplotlib.pyplot as plt
import time
import sys

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
    # Add more as needed
}

st.set_page_config(page_title="Drug Toxicity Prediction", layout="wide")
# Add custom CSS for wider content and professional color sections
st.markdown("""
<style>
/* Make main container wider */
.css-18e3th9 { max-width: 1200px; }
/* Section headers with color */
.section-header { font-size: 1.5em; font-weight: bold; margin-top: 1em; margin-bottom: 0.5em; color: #222; }
.section-tox { background: #fff4f4; border-left: 6px solid #e53935; padding: 1em; border-radius: 8px; }
.section-sol { background: #f0f8ff; border-left: 6px solid #1e88e5; padding: 1em; border-radius: 8px; }
.section-intox { background: #f9fbe7; border-left: 6px solid #cddc39; padding: 1em; border-radius: 8px; }
.section-pubchem { background: #f3e5f5; border-left: 6px solid #8e24aa; padding: 1em; border-radius: 8px; }
.section-pubchem .section-header { color: #4a148c; }
.section-tox .section-header { color: #b71c1c; }
.section-sol .section-header { color: #1565c0; }
.section-intox .section-header { color: #827717; }
</style>
""", unsafe_allow_html=True)
st.title("💊 Drug Toxicity Prediction (GNN)")

def name_to_smiles(query):
    import pubchempy as pcp
    import re
    import requests as pyrequests
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
    try:
        from pubchempy import get_compounds
        compounds = get_compounds(smiles, 'smiles')
        if compounds and compounds[0].cid:
            return compounds[0].cid
    except Exception:
        pass
    return None

def show_pubchem_details(cid):
    import pubchempy as pcp
    compound = pcp.Compound.from_cid(cid)
    st.markdown(f"**PubChem CID:** {compound.cid}")
    st.image(f"https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid={compound.cid}&t=l", caption="2D Structure")
    st.image(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{compound.cid}/PNG?record_type=3d", caption="3D Structure")
    if hasattr(compound, 'safety') and compound.safety:
        st.markdown(f"**Chemical Safety:** {compound.safety}")
    st.markdown(f"**Molecular Formula:** {compound.molecular_formula}")
    if hasattr(compound, 'canonical_smiles'):
        st.markdown(f"**SMILES:** `{compound.canonical_smiles}`")
    if compound.synonyms:
        st.markdown("**Synonyms:**")
        for syn in compound.synonyms[:5]:
            st.markdown(f"- {syn}")
        if len(compound.synonyms) > 5:
            st.markdown(f"...and {len(compound.synonyms)-5} more.")
    st.markdown(f"**Molecular Weight:** {compound.molecular_weight} g/mol")
    if hasattr(compound, 'record') and compound.record:
        if 'CreateDate' in compound.record:
            st.markdown(f"**Create:** {compound.record['CreateDate']}")
        if 'ModifyDate' in compound.record:
            st.markdown(f"**Modify:** {compound.record['ModifyDate']}")
    if hasattr(compound, 'description') and compound.description:
        st.markdown(f"**Description:** {compound.description}")
    if hasattr(compound, 'source') and compound.source:
        st.markdown(f"**Source:** {compound.source}")

# Display model accuracy at the top
@st.cache_data(ttl=60)
def get_model_accuracy():
    try:
        response = requests.get("http://localhost:8000/accuracy")
        if response.ok:
            data = response.json()
            if data.get("accuracy") is not None:
                return float(data["accuracy"])
    except Exception:
        pass
    return None

accuracy = get_model_accuracy()
if accuracy is not None:
    st.info(f"**Model Validation Toxicity Accuracy:** {accuracy*100:.2f}%")
else:
    st.warning("Model accuracy not available. Please train the model.")

# Query parameters for navigation - with fallback for older Streamlit versions
try:
    query_params = st.query_params
    detail_cid = query_params.get('cid', [None])[0] if 'cid' in query_params else None
except AttributeError:
    # Fallback for older Streamlit versions
    import urllib.parse
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    
    ctx = get_script_run_ctx()
    if ctx and hasattr(ctx, 'query_string'):
        query_string = ctx.query_string
        parsed = urllib.parse.parse_qs(query_string)
        detail_cid = parsed.get('cid', [None])[0] if 'cid' in parsed else None
    else:
        detail_cid = None

if detail_cid:
    show_pubchem_details(detail_cid)
    if st.button("Back to Results"):
        try:
            st.query_params.clear()
        except AttributeError:
            # Fallback: redirect to the same page without parameters
            st.rerun()
else:
    user_input = st.text_area("Enter one or more SMILES strings or compound names (one per line):", " ")

# === AGENTIC AI SECTION ===
from typing import List
import time

def markdown_to_html(text):
    """Convert markdown formatting to HTML for display in chat bubbles."""
    import re
    # Convert **bold** to <strong>bold</strong>
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    # Convert `code` to <code>code</code>
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    # Convert newlines to <br>
    text = text.replace('\n', '<br>')
    return text

def interpret_prediction(result, user_goal=None):
    """Return a human-friendly, property-specific interpretation of the prediction result."""
    if not result or not isinstance(result, dict):
        return "No prediction result available."
    tox = result.get("toxicity")
    sol = result.get("solubility")
    intoxicant = result.get("intoxicant")
    lines = []
    # Determine which properties to show
    show_tox = show_sol = show_intox = True
    if user_goal:
        goal = user_goal.lower()
        show_tox = any(word in goal for word in ["toxicity", "toxic"])
        show_sol = "solubility" in goal or "soluble" in goal
        show_intox = "intoxicant" in goal or "intoxication" in goal
        if not (show_tox or show_sol or show_intox):
            show_tox = show_sol = show_intox = True  # fallback for generic 'predict'
    if show_tox and tox is not None:
        if tox > 0.7:
            lines.append("⚠️ The compound is predicted to be **highly toxic**.")
        elif tox > 0.5:
            lines.append("🟠 The compound is predicted to have **moderate toxicity**.")
        else:
            lines.append("🟢 The compound is predicted to be **non-toxic**.")
        lines.append(f"Toxicity score: `{tox:.2f}` (0 = safe, 1 = toxic)")
    if show_sol and sol is not None:
        if sol > 0.7:
            lines.append("💧 The compound is predicted to be **highly soluble**.")
        elif sol > 0.5:
            lines.append("🟡 The compound is predicted to have **moderate solubility**.")
        else:
            lines.append("🔴 The compound is predicted to be **poorly soluble**.")
        lines.append(f"Solubility score: `{sol:.2f}` (0 = insoluble, 1 = soluble)")
    if show_intox and intoxicant is not None:
        if intoxicant:
            lines.append("🚨 The compound is likely to be an **intoxicant**.")
        else:
            lines.append("✅ The compound is **not likely to be an intoxicant**.")
    return "\n".join(lines)

def agentic_reasoning(user_goal: str, memory: List[dict]):
    """
    More conversational, agentic reasoning with step explanations and result interpretation.
    """
    steps = []
    actions = []
    result = None
    # Example: If user asks for toxicity/solubility, plan prediction
    if any(word in user_goal.lower() for word in ["toxicity", "solubility", "predict", "intoxicant"]):
        # Extract compound name or SMILES (very basic)
        import re
        match = re.search(r"for ([\w\s\-\+\=\(\)\[\]#@\./]+)", user_goal.lower())
        compound = match.group(1).strip() if match else None
        if not compound:
            # Try to use last mentioned compound in memory
            for m in reversed(memory):
                if m.get("compound"):
                    compound = m["compound"]
                    break
        if not compound:
            steps.append("I need to know which compound you want to analyze.")
            actions.append({"type": "ask", "message": "Could you please specify the compound name or SMILES you want to predict?"})
        else:
            steps.append(f"Let me resolve '{compound}' to a SMILES string.")
            smiles, name_error = name_to_smiles(compound)
            if not smiles:
                actions.append({"type": "error", "message": name_error or f"Sorry, I couldn't resolve '{compound}' to a SMILES string."})
            else:
                steps.append(f"Now I'll call my prediction model for {smiles}.")
                try:
                    response = requests.post("http://localhost:8000/predict", json={"smiles": smiles})
                    if response.ok:
                        result = response.json()
                        interpretation = interpret_prediction(result, user_goal)
                        actions.append({
                            "type": "result",
                            "compound": compound,
                            "smiles": smiles,
                            "result": result,
                            "interpretation": interpretation
                        })
                    else:
                        actions.append({"type": "error", "message": f"Oops, the prediction API returned an error: {response.status_code} - {response.text}"})
                except Exception as e:
                    actions.append({"type": "error", "message": f"Sorry, I couldn't complete the prediction due to: {e}"})
    else:
        steps.append("I'm here to help you predict toxicity, solubility, or intoxicant properties. Just tell me what you want to analyze!")
        actions.append({"type": "ask", "message": "What would you like me to do? (e.g., 'Predict toxicity for aspirin')"})
    return steps, actions

# --- AGENTIC UI ---
# Add improved CSS for chat bubbles and scrollable area
st.markdown("""
<style>
.agentic-chat-area {
    max-height: 350px;
    overflow-y: auto;
    padding-right: 6px;
    margin-bottom: 10px;
    background: #f8eaff;
    border-radius: 16px;
    border: 1px solid #e1bee7;
    box-shadow: 0 2px 8px rgba(80,0,80,0.04);
}
.agent-bubble, .user-bubble {
    padding: 10px 12px;
    border-radius: 16px;
    margin-bottom: 8px;
    display: block;
    word-break: break-word;
    overflow-wrap: anywhere;
    white-space: pre-line;
    font-size: 14px;
    width: 100%;
    box-sizing: border-box;
    font-family: 'Segoe UI', 'Arial', sans-serif;
}
.agent-bubble {
    background: #f3e5f5;
    color: #222;
    border-radius: 16px 16px 16px 4px;
}
.user-bubble {
    background: #e3f2fd;
    color: #1a237e;
    border-radius: 16px 16px 4px 16px;
}
.agent-bubble code, .user-bubble code {
    background: #ede7f6;
    color: #4a148c;
    border-radius: 4px;
    padding: 1px 4px;
    font-size: 13px;
}
.agentic-divider {
    border-top: 1px solid #e1bee7;
    margin: 8px 0 8px 0;
}
.agentic-sidebar-content {
    max-width: 100vw;
    overflow-x: auto;
    padding-bottom: 8px;
}
.agentic-suggestion {
    background: #ede7f6;
    color: #4a148c;
    border-radius: 8px;
    padding: 4px 10px;
    margin: 2px 0 2px 0;
    display: inline-block;
    font-size: 13px;
    cursor: pointer;
    transition: background 0.2s;
}
.agentic-suggestion:hover {
    background: #d1c4e9;
}
.agent-label, .user-label {
    font-size: 12px;
    font-weight: bold;
    margin-bottom: 2px;
    margin-left: 2px;
    color: #7b1fa2;
    letter-spacing: 0.5px;
}
.user-label {
    color: #1565c0;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🤖 Agentic AI Assistant")
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = []
if "agent_chat" not in st.session_state:
    st.session_state.agent_chat = []
if "agent_typing" not in st.session_state:
    st.session_state.agent_typing = False
if "agent_last_smiles" not in st.session_state:
    st.session_state.agent_last_smiles = None
if "agent_last_query" not in st.session_state:
    st.session_state.agent_last_query = None

with st.sidebar.expander("Agentic Chat / Task Interface", expanded=True):
    # Greeting/help on first use
    if not st.session_state.agent_chat:
        st.markdown("<div style='margin-bottom:8px'><b>👋 Hi! I'm your AI agent. I can predict toxicity, solubility, or intoxicant properties for any compound. Just type your question below!</b></div>", unsafe_allow_html=True)
    # Clear chat button
    if st.button("🧹 Clear Chat", key="clear_agentic_chat"):
        st.session_state.agent_chat = []
        st.session_state.agent_memory = []
        st.session_state.agent_last_smiles = None
        st.session_state.agent_last_query = None
        try:
            st.rerun()
        except AttributeError:
            time.sleep(0.1)
            pass
    # Always show input and suggestions, even if chat is empty
    user_goal = st.text_input("Enter your goal or question:", "", key="agent_goal")
    # Suggestions based on memory
    suggestions = []
    if st.session_state.agent_memory:
        last = st.session_state.agent_memory[-1]
        suggestions.append(f"Predict toxicity for {last['compound']}")
        suggestions.append(f"Predict solubility for {last['compound']}")
        suggestions.append(f"Explain prediction for {last['compound']}")
    else:
        suggestions = ["Predict toxicity for aspirin", "Predict solubility for caffeine", "Predict intoxicant for nicotine"]
    cols = st.columns(len(suggestions))
    for i, sug in enumerate(suggestions):
        if cols[i].button(sug, key=f"sug_{i}"):
            st.session_state.agent_chat.append({"role": "user", "content": sug})
            user_goal = sug
            st.session_state.agent_typing = True
    # Main ask button
    if st.button("Ask Agent", key="agent_ask") or st.session_state.agent_typing:
        if user_goal.strip():
            st.session_state.agent_chat.append({"role": "user", "content": user_goal})
            st.session_state.agent_typing = True
            with st.spinner("Agent is thinking..."):
                steps, actions = agentic_reasoning(user_goal, st.session_state.agent_memory)
                for step in steps:
                    st.session_state.agent_chat.append({"role": "agent", "content": f"<i>{step}</i>"})
                for action in actions:
                    if action["type"] == "ask":
                        st.session_state.agent_chat.append({"role": "agent", "content": action["message"]})
                    elif action["type"] == "error":
                        st.session_state.agent_chat.append({"role": "agent", "content": f"❌ {action['message']}"})
                    elif action["type"] == "result":
                        # Split into two messages for readability
                        summary1 = f"Here's what I found for <b>{action['compound']}</b> (SMILES: <code>{action['smiles']}</code>):"
                        summary2 = action['interpretation']
                        st.session_state.agent_chat.append({"role": "agent", "content": summary1, "smiles": action['smiles'], "compound": action['compound']})
                        st.session_state.agent_chat.append({"role": "agent", "content": summary2})
                        st.session_state.agent_memory.append({"compound": action["compound"], "smiles": action["smiles"], "result": action["result"], "goal": user_goal})
                        # Store last SMILES and query for main UI sync
                        st.session_state.agent_last_smiles = action["smiles"]
                        st.session_state.agent_last_query = user_goal
                    time.sleep(0.1)
            st.session_state.agent_typing = False
    # Scrollable chat area
    st.markdown("<div class='agentic-sidebar-content'><div class='agentic-chat-area'>", unsafe_allow_html=True)
    for idx, msg in enumerate(st.session_state.agent_chat[-20:]):
        if msg["role"] == "user":
            st.markdown(f"<div class='user-label'>You</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='user-bubble'>{markdown_to_html(msg['content'])}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='agent-label'>Agent</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='agent-bubble'>{markdown_to_html(msg['content'])}</div>", unsafe_allow_html=True)
        if idx < len(st.session_state.agent_chat[-20:]) - 1:
            st.markdown("<div class='agentic-divider'></div>", unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)
    # Show memory/history
    if st.session_state.agent_memory:
        st.markdown("<hr><b>Agent Memory:</b>", unsafe_allow_html=True)
        for mem in st.session_state.agent_memory[-3:][::-1]:
            st.markdown(f"<div style='font-size:13px;margin-bottom:2px'>• <b>{mem['compound']}</b> (SMILES: <code>{mem['smiles']}</code>)<br>Last result: {interpret_prediction(mem['result'], mem.get('goal'))}</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<small>Agentic mode: I plan, act, and remember. Try: 'Predict toxicity for caffeine', 'Predict solubility for glucose', or just say 'now do glucose'.</small>", unsafe_allow_html=True)

# --- MAIN UI: Drug Toxicity Prediction (GNN) ---
# Always create the radio widget at the top, outside any if/else
view_mode = st.radio("View mode:", ["2D", "3D"], horizontal=True, key="main_view_mode")

# Sync with agent chat if a new SMILES is set
if st.session_state.get('agent_last_smiles'):
    # Auto-fill and auto-run prediction for agent's SMILES
    user_input = st.session_state['agent_last_smiles']
    st.session_state['agent_last_smiles'] = None  # Reset after use
    predict_clicked = True
else:
    
    col1, col2 = st.columns([1, 2])
    with col1:
        predict_clicked = st.button("Predict")
    with col2:
        show_accuracy = False
        if predict_clicked:
            show_accuracy = True
        if show_accuracy:
            accuracy = get_model_accuracy()
            if accuracy is not None:
                st.info(f"**Model Validation Toxicity Accuracy:** {accuracy*100:.2f}%")
            else:
                st.warning("Model accuracy not available. Please train the model.")
# Always run prediction logic if predict_clicked is True
if predict_clicked:
    queries = [q.strip() for q in user_input.splitlines() if q.strip()]
    results_table = []
    for query in queries:
        smiles, name_error = name_to_smiles(query)
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                highlight_atoms = None
                toxicity = None
                cid = get_pubchem_cid(smiles)
                with st.spinner(f"Predicting for {query}..."):
                    try:
                        response = requests.post("http://localhost:8000/predict", json={"smiles": smiles})
                        if response.ok:
                            result = response.json()
                            toxicity = result.get("toxicity")
                            solubility = result.get("solubility")
                            intoxicant = result.get("intoxicant")
                            if toxicity is not None and toxicity > 0.5:
                                highlight_atoms = list(range(mol.GetNumAtoms()))
                            results_table.append({
                                "Input": query,
                                "SMILES": smiles,
                                "Toxicity": toxicity,
                                "Solubility": solubility,
                                "Intoxicant": 'Yes' if intoxicant else 'No',
                                "CID": cid
                            })
                            with st.expander(f"Details for {query}"):
                                show_molecule(smiles, highlight_atoms=highlight_atoms, view_mode=view_mode)
                                st.markdown(f"**SMILES:** `{smiles}`")
                                if cid:
                                    st.markdown('<div class="section-header section-pubchem">PubChem Details</div>', unsafe_allow_html=True)
                                    show_pubchem_details(cid)
                                else:
                                    st.warning("No PubChem CID found for this compound.")
                                st.markdown('<div class="section-header section-tox">Prediction Explanations</div>', unsafe_allow_html=True)
                                st.markdown('<div class="section-tox">', unsafe_allow_html=True)
                                st.markdown("""
### Toxicity
- **Definition:** Toxicity is the degree to which a substance can damage an organism. In this context, it refers to the probability that the molecule is toxic to biological systems, as predicted by a Graph Neural Network (GNN) trained on the Tox21 dataset.
- **How to interpret:**
    - **Score close to 1:** High likelihood of toxicity. The molecule may cause adverse biological effects and should be handled with caution.
    - **Score close to 0:** Low likelihood of toxicity. The molecule is predicted to be relatively safe, but experimental validation is always recommended.
- **Model limitations:**
    - The prediction is based on known toxicological data and molecular structure. It does not account for all possible biological contexts, dosages, or long-term effects.
    - False positives/negatives are possible. Always consult experimental data for critical decisions.
""")
                                st.markdown('</div>', unsafe_allow_html=True)
                                st.markdown('<div class="section-header section-sol">Solubility</div>', unsafe_allow_html=True)
                                st.markdown('<div class="section-sol">', unsafe_allow_html=True)
                                st.markdown("""
- **Definition:** Solubility is the ability of a substance to dissolve in a solvent (here, water or biological fluids). The score estimates the probability that the molecule is soluble.
- **How to interpret:**
    - **Score close to 1:** The molecule is likely to dissolve well in water/biological fluids, which can affect absorption, distribution, and excretion in living organisms.
    - **Score close to 0:** The molecule is likely poorly soluble, which may limit its bioavailability or effectiveness as a drug.
- **Model limitations:**
    - The solubility score is a model estimate, not a direct measurement. Real-world solubility can be affected by pH, temperature, and other factors.
    - In this demo, the solubility score may be a placeholder and not based on experimental data.
""")
                                st.markdown('</div>', unsafe_allow_html=True)
                                st.markdown('<div class="section-header section-intox">Intoxicant</div>', unsafe_allow_html=True)
                                st.markdown('<div class="section-intox">', unsafe_allow_html=True)
                                st.markdown("""
- **Definition:** An intoxicant is a substance that can cause intoxication or adverse effects on the central nervous system. Here, it is a binary label based on the predicted toxicity score.
- **How to interpret:**
    - **Yes:** The model predicts the molecule is likely to be intoxicating (toxicity score > 0.5).
    - **No:** The molecule is predicted to be non-intoxicating (toxicity score ≤ 0.5).
- **Model limitations:**
    - This is a simplified threshold and does not reflect regulatory or clinical definitions of intoxicants.
    - Use this as a quick flag for potentially hazardous molecules, but always consult detailed toxicological and regulatory data for safety-critical applications.
""")
                                st.markdown('</div>', unsafe_allow_html=True)
                                st.markdown('<hr>', unsafe_allow_html=True)
                        else:
                            st.error(f"API error: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Request failed: {e}")
            else:
                st.error(f"Invalid SMILES string for '{query}'.")
        else:
            st.error(name_error or f"Invalid input: {query}")
    if results_table:
        st.markdown("## Summary Table")
        df = pd.DataFrame(results_table)
        # Color formatting for toxicity and solubility
        def color_toxicity(val):
            if pd.isnull(val):
                return ''
            if val > 0.7:
                return 'background-color: #ffcccc; color: #b30000'  # Red
            elif val > 0.5:
                return 'background-color: #fff2cc; color: #b36b00'  # Orange
            else:
                return 'background-color: #ccffcc; color: #006600'  # Green
        def color_solubility(val):
            if pd.isnull(val):
                return ''
            if val > 0.7:
                return 'background-color: #ccffcc; color: #006600'  # Green
            elif val > 0.5:
                return 'background-color: #fff2cc; color: #b36b00'  # Orange
            else:
                return 'background-color: #ffcccc; color: #b30000'  # Red
        styled_df = df.style.applymap(color_toxicity, subset=['Toxicity']) \
                          .applymap(color_solubility, subset=['Solubility'])
        st.dataframe(styled_df, use_container_width=True)
        # --- Add charts ---
        if 'Toxicity' in df and 'Solubility' in df:
            st.markdown("### 📈 Toxicity & Solubility Distribution")
            x = np.arange(len(df))
            width = 0.35
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            # Glossy gradient colors matching About & Workflow section
            from matplotlib.patches import Rectangle
            def glossy_bar(ax, x, height, width, color1, color2, label=None):
                for i, h in enumerate(height):
                    bar = Rectangle((x[i] - width/2, 0), width, h, color=color1, zorder=2, label=label if i==0 else "")
                    ax.add_patch(bar)
                    # Overlay a glossy gradient
                    ax.add_patch(Rectangle((x[i] - width/2, 0), width, h, zorder=3, 
                        facecolor=color2, alpha=0.35, label=None))
            # Toxicity: purple gradient (#6a11cb to #2575fc)
            glossy_bar(ax1, x - width/2, df['Toxicity'], width, '#6a11cb', '#2575fc', label='Toxicity')
            # Solubility: green-cyan gradient (#43e97b to #38f9d7)
            glossy_bar(ax1, x + width/2, df['Solubility'], width, '#43e97b', '#38f9d7', label='Solubility')
            ax1.set_xlim(-1, len(df))
            ax1.set_xlabel('Compound Index')
            ax1.set_ylabel('Score')
            ax1.set_title('Toxicity & Solubility Distribution', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels([str(i+1) for i in x], rotation=0)
            ax1.legend(['Toxicity', 'Solubility'], loc='upper right')
            ax1.grid(axis='y', linestyle='--', alpha=0.4, zorder=1)
            for i, h in enumerate(df['Toxicity']):
                ax1.text(x[i] - width/2, h + 0.01, f'{h:.2f}', ha='center', va='bottom', fontsize=9, color='#6a11cb', fontweight='bold')
            for i, h in enumerate(df['Solubility']):
                ax1.text(x[i] + width/2, h + 0.01, f'{h:.2f}', ha='center', va='bottom', fontsize=9, color='#43e97b', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig1)
        if 'Intoxicant' in df:
            st.markdown("### 🧪 Intoxicant Class Distribution")
            intoxicant_counts = df['Intoxicant'].value_counts()
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            colors = ['#1976d2', '#e53935'] if 'Yes' in intoxicant_counts.index else ['#43a047', '#e53935']
            wedges, texts, autotexts = ax2.pie(
                intoxicant_counts,
                labels=intoxicant_counts.index,
                autopct='%1.0f%%',
                startangle=90,
                colors=colors,
                wedgeprops=dict(width=0.4, edgecolor='w'),
                textprops={'fontsize': 12}
            )
            ax2.set_title('Intoxicant Distribution', fontsize=13, fontweight='bold')
            ax2.legend(wedges, intoxicant_counts.index, title="Intoxicant", loc="center left", bbox_to_anchor=(1, 0.5))
            plt.setp(autotexts, size=13, weight="bold", color='white')
            ax2.axis('equal')
            st.pyplot(fig2)
