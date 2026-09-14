import os
import sys
import ctypes

# 1. Allow duplicate OpenMP DLLs & Bypass CUDA initialization
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 2. Add site-packages/torch/lib explicitly to Windows DLL directory search path
torch_lib_dir = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
if os.path.exists(torch_lib_dir) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(torch_lib_dir)
    except Exception:
        pass

# 3. Preload C++ runtime DLLs
for dll_name in ["vcruntime140.dll", "msvcp140.dll", "vcruntime140_1.dll"]:
    try:
        ctypes.CDLL(dll_name)
    except Exception:
        pass

import re
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st

# Try importing RDKit for 2D Molecule rendering and property calculation
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, Draw
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="AI Molecular Discovery Platform",
    page_icon="https://unicons.iconscout.com/release/v4.0.8/svg/line/molecule.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load IconScout Unicons CSS & Custom Glassmorphism UI
st.markdown("""
<link rel="stylesheet" href="https://unicons.iconscout.com/release/v4.0.8/css/line.css">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Container Styling */
    .stApp {
        background-color: #090D16;
        color: #F1F5F9;
    }

    /* Header Banner */
    .header-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 28px;
        backdrop-filter: blur(12px);
    }

    .main-title {
        font-size: 30px;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .sub-title {
        font-size: 14px;
        color: #94A3B8;
        font-weight: 400;
    }

    /* Status Banners */
    .status-box {
        display: flex;
        align-items: center;
        gap: 10px;
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.25);
        color: #34D399;
        padding: 12px 18px;
        border-radius: 10px;
        font-size: 13.5px;
        font-weight: 500;
        margin-bottom: 24px;
    }

    /* Card Box */
    .mol-card-box {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .mol-card-box:hover {
        border-color: #3B82F6;
    }

    /* Badges */
    .badge-compliant {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .badge-non-compliant {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* Property Metric Cards */
    .prop-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 10px 6px;
        text-align: center;
    }

    .prop-icon {
        font-size: 14px;
        color: #38BDF8;
        margin-bottom: 2px;
    }

    .prop-label-text {
        font-size: 10px;
        color: #94A3B8;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .prop-val-text {
        font-size: 14px;
        color: #F8FAFC;
        font-weight: 700;
    }

    /* Sidebar Styling */
    div[data-testid="stSidebar"] {
        background-color: #0B0F19;
        border-right: 1px solid #1E293B;
    }
    
    .sidebar-header {
        font-size: 14px;
        font-weight: 700;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 1. SMILES TOKENIZER & MODEL CLASSES
# ==============================================================================
SMILES_REGEX_PATTERN = r"(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\=|\#|\-|\+|\;|\:|\/|\\|\@|\.|\%[0-9]{2}|[0-9])"

class SMILESTokenizer:
    def __init__(self):
        self.pad_token = "<PAD>"
        self.start_token = "<START>"
        self.end_token = "<END>"
        self.unk_token = "<UNK>"
        self.special_tokens = [self.pad_token, self.start_token, self.end_token, self.unk_token]
        self.token_to_idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.idx_to_token = {idx: token for idx, token in enumerate(self.special_tokens)}
        
    def tokenize(self, smiles_str):
        return re.findall(SMILES_REGEX_PATTERN, smiles_str)
        
    def decode(self, indices, clean_special=True):
        tokens = []
        for idx in indices:
            token = self.idx_to_token.get(idx, "")
            if clean_special and token in self.special_tokens:
                if token == self.end_token:
                    break
                continue
            tokens.append(token)
        return "".join(tokens)

    @property
    def vocab_size(self):
        return len(self.token_to_idx)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            token_to_idx = pickle.load(f)
        tok = cls()
        tok.token_to_idx = token_to_idx
        tok.idx_to_token = {v: k for k, v in token_to_idx.items()}
        return tok

class ConditionalLSTMVAE(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, latent_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        
        self.fc_h0 = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.fc_c0 = nn.Linear(latent_dim, hidden_dim * num_layers)
        
        self.decoder = nn.LSTM(embed_dim + latent_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.property_predictor = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 3))


# ==============================================================================
# 2. MODEL LOADING & INFERENCE ENGINE
# ==============================================================================
@st.cache_resource
def load_trained_model():
    model_path = "best_lstm_vae.pt"
    tokenizer_path = "tokenizer.pkl"
    
    if not (os.path.exists(model_path) and os.path.exists(tokenizer_path)):
        return None, None
        
    tokenizer = SMILESTokenizer.load(tokenizer_path)
    model = ConditionalLSTMVAE(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, tokenizer

def generate_smiles(model, tokenizer, num_samples=10, temperature=0.8, top_p=0.9):
    start_idx = tokenizer.token_to_idx[tokenizer.start_token]
    end_idx = tokenizer.token_to_idx[tokenizer.end_token]
    
    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim, device="cpu")
        h = model.fc_h0(z).view(model.num_layers, num_samples, model.hidden_dim).contiguous()
        c = model.fc_c0(z).view(model.num_layers, num_samples, model.hidden_dim).contiguous()
        
        curr_token = torch.full((num_samples, 1), start_idx, dtype=torch.long, device="cpu")
        active_mask = torch.ones(num_samples, dtype=torch.bool, device="cpu")
        generated_indices = [[] for _ in range(num_samples)]
        
        for _ in range(100):
            if not active_mask.any():
                break
            emb = model.embedding(curr_token)
            dec_in = torch.cat([emb, z.unsqueeze(1)], dim=-1)
            
            out, (h, c) = model.decoder(dec_in, (h, c))
            logits = model.output_layer(out[:, -1, :]) / temperature
            
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            for i in range(num_samples):
                if active_mask[i]:
                    idx_val = next_token[i].item()
                    if idx_val == end_idx:
                        active_mask[i] = False
                    else:
                        generated_indices[i].append(idx_val)
            curr_token = next_token

    return [tokenizer.decode(indices) for indices in generated_indices]


# ==============================================================================
# 3. STREAMLIT WEB APP UI
# ==============================================================================

# Header Section
st.markdown("""
<div class="header-box">
    <div class="main-title"><i class="uil uil-molecule"></i> De Novo Molecular Generation</div>
    <div class="sub-title">LSTM-Based Variational Autoencoder with Bidirectional Encoder</div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configurations with IconScout Unicons
st.sidebar.markdown('<div class="sidebar-header"><i class="uil uil-sliders-v-alt"></i> Generation Controls</div>', unsafe_allow_html=True)

num_molecules = st.sidebar.slider("Number of Molecules", min_value=1, max_value=50, value=10, step=1)
temperature = st.sidebar.slider("Sampling Temperature (T)", min_value=0.5, max_value=1.5, value=0.8, step=0.05, help="Controls latent space sampling stochasticity.")
top_p = st.sidebar.slider("Nucleus Sampling (Top-p)", min_value=0.70, max_value=0.99, value=0.90, step=0.01)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-header"><i class="uil uil-filter"></i> Molecular Filters & Rules</div>', unsafe_allow_html=True)
lipinski_filter = st.sidebar.checkbox("Enforce Lipinski's Rule of Five", value=True)

# Model Status Banner
model, tokenizer = load_trained_model()

if model is None:
    st.markdown('<div class="status-box" style="color: #EF4444; border-color: rgba(239, 68, 68, 0.3); background: rgba(239, 68, 68, 0.08);"><i class="uil uil-exclamation-triangle"></i> Model weights `best_lstm_vae.pt` or `tokenizer.pkl` not found in root path.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-box"><i class="uil uil-check-circle"></i> PyTorch LSTM-VAE Architecture Active | Device: CPU | Status: Ready</div>', unsafe_allow_html=True)
    
    # Generate Action Button
    if st.button("Generate Molecular Structures", type="primary", use_container_width=True):
        with st.spinner("Sampling continuous latent space z ~ N(0, I) and generating novel SMILES..."):
            raw_smiles_list = generate_smiles(model, tokenizer, num_samples=num_molecules * 2 if lipinski_filter else num_molecules, temperature=temperature, top_p=top_p)
            
            processed_data = []
            for idx, smiles in enumerate(raw_smiles_list):
                mol = Chem.MolFromSmiles(smiles) if RDKIT_AVAILABLE else None
                is_valid = mol is not None
                
                if is_valid:
                    mw = round(Descriptors.MolWt(mol), 2)
                    logp = round(Descriptors.MolLogP(mol), 2)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)
                    qed_score = round(QED.qed(mol), 3)
                    
                    violations = 0
                    if mw > 500: violations += 1
                    if logp > 5: violations += 1
                    if hbd > 5: violations += 1
                    if hba > 10: violations += 1
                    
                    is_druglike = (violations <= 1)
                else:
                    mw, logp, hbd, hba, violations, qed_score = None, None, None, None, 4, None
                    is_druglike = False
                    
                if lipinski_filter and not is_druglike:
                    continue
                    
                processed_data.append({
                    "SMILES": smiles,
                    "Mol": mol,
                    "Valid": is_valid,
                    "MW": mw,
                    "LogP": logp,
                    "HBD": hbd,
                    "HBA": hba,
                    "Violations": violations,
                    "DrugLike": is_druglike,
                    "QED": qed_score
                })
                
                if len(processed_data) >= num_molecules:
                    break
                    
        st.markdown(f"#### Generated Molecules({len(processed_data)} Results)")
        
        # Display Cards with IconScout Vector Icons
        cols_per_row = 2
        for i in range(0, len(processed_data), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(processed_data):
                    item = processed_data[i + j]
                    with cols[j]:
                        with st.container():
                            status_badge = '<span class="badge-compliant"><i class="uil uil-shield-check"></i> LIPINSKI COMPLIANT</span>' if item["DrugLike"] else '<span class="badge-non-compliant"><i class="uil uil-times-circle"></i> NON-COMPLIANT</span>'
                            
                            st.markdown(f"""
                            <div class="mol-card-box">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <span style="font-weight: 700; color: #F8FAFC; font-size: 15px;"><i class="uil uil-flask" style="color: #38BDF8;"></i> Molecule #{i+j+1}</span>
                                    {status_badge}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.code(item["SMILES"], language="text")
                            
                            # Render 2D Chemical Diagram
                            if RDKIT_AVAILABLE and item["Mol"] is not None:
                                img = Draw.MolToImage(item["Mol"], size=(400, 220))
                                st.image(img, use_container_width=True)
                                
                            col_a, col_b, col_c, col_d, col_e = st.columns(5)
                            col_a.markdown(f'<div class="prop-card"><div class="prop-icon"><i class="uil uil-weight"></i></div><div class="prop-label-text">MW</div><div class="prop-val-text">{item["MW"]}</div></div>', unsafe_allow_html=True)
                            col_b.markdown(f'<div class="prop-card"><div class="prop-icon"><i class="uil uil-tear"></i></div><div class="prop-label-text">LOGP</div><div class="prop-val-text">{item["LogP"]}</div></div>', unsafe_allow_html=True)
                            col_c.markdown(f'<div class="prop-card"><div class="prop-icon"><i class="uil uil-atom"></i></div><div class="prop-label-text">HBD</div><div class="prop-val-text">{item["HBD"]}</div></div>', unsafe_allow_html=True)
                            col_d.markdown(f'<div class="prop-card"><div class="prop-icon"><i class="uil uil-layers"></i></div><div class="prop-label-text">HBA</div><div class="prop-val-text">{item["HBA"]}</div></div>', unsafe_allow_html=True)
                            col_e.markdown(f'<div class="prop-card"><div class="prop-icon"><i class="uil uil-award"></i></div><div class="prop-label-text">QED</div><div class="prop-val-text">{item["QED"]}</div></div>', unsafe_allow_html=True)
                            
                            st.markdown("<br>", unsafe_allow_html=True)

        # Export CSV Data
        df_export = pd.DataFrame([{
            "Molecule_ID": f"MOL_{idx+1:03d}",
            "SMILES": d["SMILES"],
            "Is_Valid": d["Valid"],
            "Molecular_Weight": d["MW"],
            "LogP": d["LogP"],
            "HBD": d["HBD"],
            "HBA": d["HBA"],
            "Lipinski_Violations": d["Violations"],
            "Is_Drug_Like": d["DrugLike"],
            "QED_Score": d["QED"]
        } for idx, d in enumerate(processed_data)])
        
        csv_bytes = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Generated Candidates (CSV)",
            data=csv_bytes,
            file_name="generated_molecular_candidates.csv",
            mime="text/csv",
            type="secondary"
        )
