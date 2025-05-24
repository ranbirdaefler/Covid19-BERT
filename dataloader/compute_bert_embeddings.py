
# ------------------------------------------------------------------ #
# 1  Paths (EDIT THESE TO POINT TO THE FOLDER CONTAINING THE .FASTA FILES FOR DATA_DIR AND THE REFERENCE WUHAN VIRUS PATH FOR REF_FP)
# ------------------------------------------------------------------ #
DATA_DIR = r"C:\Users\avsd8\OneDrive\Desktop\covid19\project\data"
REF_FP   = r"C:\Users\avsd8\OneDrive\Desktop\covid19\project\data\reference.fasta"

# ------------------------------------------------------------------ #
# 2  Imports
# ------------------------------------------------------------------ #
import os, sys, numpy as np, torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from loader import SpikeMultiDataLoaderWithReference

# ------------------------------------------------------------------ #
# 3  Check GPU
# ------------------------------------------------------------------ #
if not torch.cuda.is_available():
    sys.exit("❌  No CUDA device detected.  Aborting re-embedding.")

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(device)
print(f"✅  Using GPU: {gpu_name}")

# heuristically pick batch size
vram_gb = torch.cuda.get_device_properties(device).total_memory / 2**30
BATCH = 64 if vram_gb >= 24 else 32 if vram_gb >= 12 else 16
print(f"Batch size set to {BATCH} (VRAM ≈ {vram_gb:.1f} GiB)")

# ------------------------------------------------------------------ #
# 4  Load sequences in desired order
# ------------------------------------------------------------------ #
loader = SpikeMultiDataLoaderWithReference(DATA_DIR, REF_FP)
df = loader.load_all_data()                # preserves original order
ref_seq = df.ReferenceSequence.iloc[0]
N = len(df)
print(f"Total sequences: {N:,}")

# ------------------------------------------------------------------ #
# 5  Init ProtBERT
# ------------------------------------------------------------------ #
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert").to(device)
model.eval()

hidden = model.config.hidden_size   
print(f"Hidden size = {hidden}")
emb = np.empty((N, hidden), dtype=np.float32)

def prep(seq): return " ".join(list(seq[:1273]))  # truncate >1273 aa
# 6  Embed
# ------------------------------------------------------------------ #


with torch.no_grad():
    for start in tqdm(range(0, N, BATCH), desc="Embedding"):
        end = min(start + BATCH, N)

        var_tok = [prep(s) for s in df.Sequence.iloc[start:end]]
        ref_tok = [prep(ref_seq)] * len(var_tok)

        tok_v = tokenizer(var_tok, return_tensors="pt", padding=True).to(device)
        tok_r = tokenizer(ref_tok, return_tensors="pt", padding=True).to(device)

        cls_v = model(**tok_v).last_hidden_state[:, 0, :]
        cls_r = model(**tok_r).last_hidden_state[:, 0, :]

        batch_delta = (cls_v - cls_r).float().cpu().numpy()   # ensure FP32
        emb[start : start + batch_delta.shape[0]] = batch_delta

# 7  Save aligned embeddings + labels
# ------------------------------------------------------------------ #
np.save("bert_full_X.npy", emb.astype(np.float32))
np.save("bert_full_y.npy", df.Label.to_numpy().astype("U"))
print("✅  Saved bert_full_X.npy & bert_full_y.npy")
