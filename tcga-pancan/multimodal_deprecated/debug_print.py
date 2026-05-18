# =======================
# DEBUG HELPERS (ADJ/BATCH)
# =======================
import numpy as np
import scipy.sparse as sp
import torch

DEBUG_ADJ = True
DEBUG_BATCH = True

def _sparse_min_max_including_zeros(A: sp.spmatrix):
    """
    Ritorna (min_global, max_global) includendo gli zeri impliciti.
    Nota: per matrici molto sparse, min_global è quasi sempre 0.0.
    """
    # max globale
    max_global = float(A.max()) if A.nnz > 0 else 0.0
    # se non è completamente densa, il min globale include zeri impliciti -> 0.0
    total = A.shape[0] * A.shape[1]
    if A.nnz < total:
        min_global = 0.0
    else:
        # matrice completamente densa: il min è il min dei data
        min_global = float(A.data.min()) if A.nnz > 0 else 0.0
    return min_global, max_global

def debug_sparse_matrix(name: str, A: sp.spmatrix, max_show_unique: int = 10):
    A = A.tocsr()
    n, m = A.shape
    nnz = int(A.nnz)
    total = int(n * m)
    density = nnz / total if total else 0.0

    diag = A.diagonal()
    diag_nnz = int(np.count_nonzero(diag))
    offdiag_nnz = nnz - diag_nnz

    if nnz > 0:
        data_min_nz = float(A.data.min())
        data_max_nz = float(A.data.max())
        uniq = np.unique(A.data)
        uniq_preview = uniq[:max_show_unique]
    else:
        data_min_nz = None
        data_max_nz = None
        uniq_preview = []

    min_global, max_global = _sparse_min_max_including_zeros(A)

    # mismatch simmetria (su nnz piccolo è economico)
    try:
        sym_mismatch = int((A != A.T).nnz)
    except Exception:
        sym_mismatch = -1

    print(f"\n=== {name} ===")
    print(f"shape={A.shape} nnz={nnz} density={density:.6e}")
    print(f"nonzero_data_min={data_min_nz} nonzero_data_max={data_max_nz} unique(nonzero)[:{max_show_unique}]={uniq_preview}")
    print(f"global_min_including_zeros={min_global} global_max_including_zeros={max_global}")
    print(f"diag_nnz={diag_nnz} offdiag_nnz={offdiag_nnz}")
    print(f"symmetry_mismatch_nnz={sym_mismatch} (0 => perfettamente simmetrica)")

def debug_supra_blocks(A: sp.spmatrix, num_gene: int, num_mirna: int):
    """
    Stampa i nnz per blocchi:
      [ GG  GM ]
      [ MG  MM ]
    dove GM=gene->mirna, MG=mirna->gene
    """
    A = A.tocsr()
    G = num_gene
    M = num_mirna
    assert A.shape[0] == G + M and A.shape[1] == G + M, f"Supra shape inattesa: {A.shape} vs atteso {(G+M, G+M)}"

    GG = A[:G, :G]
    GM = A[:G, G:G+M]
    MG = A[G:G+M, :G]
    MM = A[G:G+M, G:G+M]

    print("\n--- Supra blocks nnz ---")
    print(f"nnz gene-gene (GG): {GG.nnz}")
    print(f"nnz gene->mirna (GM): {GM.nnz}")
    print(f"nnz mirna->gene (MG): {MG.nnz}")
    print(f"nnz mirna-mirna (MM): {MM.nnz}")
    # check coerenza tipica: MG dovrebbe essere transpose di GM
    mismatch_blocks = int(((GM != MG.T).nnz)) if (GM.nnz + MG.nnz) > 0 else 0
    print(f"mismatch GM vs (MG)^T nnz: {mismatch_blocks} (0 => coerenti)")

def debug_edge_index(name: str, edge_index: torch.Tensor, num_nodes: int, edge_weight: torch.Tensor = None):
    ei = edge_index
    print(f"\n=== {name} ===")
    print(f"edge_index shape={tuple(ei.shape)} E={ei.size(1)} min={int(ei.min())} max={int(ei.max())} num_nodes={num_nodes}")
    if edge_weight is not None:
        ew = edge_weight
        print(f"edge_weight shape={tuple(ew.shape)} min={float(ew.min())} max={float(ew.max())}")
        try:
            u = torch.unique(ew)
            print(f"edge_weight unique count={u.numel()} preview={u[:10].detach().cpu().numpy()}")
        except Exception:
            pass

def debug_batch_edges(batch, N: int):
    """
    Verifica che nessun arco colleghi nodi di grafi diversi nel batch.
    N = num_nodes per grafo
    """
    src = batch.edge_index[0]
    dst = batch.edge_index[1]
    src_g = src // N
    dst_g = dst // N
    cross = (src_g != dst_g).sum().item()
    print(f"[DEBUG BATCH] cross-graph edges: {cross}/{batch.edge_index.size(1)} (atteso: 0)")
    if hasattr(batch, "edge_weight") and batch.edge_weight is not None:
        print(f"[DEBUG BATCH] edge_weight len: {batch.edge_weight.numel()} vs edges: {batch.edge_index.size(1)}")
