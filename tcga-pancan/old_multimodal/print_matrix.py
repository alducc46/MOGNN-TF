import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_confusion_matrix_table_style(
    cm_counts: np.ndarray,
    class_names: list[str],
    out_png_path: str,
    normalize: bool = True,
):
    """
    Plot confusion matrix in "table style" like the provided screenshot:
      - x labels on top, not rotated
      - black grid
      - zeros stay white
      - annotations in every cell (0 or value)
      - SUM column at right (row totals from counts)
    If normalize=True, colors/values are normalized per-row (recall-style).
    """
    cm_counts = np.asarray(cm_counts)
    n = len(class_names)

    # Row sums from counts (for SUM column)
    row_sums = cm_counts.sum(axis=1).astype(int)

    # Values for heatmap (counts or normalized)
    if normalize:
        denom = cm_counts.sum(axis=1, keepdims=True)
        cm_vals = np.divide(cm_counts, denom, out=np.zeros_like(cm_counts, dtype=float), where=denom != 0)
        fmt = "float"
    else:
        cm_vals = cm_counts.astype(float)
        fmt = "int"

    # Build extended matrix with an extra SUM column (kept as NaN -> stays white)
    ext = np.full((n, n + 1), np.nan, dtype=float)
    ext[:, :n] = cm_vals

    # Mask zeros and NaNs to keep background white like the screenshot
    masked = np.ma.masked_where((ext == 0) | np.isnan(ext), ext)

    # Colormap similar to screenshot (yellow-green-blue); NaNs/zeros are white
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad(color="white")

    vmax = masked.max() if masked.count() else 1.0

    # Figure size: adjust if you have many classes
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest", aspect="auto")

    # Ticks/labels (x labels on TOP, not rotated)
    ax.set_xticks(np.arange(n + 1))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names + ["SUM"], rotation=0, fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    ax.set_ylabel("Ground-truth", fontsize=11)

    # Black grid lines (cell borders)
    ax.set_xticks(np.arange(-0.5, n + 1, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate every cell + SUM column
    for i in range(n):
        for j in range(n):
            v = cm_vals[i, j]
            if fmt == "int":
                text = str(int(v))
            else:
                text = "0" if v == 0 else f"{v:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color="black")
            thr = 0.6  # soglia (per normalized). Per counts usa una soglia diversa.
            txt_color = "white" if v > thr else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=txt_color)

        # SUM column: always from counts
        ax.text(n, i, str(int(row_sums[i])), ha="center", va="center", fontsize=8, color="black")

    # Thicker outer border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("black")

    plt.tight_layout()
    fig.savefig(out_png_path, dpi=200)
    plt.close(fig)


def find_all_result_folders():
    """Trova tutti i folder dei risultati con confusion matrix CSV"""
    from pathlib import Path
    
    workspace_root = Path(__file__).parent
    result_patterns = ["result/*", "result_*/*", "result_*"]
    
    all_folders = []
    for pattern in result_patterns:
        all_folders.extend(workspace_root.glob(pattern))
    
    # Filtra solo le directory che hanno il file CSV della confusion matrix
    valid_folders = []
    for folder in all_folders:
        if folder.is_dir():
            val_csv = folder / "val_confusion_matrix_cm.csv"
            train_csv = folder / "confusion_matrix_cm.csv"
            # Anche vecchio nome
            old_csv = folder / "confusion_matrix_counts.csv"
            if val_csv.exists() or train_csv.exists() or old_csv.exists():
                valid_folders.append(folder)
    
    return sorted(valid_folders)


def process_folder(folder_path):
    """Processa un singolo folder generando le confusion matrix"""
    from pathlib import Path
    
    print(f"\n📁 {folder_path.name}")
    
    csv_files = [
        ("val_confusion_matrix_cm.csv", "val"),
        ("confusion_matrix_cm.csv", "train"),
        ("confusion_matrix_counts.csv", "confusion")
    ]
    
    generated = []
    
    for csv_name, prefix in csv_files:
        csv_path = folder_path / csv_name
        if not csv_path.exists():
            continue
        
        try:
            counts_df = pd.read_csv(csv_path, index_col=0)
            
            # Rimuovi colonna SUM se presente
            if 'SUM' in counts_df.columns:
                counts_df = counts_df.drop(columns=['SUM'])
            
            class_names = counts_df.columns.tolist()
            cm_counts = counts_df.values
            
            # Genera versione normalizzata
            out_norm = folder_path / f"{prefix}_confusion_matrix_normalized_table.png"
            plot_confusion_matrix_table_style(
                cm_counts=cm_counts,
                class_names=class_names,
                out_png_path=str(out_norm),
                normalize=True
            )
            generated.append(out_norm.name)
            
            # Genera versione counts
            out_counts = folder_path / f"{prefix}_confusion_matrix_counts_table.png"
            plot_confusion_matrix_table_style(
                cm_counts=cm_counts,
                class_names=class_names,
                out_png_path=str(out_counts),
                normalize=False
            )
            generated.append(out_counts.name)
            
        except Exception as e:
            print(f"   ❌ Error processing {csv_name}: {e}")
    
    if generated:
        print(f"   ✅ Generated: {len(generated)} images")
    
    return len(generated)


# ------------------ ESEMPIO USO ------------------
if __name__ == "__main__":
    print("🔍 Searching for result folders with confusion matrices...")
    folders = find_all_result_folders()
    
    if not folders:
        print("❌ No result folders found!")
    else:
        print(f"\n📊 Found {len(folders)} folders to process")
        print("="*80)
        
        total_images = 0
        success_count = 0
        
        for i, folder in enumerate(folders, 1):
            print(f"[{i}/{len(folders)}]", end=" ")
            try:
                num_generated = process_folder(folder)
                if num_generated > 0:
                    success_count += 1
                    total_images += num_generated
            except Exception as e:
                print(f"\n   ❌ Unexpected error: {e}")
        
        print("\n" + "="*80)
        print(f"🎉 COMPLETED! Processed: {success_count}/{len(folders)} folders")
        print(f"   Generated: {total_images} images")
        print("="*80)
