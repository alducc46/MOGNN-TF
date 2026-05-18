"""
Main script for TCGA-PANCAN data pipeline
This script executes the complete data preparation pipeline before GNN training.

Pipeline steps:
1. Download raw TCGA-PANCAN data from Xena
2. Generate prior knowledge networks (GGI and miRNA-gene interactions)
3. Process omics data (CNV, expression, miRNA)
4. Prepare data for Multimodal GNN training

Author: Laterza
Date: December 2025
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def run_script(script_path, description, cwd=None):
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP: {description}")
    logger.info(f"Executing: {script_path}")
    logger.info(f"{'='*80}\n")

    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", script_path],  # -u = unbuffered -> print live
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # unifica stderr in stdout
            encoding="utf-8",   # <-- fondamentale: decodifica output figlio in UTF-8
            errors="replace",   # <-- evita crash su caratteri strani
            text=True,
            bufsize=1
        )

        # leggi riga per riga e “tee” su terminale + logger
        for line in proc.stdout:
            line = line.rstrip("\n")
            #print(line, flush=True)     # live terminal
            logger.info(line)           # log file + console handler

        return_code = proc.wait()
        if return_code == 0:
            logger.info(f"SUCCESS: {description} completed successfully\n")
            return True
        else:
            logger.error(f"!! Error in {description}: Return code {return_code}")
            return False

    except Exception as e:
        logger.error(f"!! Unexpected error in {description}: {str(e)}")
        return False


def check_file_exists(file_path, description):
    """Check if a required file exists."""
    if os.path.exists(file_path):
        logger.info(f"Found: {description} ({file_path})")
        return True
    else:
        logger.warning(f"!! Missing: {description} ({file_path})")
        return False


def main():
    """Main pipeline execution function."""
    logger.info("\n" + "="*80)
    logger.info("TCGA-PANCAN DATA PROCESSING PIPELINE")
    logger.info("="*80 + "\n")
    
    # Get project root directory
    project_root = Path(__file__).parent.parent.absolute()
    tcga_pancan_dir = Path(__file__).parent.absolute()
    preprocessing_dir = tcga_pancan_dir / "preprocessing"
    scripts_dir = tcga_pancan_dir / "scripts"
    datasets_dir = project_root / "datasets"
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"TCGA-PANCAN directory: {tcga_pancan_dir}")
    logger.info(f"Datasets directory: {datasets_dir}")
    
    # Check if required directories exist
    if not datasets_dir.exists():
        logger.error(f"Datasets directory not found: {datasets_dir}")
        return False
    
    # Pipeline execution
    success = True
    
    # Step 1: Download TCGA-PANCAN data
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: DATA DOWNLOAD")
    logger.info("="*80)

    
    get_pancan_script = scripts_dir / "download_pancan.py"

    # check if data already downloaded
    pancan_data_dir = tcga_pancan_dir / "data" / "pancan_tcga" / "raw"
    if pancan_data_dir.exists() and any(pancan_data_dir.iterdir()):
        logger.info(f"Found: TCGA-PANCAN raw data in {pancan_data_dir}, skipping download step.")
    else:
        logger.info("TCGA-PANCAN raw data not found, downloading...")
        if not run_script(str(get_pancan_script), 
                        "Download TCGA-PANCAN raw data from Xena", 
                        cwd=str(tcga_pancan_dir)):
            logger.error("Data download failed. Stopping pipeline.")
            return False
        
    # Step 2: Generate prior knowledge networks (GGI and miRNA interactions)

    logger.info("\n" + "="*80)
    logger.info("PHASE 2: PRIOR KNOWLEDGE NETWORK GENERATION")
    logger.info("="*80)
    
    # Step 2.1: Download raw interaction data
    get_raw_data_script = datasets_dir / "get_raw_data.py"
    # check if output files already exist
    prior_knowledge_dir = datasets_dir / "data" / "prior_knowledge"
    
    if prior_knowledge_dir.exists() and any(prior_knowledge_dir.iterdir()):
        logger.info(f"Found: Prior knowledge data in {prior_knowledge_dir}, skipping download step.")
    else:
        logger.info("Prior knowledge data not found, downloading...")
        if not run_script(str(get_raw_data_script),
                        "Download raw interaction data (GGI and miRNA)",
                        cwd=str(datasets_dir)):
            logger.error("Raw data download failed. Stopping pipeline.")
            return False
    
    # Step 2.2: Generate RefSeq to gene mappings
    refseq2gene_script = datasets_dir / "refseq2gene.py"
    # check if output file already exists
    refseq2gene_output = datasets_dir / "refseq2gene_mappings.tsv"
    logger.info(f"checking for {refseq2gene_output}")
    if refseq2gene_output.exists():
        logger.info(f"Found: RefSeq to gene symbol mappings ({refseq2gene_output})")
    else:
        logger.info("not found, generating...")
        if not run_script(str(refseq2gene_script),
                         "Generate RefSeq to gene symbol mappings",
                         cwd=str(datasets_dir)):
            logger.error("RefSeq to gene mapping failed. Stopping pipeline.")
            return False
    
    # Step 2.3: Load and process interaction networks
    load_interaction_script = datasets_dir / "load_interaction.py"
    interactions_files = datasets_dir / "data" / "prior_knowledge" / "GGI"/ "GGI_edges_undirected.csv"
    if interactions_files.exists():
        logger.info(f"Found: Processed prior knowledge networks in {interactions_files}, skipping generation step.")
    else:
        if not run_script(str(load_interaction_script),
                         "Generate GGI and miRNA-gene interaction networks",
                         cwd=str(datasets_dir)):
            logger.error("Prior knowledge generation failed. Stopping pipeline.")
            return False
    
    # Step 3: Process omics data
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: OMICS DATA PROCESSING")
    logger.info("="*80)
    
    logger.info("preparing omics data (CNV, expression, miRNA)...")

    prepare_pancan_script = preprocessing_dir / "prepare_pancan.py"
    # check if output files already exist
    prepare_pancan_output = tcga_pancan_dir / "data" / "pancan_tcga" / "processed"
    if prepare_pancan_output.exists() and any(prepare_pancan_output.iterdir()):
        logger.info(f"Found: Processed omics data in {prepare_pancan_output}, skipping preparation step.")
    else:
        if not run_script(str(prepare_pancan_script),
                         "Prepare omics data (CNV, expression, miRNA)",
                         cwd=str(tcga_pancan_dir)):
            logger.error("Omics data preparation failed. Stopping pipeline.")
        return False

    process_data_script = preprocessing_dir / "process_data_pancan.py"
    omics_output = tcga_pancan_dir / "data" / "omics"
    if omics_output.exists() and any(omics_output.iterdir()):
        logger.info(f"Found: Processed omics data in {omics_output}, skipping processing step.")
    else:
        if not run_script(str(process_data_script),
                        "Process omics data (CNV, expression, miRNA)",
                        cwd=str(tcga_pancan_dir)):
            logger.error("Omics data processing failed. Stopping pipeline.")
            return False
        
    # Step 4: Prepare data for Multimodal GNN
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: DATA WRAPPER FOR MULTIMODAL GNN")
    logger.info("="*80)
    
    data_wrapper_script = preprocessing_dir / "data_wrapper.py"
    traning_data_dir = tcga_pancan_dir / "traning_data"
    if traning_data_dir.exists() and any(traning_data_dir.iterdir()):
        logger.info(f"Found: Training data in {traning_data_dir}, skipping data wrapper step.")
    else:
        if not run_script(str(data_wrapper_script),
                        "Prepare data for Multimodal GNN training",
                        cwd=str(tcga_pancan_dir)):
            logger.error("Data wrapper failed. Stopping pipeline.")
            return False
    
    # Verify output files
    logger.info("\n" + "="*80)
    logger.info("VERIFICATION: CHECKING OUTPUT FILES")
    logger.info("="*80 + "\n")
    
    required_files = {
        "Clinical data": tcga_pancan_dir / "data/omics/clinical.csv",
        "CNV data": tcga_pancan_dir / "data/omics/cnv.csv",
        "Expression data": tcga_pancan_dir / "data/omics/fpkm.csv",
        "miRNA data": tcga_pancan_dir / "data/omics/mirna.csv",
        "GGI adjacency matrix": datasets_dir / "data/prior_knowledge/GGI/GGI_adjacency_sparse.npz",
        "GGI node list": datasets_dir / "data/prior_knowledge/GGI/GGI_nodes_list.csv",
        "miRNA-gene adjacency matrix": datasets_dir / "data/prior_knowledge/miRNA/gene_miRNA_adjacency_sparse.npz",
        "miRNA node list": datasets_dir / "data/prior_knowledge/miRNA/miRNA_nodes_list.csv",
        "tf interactions": datasets_dir / "data/prior_knowledge/TF/tf_interactions.tsv",
        "Training data for GNN": tcga_pancan_dir / "traning_data/expression_data_pancan.tsv"
    }
    
    all_files_present = True
    for description, file_path in required_files.items():
        if not check_file_exists(str(file_path), description):
            all_files_present = False
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("="*80)
    
    if all_files_present:
        logger.info("All pipeline steps completed successfully!")
        logger.info("All required files are present.")
        logger.info("\nThe data is ready for Multimodal GNN training.")
        logger.info("Next step: Run cancer_test.py in Multimodal-GNN-for-Cancer-Subtype-Clasification/")
        return True
    else:
        logger.warning("Error: Pipeline completed with some missing files.")
        logger.warning("Please check the logs above for details.")
        return False


if __name__ == "__main__":
    logger.info("Starting TCGA-PANCAN data processing pipeline...")
    success = main()
    
    if success:
        logger.info("\nPipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("\nError: Pipeline failed. Check logs for details.")
        sys.exit(1)

