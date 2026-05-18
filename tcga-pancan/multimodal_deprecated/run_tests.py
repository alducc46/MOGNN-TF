# run_grid.py
import json
import logging
import itertools
import os
import re
import sys
import time
import subprocess
from pathlib import Path

logger = logging.getLogger("runner")


def sanitize(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.=-]+", "_", s)
    return s.strip("_") or "run"


def to_cli_args(d: dict) -> list[str]:
    args: list[str] = []
    for k, v in d.items():
        args.append(f"--{k}")
        args.append(str(v))
    return args

def run_cmd(cmd: list[str], cwd: Path, desc: str) -> bool:
    logger.info("=" * 80)
    logger.info("STEP: %s", desc)
    logger.info("CWD : %s", str(cwd))
    logger.info("CMD : %s", " ".join(cmd))
    logger.info("=" * 80)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        # stderr -> stdout per mantenere ordine temporale e streaming semplice
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert p.stdout is not None
        for line in p.stdout:
            line = line.rstrip("\n")
            print(line, flush=True)      # LIVE su console
            logger.info(line)           # anche su logger

        rc = p.wait()
        if rc == 0:
            logger.info("✓ OK: %s\n", desc)
            return True

        logger.error("✗ FAIL: %s (returncode=%s)\n", desc, rc)
        return False

    except Exception:
        logger.exception("✗ Unexpected error in %s", desc)
        return False

def resolve_rel(p: str | Path, base: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (base / p)


def main(config_path: str):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg_path = Path(config_path).resolve()
    cfg_dir = cfg_path.parent  # base per i path relativi del JSON
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    cwd = resolve_rel(cfg.get("cwd", "."), cfg_dir).resolve()
    script = resolve_rel(cfg["script"], cfg_dir).resolve()

    stop_on_fail = bool(cfg.get("stop_on_fail", False))

    base_args = cfg.get("base_args", {})
    if isinstance(base_args, dict):
        base_cli = to_cli_args(base_args)
    elif isinstance(base_args, list):
        base_cli = [str(x) for x in base_args]
    else:
        raise ValueError("base_args must be dict or list")

    grid: dict = cfg["grid"]
    grid_keys = list(grid.keys())
    grid_values = [grid[k] for k in grid_keys]

    name_keys = cfg.get("output_name_keys", grid_keys)

    # output_root RELATIVO a cwd (default: ./runs)
    output_root_cfg = Path(cfg.get("output_root", "runs"))
    out_root_base = (cwd / output_root_cfg) if not output_root_cfg.is_absolute() else output_root_cfg

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = out_root_base / ts
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {out_root.relative_to(cwd)}")

    abbrev_map = {
            "omic_mode": "om",
            "num_gene": "ng",
            "batch_size": "bs",
            "sampler_strategy": "smpl",
            "decoder": "dec",
            "l2": "l2",
            "parallel": "par",
            "dropout": "do",
            "lr": "lr",
            "epochs": "ep",
            "edge_weight": "ew"
        }
    
    run_idx = 0
    for combo in itertools.product(*grid_values):
        run_idx += 1
        params = {k: v for k, v in zip(grid_keys, combo)}
        parts = []
        for k in name_keys:
            if k in params:
                # Usa l'abbreviazione se esiste, altrimenti usa il nome completo
                short_key = abbrev_map.get(k, k)
                val = params[k]
                
                # Accorcia anche i booleani
                if val is True: val = "T"
                elif val is False: val = "F"
                
                parts.append(f"{short_key}={val}")
        run_name = sanitize("__".join(parts))
        run_dir = out_root / f"{run_idx:03d}__{run_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

        # --output_folder passato RELATIVO a cwd (quindi non "sporchi" con assoluti)
        output_folder_rel = run_dir.relative_to(cwd)

        cmd = [
            sys.executable, "-u",
            str(script),
            *base_cli,
            *to_cli_args(params),
            "--output_folder", str(output_folder_rel),
        ]


        ok = run_cmd(cmd, cwd=cwd, desc=f"run {run_idx}: {run_name}")
        if not ok and stop_on_fail:
            sys.exit(1)

    logger.info("DONE. Output root: %s", str(out_root.relative_to(cwd)))


if __name__ == "__main__":
    # Esempio: python run_grid.py configs/run_config.json
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "run_config.json"
    print(f"Using config path: {cfg_path}")
    main(cfg_path)
