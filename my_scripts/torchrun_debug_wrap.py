# my_scripts/torchrun_debug_wrap.py
import os, sys, runpy
from pathlib import Path

# --- ensure we can import 'verl' regardless of where this wrapper lives ---
repo_root = Path(__file__).resolve().parents[1]  # /workspace/verl
for p in (repo_root, repo_root / "src"):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
# -------------------------------------------------------------------------

# Which rank(s) should wait for the debugger?
debug_all = os.getenv("DEBUG_ALL_RANKS", "0") == "1"
target_rank = int(os.getenv("DEBUG_RANK", "0"))
rank = int(os.getenv("LOCAL_RANK", "0"))

if debug_all or rank == target_rank:
    import debugpy
    base = int(os.getenv("BASE_PORT", "5678"))
    port = base + (rank if debug_all else 0)
    debugpy.listen(("0.0.0.0", port))
    print(f"[debugpy] rank={rank} listening on :{port}", flush=True)
    if os.getenv("DEBUG_WAIT", "1") == "1":
        debugpy.wait_for_client()

# Hand off to the real trainer module with the original Hydra args
sys.argv = ["verl.trainer.fsdp_sft_trainer", *sys.argv[1:]]
runpy.run_module(sys.argv[0], run_name="__main__")