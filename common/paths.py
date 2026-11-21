"""Chemins des données et environnements de l'équipe SLZ sur JURECA (créés par hackathon_setup.sh)."""
from pathlib import Path

TEAM_DIR = Path("/p/scratch/training2666/mouhali1/SLZ")

TASK_DIRS = {
    "c1": TEAM_DIR / "Hackathon_Watermark_Localisation_Task",
    "c2": TEAM_DIR / "model_equality",
    "c3": TEAM_DIR / "unlearning",
}


def task_dir(challenge: str) -> Path:
    """Dossier officiel d'une tâche (données, .venv, output/) ; `challenge` vaut c1, c2 ou c3."""
    return TASK_DIRS[challenge]
