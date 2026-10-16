"""Saved upscale runs, so a later call can redraw only some tiles of one.

Each run keeps the source image, the tile plan and every tile as it came out
under ``data/image_hd_upscale/jobs/<id>/`` (data, not tmp: the host's cleanup
may empty tmp, and a retry can come days later). The short id is shown in
the result; a retry names it explicitly, so two upscales in one chat are
never mixed up. The exact output file is accepted too (the WebUI hands back
the local path). A retry that names neither falls back to the newest run of
the same chat, and the reply says which task it used.
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image

from app.logger import setup_logger

from .tiling import Tile, TilePlan, to_array, to_image
from .upscale import TileOutcome, UpscaleResult


logger = setup_logger(__name__)

META = "meta.json"
VERSION = 1
JOB_ID = re.compile(r"^[0-9a-f]{6}$")


@dataclass
class Job:
    id: str
    directory: Path
    plan: TilePlan
    scale: float
    text_hint: str = ""
    fixes: List[Tuple[str, str]] = field(default_factory=list)
    session_id: str = ""
    outputs: List[str] = field(default_factory=list)
    statuses: Dict[str, dict] = field(default_factory=dict)
    updated: float = 0.0

    def source(self) -> Image.Image:
        with Image.open(self.directory / "source.png") as img:
            img.load()
            return img.copy()

    def tiles(self) -> Dict[str, TileOutcome]:
        outcomes = {}
        for label, info in self.statuses.items():
            with Image.open(self.directory / "tiles" / f"{label}.png") as img:
                array = to_array(img.convert("RGB"))
            outcomes[label] = TileOutcome(image=array, status=info["status"], patched=float(info.get("patched", 0.0)))
        return outcomes


def _plan_to_dict(plan: TilePlan) -> dict:
    data = asdict(plan)
    data["tiles"] = [asdict(t) for t in plan.tiles]
    return data


def _plan_from_dict(data: dict) -> TilePlan:
    data = dict(data)
    tiles = [Tile(**t) for t in data.pop("tiles")]
    return TilePlan(**data, tiles=tiles)


def save_job(
    root: Path,
    *,
    source: Image.Image,
    result: UpscaleResult,
    output: Path,
    text_hint: str,
    session_id: str,
    fixes: Sequence[Tuple[str, str]] = (),
    job: Optional[Job] = None,
    keep: int = 0,
) -> Job:
    """Record a run (or update ``job`` after a retry); with ``keep`` > 0, prune old runs beyond it."""
    if job is None:
        root.mkdir(parents=True, exist_ok=True)
        job_id = secrets.token_hex(3)
        while (root / job_id).exists():
            job_id = secrets.token_hex(3)
        job = Job(id=job_id, directory=root / job_id, plan=result.plan, scale=result.scale)
        (job.directory / "tiles").mkdir(parents=True, exist_ok=True)
        source.save(job.directory / "source.png")
    job.text_hint = text_hint
    job.fixes = [tuple(f) for f in fixes]
    job.session_id = session_id or job.session_id
    job.outputs.append(str(output.resolve()))
    for label in result.redrawn or list(result.tiles):
        tile = result.tiles[label]
        to_image(tile.image).save(job.directory / "tiles" / f"{label}.png")
    job.statuses = {
        label: {"status": t.status, "patched": round(t.patched, 4)}
        for label, t in result.tiles.items()
    }
    job.updated = time.time()
    meta = {
        "version": VERSION,
        "scale": job.scale,
        "text_hint": job.text_hint,
        "fixes": [list(f) for f in job.fixes],
        "session_id": job.session_id,
        "outputs": job.outputs,
        "plan": _plan_to_dict(job.plan),
        "tiles": job.statuses,
        "updated": job.updated,
    }
    (job.directory / META).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    if keep > 0:
        _prune(root, keep)
    return job


def _load(directory: Path) -> Optional[Job]:
    try:
        meta = json.loads((directory / META).read_text(encoding="utf-8"))
        if meta.get("version") != VERSION:
            return None
        return Job(
            id=directory.name,
            directory=directory,
            plan=_plan_from_dict(meta["plan"]),
            scale=float(meta["scale"]),
            text_hint=meta.get("text_hint", ""),
            fixes=[tuple(f) for f in meta.get("fixes", []) if len(f) == 2],
            session_id=meta.get("session_id", ""),
            outputs=list(meta.get("outputs", [])),
            statuses=dict(meta.get("tiles", {})),
            updated=float(meta.get("updated", 0.0)),
        )
    except Exception as e:
        logger.debug(f"[图片放大] 读取放大记录 {directory.name} 失败: {e}")
        return None


def list_jobs(root: Path) -> List[Job]:
    if not root.is_dir():
        return []
    jobs = [job for d in root.iterdir() if d.is_dir() and (job := _load(d))]
    return sorted(jobs, key=lambda j: j.updated, reverse=True)


def _prune(root: Path, keep: int) -> None:
    for job in list_jobs(root)[max(1, keep):]:
        shutil.rmtree(job.directory, ignore_errors=True)


def find_job(root: Path, *, job_id: str = "", path: str = "", session_id: Optional[str] = None) -> Optional[Job]:
    """The run named by ``job_id``, else the one that produced the exact file ``path``.

    With ``session_id`` given (a retry that names no task), fall back to the
    newest run of that session; runs of other chats are never picked.
    """
    job_id = job_id.strip().lower()
    if job_id:
        return _load(root / job_id) if JOB_ID.match(job_id) else None
    jobs = list_jobs(root)
    if path:
        try:
            resolved = str(Path(path).resolve())
        except (OSError, ValueError):
            resolved = ""
        for job in jobs:
            if resolved and resolved in job.outputs:
                return job
    if session_id is not None:
        for job in jobs:
            if job.session_id == session_id:
                return job
    return None


def split_retry_text(text: str) -> tuple[str, str]:
    """``"3fa9c1 2-3,1-1"`` -> ``("3fa9c1", "2-3,1-1")``; the id may be anywhere, or absent."""
    job_id, rest = "", []
    for token in re.split(r"[\s,，、]+", text.strip()):
        if not token:
            continue
        if not job_id and JOB_ID.match(token.lower()):
            job_id = token.lower()
        else:
            rest.append(token)
    return job_id, ",".join(rest)


def parse_labels(text: str, plan: TilePlan) -> List[str]:
    """``"2-3, 1-1"`` / ``"2-3 1-1"`` / ``"2_3"`` -> valid labels; raises on unknown tiles."""
    valid = {f"{t.row + 1}-{t.col + 1}" for t in plan.tiles}
    labels: List[str] = []
    for part in text.replace("，", ",").replace("、", ",").replace(" ", ",").split(","):
        part = part.strip().replace("_", "-").replace("x", "-").replace("×", "-")
        if not part:
            continue
        if part not in valid:
            raise ValueError(f"没有第 {part} 块（这张图共 {plan.rows} 行 {plan.cols} 列，块号写成“行-列”，如 2-3）")
        if part not in labels:
            labels.append(part)
    return labels
