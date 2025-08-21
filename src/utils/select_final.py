# final_checkpoint_promoter.py
from __future__ import annotations
import json, hashlib, os, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

@dataclass(frozen=True)
class Candidate:
    val_total: float
    epoch: int
    pt_path: Path
    json_path: Path
    manifest: Dict

class FinalCheckpointPromoter:
    """
    Promotes the best milestone to a stable final checkpoint without symlinks.
    Prefers a hardlink; falls back to a byte-for-byte copy. Verifies integrity via SHA-256.

    Typical usage:
        promoter = FinalCheckpointPromoter(
            milestones_dir=Path(r"C:\...\results\checkpoints\diffusion\milestones"),
            final_name="final.ckpt"
        )
        result = promoter.finalize(expected={"sampling_method":"ddim", "sampling_steps":50, "cfg_scale":0.5})
        print(result)

    Returned `result` dict includes:
        {
          "best_pt": Path(...),
          "best_json": Path(...),
          "val_total": float,
          "epoch": int,
          "final_ckpt": Path(...),
          "mode": "hardlink" | "copy",
          "ckpt_sha256": str,
          "json_sha256": str,
        }
    """

    def __init__(self, milestones_dir: Path, final_name: str = "final.ckpt") -> None:
        self.milestones_dir = Path(milestones_dir)
        self.final_dir = self.milestones_dir.parent  # .../diffusion
        self.final_name = final_name

    # ---------- Public API ----------

    def finalize(
        self,
        expected: Optional[Dict[str, object]] = None,
        glob_pattern: str = "milestone_*.json",
        prefer: str = "hardlink_then_copy",  # or "copy_only"
        write_sidecars: bool = True,
    ) -> Dict[str, object]:
        """
        Select best checkpoint, optionally verify manifest fields, promote to final, write sidecars.
        """
        candidates = self.list_candidates(glob_pattern)
        best = self.select_best(candidates)

        if expected:
            self.verify_manifest(best.manifest, expected)

        ckpt_sha = self.sha256(best.pt_path)
        json_sha = self.sha256(best.json_path)

        final_ckpt = self.final_dir / self.final_name
        mode = self._promote(best.pt_path, final_ckpt, prefer=prefer)

        if write_sidecars:
            (self.final_dir / f"{self.final_name}.sha256").write_text(f"{ckpt_sha}  {best.pt_path.name}\n", encoding="utf-8")
            (self.final_dir / "final.json.sha256").write_text(f"{json_sha}  {best.json_path.name}\n", encoding="utf-8")

        return {
            "best_pt": best.pt_path,
            "best_json": best.json_path,
            "val_total": best.val_total,
            "epoch": best.epoch,
            "final_ckpt": final_ckpt,
            "mode": mode,
            "ckpt_sha256": ckpt_sha,
            "json_sha256": json_sha,
        }

    def list_candidates(self, glob_pattern: str = "milestone_*.json") -> List[Candidate]:
        """
        Scan milestones dir for JSON manifests with a matching .pt and return Candidate list.
        """
        if not self.milestones_dir.exists():
            raise FileNotFoundError(f"Milestones dir not found: {self.milestones_dir}")

        candidates: List[Candidate] = []
        for j in sorted(self.milestones_dir.glob(glob_pattern)):
            manifest = json.loads(j.read_text(encoding="utf-8"))
            pt = j.with_suffix(".pt")
            if not pt.exists():
                continue
            val_total = float(manifest.get("val_total"))
            epoch = int(manifest.get("epoch"))
            candidates.append(Candidate(val_total=val_total, epoch=epoch, pt_path=pt, json_path=j, manifest=manifest))

        if not candidates:
            raise RuntimeError("No milestones found.")
        return candidates

    def select_best(self, candidates: List[Candidate]) -> Candidate:
        """
        Choose lowest val_total; tie-break by highest epoch.
        """
        candidates = sorted(candidates, key=lambda c: (c.val_total, -c.epoch))
        return candidates[0]

    def verify_manifest(self, manifest: Dict[str, object], expected: Dict[str, object]) -> None:
        """
        Assert selected milestone matches expected fields (e.g., sampling_method/steps, cfg_scale).
        """
        for k, v in expected.items():
            mv = manifest.get(k)
            if str(mv) != str(v):
                raise AssertionError(f"Manifest mismatch for '{k}': got {mv} != expected {v}")

    # ---------- Helpers ----------

    @staticmethod
    def sha256(p: Path, chunk: int = 1 << 20) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(chunk), b""):
                h.update(b)
        return h.hexdigest()

    def _promote(self, src: Path, dst: Path, prefer: str = "hardlink_then_copy") -> str:
        """
        Create final checkpoint at `dst` without symlinks.
        prefer == "hardlink_then_copy": try hardlink (same volume) then copy
        prefer == "copy_only": always copy
        Verifies SHA-256 equality post-promotion.
        """
        if dst.exists() or dst.is_symlink():
            dst.unlink()

        mode: str
        if prefer == "copy_only":
            shutil.copy2(src, dst)
            mode = "copy"
        else:
            # Try hardlink first (works on same NTFS/volume; no admin required).
            try:
                # Quick same-drive check on Windows to avoid cryptic errors
                if hasattr(src, "drive") and hasattr(dst, "drive") and src.drive and dst.drive and (src.drive != dst.drive):
                    raise OSError("Source and destination are on different drives; hardlink not possible.")
                os.link(src, dst)
                mode = "hardlink"
            except Exception:
                shutil.copy2(src, dst)
                mode = "copy"

        # Integrity check
        if self.sha256(src) != self.sha256(dst):
            # clean up to avoid a bad final
            try:
                if dst.exists():
                    dst.unlink()
            finally:
                raise RuntimeError("final.ckpt content mismatch after promotion")

        return mode
