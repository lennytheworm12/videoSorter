#!/usr/bin/env python3
"""Download and provenance Stanza parser assets for Phase 2I (network path).

Phase 2I evaluation itself is strictly offline: it loads local assets with
``DownloadMethod.NONE``.  This script is the only network path for parser
assets.  It:

1. preflights the lexically supplied ``--assets-dir`` (default
   ``data/phase2i_assets``) with an lstat walk from the filesystem anchor
   through the root, rejecting any symlink ancestor before it creates,
   downloads into, hashes, or reads through the path; then runs
   ``stanza.download`` into it with a writable Hugging Face cache;
2. writes ``ASSET_PROVENANCE.json`` recording stanza version, package,
   processors, and per-file SHA-256 hashes plus a manifest hash;
3. verifies the assets with :func:`pipeline.phase2i_syntax.verify_assets_provenance`.

The evaluation scripts never call this script and never download.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.phase2i_syntax import (
    STANZA_LANGUAGE,
    STANZA_PACKAGE,
    STANZA_PROCESSORS,
    verify_parser_asset_path,
    verify_assets_provenance,
    write_asset_provenance,
)


ROOT = Path(__file__).resolve().parents[1]


def _prepare_asset_path(raw: str | Path) -> Path | None:
    """Lexically absolutize and symlink-preflight the asset path.

    ``None`` is returned (with the problems printed) when any existing
    component from the filesystem anchor through the supplied root is a
    symlink, so ``main`` never downloads or hashes through a link and never
    resolves the evidence away before checking it.
    """
    check = verify_parser_asset_path(raw)
    if not check["verified"]:
        print("[phase2i-assets] parser asset path rejected:")
        for problem in check["problems"]:
            print("  -", problem)
        return None
    return Path(check["path"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=ROOT / "data" / "phase2i_assets",
    )
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=None,
        help=(
            "writable Hugging Face cache (HF_HOME); defaults to a "
            "phase2i_assets/.hf_cache directory when the default "
            "~/.cache/huggingface is not writable"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    assets_dir = _prepare_asset_path(args.assets_dir)
    if assets_dir is None:
        return 1
    assets_dir.mkdir(parents=True, exist_ok=True)
    hf_cache = (
        Path(os.path.abspath(os.fspath(args.hf_cache)))
        if args.hf_cache else None
    )
    default_hf = Path.home() / ".cache" / "huggingface"
    if hf_cache is None:
        try:
            default_hf.mkdir(parents=True, exist_ok=True)
            probe = default_hf / ".phase2i-write-probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            hf_cache = default_hf
        except OSError:
            hf_cache = assets_dir / ".hf_cache"
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(hf_cache)

    import stanza

    print(
        f"[phase2i-assets] stanza {stanza.__version__} "
        f"language={STANZA_LANGUAGE} package={STANZA_PACKAGE} "
        f"processors={','.join(STANZA_PROCESSORS)}",
    )
    print(f"[phase2i-assets] assets dir: {assets_dir}")
    print(f"[phase2i-assets] hf cache: {hf_cache}")
    stanza.download(
        STANZA_LANGUAGE,
        model_dir=str(assets_dir),
        processors=",".join(STANZA_PROCESSORS),
        package=STANZA_PACKAGE,
        verbose=args.verbose,
    )
    provenance_path = write_asset_provenance(
        assets_dir,
        stanza_version=stanza.__version__,
        package=STANZA_PACKAGE,
        processors=STANZA_PROCESSORS,
        created_at=datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z",
        ),
    )
    verification = verify_assets_provenance(assets_dir)
    if not verification["verified"]:
        print("[phase2i-assets] asset verification FAILED:")
        for problem in verification["problems"]:
            print("  -", problem)
        return 1
    print(
        f"[phase2i-assets] verified {len(verification.get('problems', []))} "
        "problems; provenance: " + str(provenance_path),
    )
    print(
        "[phase2i-assets] manifest sha256: "
        + verification["manifest_sha256"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
