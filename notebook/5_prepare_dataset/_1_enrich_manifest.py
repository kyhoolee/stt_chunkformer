import json
import pandas as pd
from pathlib import Path


def enrich_manifest(manifest_path: Path, df_meta: pd.DataFrame, output_path: Path):
    # Tạo mapping: FILE_NAME → {GOLD, PRED}
    file_map = {
        row["FILE_NAME"]: {
            "gold_corrected": row["GOLD"],
            "pred_old": row["PRED"]
        }
        for _, row in df_meta.iterrows()
    }

    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    enriched = []
    missing = 0

    for e in entries:
        file_key = e["utt_id"] + ".txt"
        extra = file_map.get(file_key, {})
        if not extra:
            missing += 1
        e.update(extra)
        enriched.append(e)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved: {output_path.name} | {len(enriched)} entries (missing: {missing})")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    manifest_dir = base_dir / "manifest_vpb_non_empty_full"
    df_path = base_dir / "code/df_call_metadata.csv"

    df = pd.read_csv(df_path)

    enrich_manifest(
        manifest_path=manifest_dir / "train_meta.json",
        df_meta=df,
        output_path=manifest_dir / "train_meta_debug.json"
    )

    enrich_manifest(
        manifest_path=manifest_dir / "valid_meta.json",
        df_meta=df,
        output_path=manifest_dir / "valid_meta_debug.json"
    )
