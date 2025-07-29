import re
import pendulum
import jiwer
import pandas as pd
from pathlib import Path

def generate_call_metadata(
    corrected_dirs,
    predict_dir,
    output_csv_path="df_call_metadata.csv"
):
    call_metadata = []

    for folder in corrected_dirs:
        print(f"🔍 Scanning folder: {folder}")
        count = 0

        for filepath in folder.glob('*'):
            if filepath.name.lower().startswith('e_'):
                match = re.match(
                    r"E_(.*?)_D_(\d{4}-\d{2}-\d{2})_H_(\d{2})(\d{2})(\d{2})_(\d{3})_CLID_(\d+)___.*",
                    filepath.name
                )

                if match:
                    with open(filepath, encoding="utf-8") as f:
                        gold = f.read().strip()
                    if not gold:
                        continue  # Skip empty transcript

                    pred_file = predict_dir / filepath.name
                    if not pred_file.exists():
                        print(f"⚠️ Missing prediction file for: {filepath.name}")
                        continue

                    with open(pred_file, encoding="utf-8") as f:
                        pred = f.read().strip()

                    agent_username = match.group(1)
                    business_date = pendulum.parse(match.group(2)).date()
                    ref_date = business_date.end_of('month')
                    call_time = f"{match.group(2)} {match.group(3)}:{match.group(4)}:{match.group(5)}"
                    phone_number = match.group(7)

                    call_metadata.append({
                        'REF_DATE': ref_date,
                        'BUSINESS_DATE': business_date,
                        'AGENT_ID': agent_username,
                        'CALL_START': call_time,
                        'CALL_PHONE': phone_number,
                        'FILE_NAME': filepath.name,
                        'GOLD': gold,
                        'PRED': pred,
                    })

                    count += 1

        print(f"✅ Total processed in {folder.name}: {count}")
        print("-" * 40)

    # Ghi ra file CSV
    df = pd.DataFrame(call_metadata)
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Saved to {output_csv_path}")
    print("🔎 Dataset size:", len(df))
    print("📊 Overall WER:", jiwer.wer(df['GOLD'].tolist(), df['PRED'].tolist()))

    # Agent & client split
    df_agent = df[df['FILE_NAME'].str.contains('___left')]
    df_client = df[df['FILE_NAME'].str.contains('___right')]

    print("🎙️ Agent samples:", len(df_agent), "| WER:", jiwer.wer(df_agent['GOLD'].tolist(), df_agent['PRED'].tolist()))
    print("📞 Client samples:", len(df_client), "| WER:", jiwer.wer(df_client['GOLD'].tolist(), df_client['PRED'].tolist()))


if __name__ == "__main__":
    base_path = Path(__file__).resolve().parent.parent  # trỏ tới vpb_dataset/
    label_base = base_path / "label"
    archive_base = base_path / "archive/tts_dataset_best_call_agent_audio/tts_dataset_best_call_agent_audio"

    corrected_dirs = [
        label_base / "transcript_corrected_hieudm13_cutoff_20250721/transcript_corrected",
        label_base / "transcripts_corrected_quangdm4_cutoff_21_7/transcripts_corrected",
    ]

    predict_dir = archive_base / "transcripts"
    output_csv_path = base_path / "code/df_call_metadata.csv"

    generate_call_metadata(corrected_dirs, predict_dir, output_csv_path)
