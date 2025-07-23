# %%
import re
import pendulum
import jiwer
import pandas as pd
from pathlib import Path

# %%
corrected_dirs = [
    # vpb_dataset/label/transcript_corrected_hieudm13_cutoff_20250721/transcript_corrected
    # vpb_dataset/label/transcripts_corrected_quangdm4_cutoff_21_7/transcripts_corrected
    Path("../../../vpb_dataset/label/transcript_corrected_hieudm13_cutoff_20250721/transcript_corrected"),
    Path("../../../vpb_dataset/label/transcripts_corrected_quangdm4_cutoff_21_7/transcripts_corrected"),
]

# vpb_dataset/archive/tts_dataset_best_call_agent_audio/tts_dataset_best_call_agent_audio/transcripts
predict_dir = Path("../../../vpb_dataset/archive/tts_dataset_best_call_agent_audio/tts_dataset_best_call_agent_audio/transcripts")

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

# %%
df = pd.DataFrame(call_metadata)
df.to_csv("df_call_metadata.csv", index=False)
print("✅ Saved to df_call_metadata.csv")

# %%
print("🔎 Full dataset size:", len(df))
print("📊 Overall WER:", jiwer.wer(df['GOLD'].tolist(), df['PRED'].tolist()))

# %% Filter agent side
df_agent = df[df['FILE_NAME'].str.contains('___left')]
print("🎙️ Agent samples:", len(df_agent))
print("🎯 Agent WER:", jiwer.wer(df_agent['GOLD'].tolist(), df_agent['PRED'].tolist()))

# %% Filter client side
df_client = df[df['FILE_NAME'].str.contains('___right')]
print("📞 Client samples:", len(df_client))
print("🎯 Client WER:", jiwer.wer(df_client['GOLD'].tolist(), df_client['PRED'].tolist()))
