#!/bin/bash

OUTPUT="stt_chunkformer_code.txt"
echo "Exporting all Python source code from stt_chunkformer/ to $OUTPUT"
echo "# === Combined source code from stt_chunkformer ===" > "$OUTPUT"

find ../model -type f -name "*.py" | sort | while read -r file; do
    echo -e "\n\n# === FILE: $file ===" >> "$OUTPUT"
    cat "$file" >> "$OUTPUT"
done

echo "✅ Done. Saved to $OUTPUT"
