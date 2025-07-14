#!/usr/bin/env bash
# merge_code.sh — gom tất cả .py files trong thư mục hiện tại vào 1 file duy nhất

# Nếu bạn truyền tên file đích khi chạy script, nó sẽ dùng tên đó; 
# ngược lại mặc định là combined.py
OUTPUT_FILE="${1:-combined.py}"

# Xóa file đích cũ (nếu có) để bắt đầu sạch
rm -f "$OUTPUT_FILE"

# Thêm header vào file đích
echo "#!/usr/bin/env python3" > "$OUTPUT_FILE"
echo "# Combined script generated on $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Duyệt qua tất cả file .py theo thứ tự tên, chèn comment phân vùng rồi append nội dung
for f in $(ls *.py | sort); do
  echo "## --- Begin file: $f ---" >> "$OUTPUT_FILE"
  cat "$f" >> "$OUTPUT_FILE"
  echo "" >> "$OUTPUT_FILE"
done

# Thiết lập quyền executable nếu muốn chạy trực tiếp
chmod +x "$OUTPUT_FILE"

echo "Merged $(ls *.py | wc -l) files into $OUTPUT_FILE"
