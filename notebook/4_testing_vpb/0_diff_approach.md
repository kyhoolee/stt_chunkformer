Rất hợp lý! Dưới đây là các bước debug chi tiết giúp bạn truy vết sự khác biệt giữa label corrected và output model (cũ và mới), để xác định xem pipeline training hiện tại có thay đổi logic decode hay không:

---

## 🧭 **Bước 1: Mở rộng `VivosDataset` để load thêm các label (gold gốc & pred cũ)**

### ✅ Mục tiêu:

* Dataset trả về cả `entry.text` (label gốc theo json), `gold_corrected` và `pred_old`.

### ✏️ Gợi ý mở rộng:

1. Tạo một dict tra cứu `gold_corrected` và `pred_old` theo `utt_id` (tên file).
2. Trong `__getitem__`, load thêm `gold_corrected` và `pred_old` nếu có.

```python
# Giả sử bạn đã có df_call_metadata.csv
corrected_map = {row['FILE_NAME']: row for _, row in df.iterrows()}
...

entry = self.meta[idx]
base_name = os.path.basename(entry.audio_path)
corrected = corrected_map.get(base_name)

gold_corrected = corrected['GOLD'] if corrected else None
pred_old = corrected['PRED'] if corrected else None
```

> 🔁 Trả về `return feats, feats.size(0), toks, len(token_ids), entry, gold_corrected, pred_old`

---

## 🧭 **Bước 2: So sánh từng sample trong đánh giá (decode cũ vs decode mới vs label corrected)**

### ✅ Mục tiêu:

* In ra một số mẫu có sự khác biệt lớn.
* Ghi log lại các dòng có `diff > threshold` (ví dụ WER > 0.3)

### ✏️ Gợi ý sửa `evaluate()`:

Thêm log chi tiết sau mỗi lượt decode:

```python
for i in range(feats.size(0)):
    ...
    ref_ids = y[0, :y_lens].tolist()
    ref_text = tokenizer.decode(ref_ids)
    hyp_text = tokenizer.decode(hyp_ids)  # model decode ra

    sample_entry = entries[i]  # lấy từ batch collate
    utt_id = sample_entry.utt_id

    corrected = corrected_map.get(utt_id)
    gold_corrected = corrected['GOLD'] if corrected else None
    pred_old = corrected['PRED'] if corrected else None

    print("🧾 Sample:", utt_id)
    print("🔹Ref     :", ref_text)
    print("🔸Hyp     :", hyp_text)
    print("🟢Correct :", gold_corrected)
    print("🟡OldPred :", pred_old)

    if gold_corrected:
        print("✅ WER new vs corrected:", jiwer.wer(gold_corrected, hyp_text))
    if pred_old:
        print("🔁 WER old vs corrected:", jiwer.wer(gold_corrected, pred_old))
    print("-" * 40)
```

> Có thể ghi log vào file CSV để so sánh bằng pandas.

---

## 🧭 **Bước 3: Tách logic decode thành function có thể gọi độc lập**

### ✅ Mục tiêu:

* Đảm bảo logic decode trong training eval giống hệt decode trong inference trước đó.
* Tách `decode_long_form()` để có thể dùng chung cho cả hai pipeline.

---

## 🧭 **Bước 4: Lọc thống kê tổng hợp**

### ✅ Mục tiêu:

* Xác định bao nhiêu mẫu:

  * model mới tốt hơn
  * model cũ tốt hơn
  * giữ nguyên kết quả
  * cả hai đều sai với label corrected

### ✏️ Gợi ý:

Tạo dataframe:

```python
df_compare = pd.DataFrame({
    'utt_id': [...],
    'gold_corrected': [...],
    'pred_old': [...],
    'pred_new': [...],
    'wer_old': [...],
    'wer_new': [...],
})
```

---

## 🧪 **Bonus – chạy chế độ kiểm thử nhỏ**

Chạy `get_dataloaders_smoke()` để debug chỉ 10-20 sample cho dễ quan sát và in log chi tiết.

---

Nếu bạn muốn mình code khung cụ thể cho từng bước (ví dụ sửa `VivosDataset`, `evaluate`, log ra file...), mình có thể hỗ trợ viết sẵn.

Bạn muốn bắt đầu với bước nào trước?
