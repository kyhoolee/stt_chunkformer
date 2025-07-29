Dưới đây là **bảng ước tính tổng thời gian xử lý augment cho 3000 giờ dữ liệu** (\~10.8 triệu giây), theo từng loại augment — dựa trên số liệu benchmark bạn cung cấp:

---

### 📊 Bảng tổng thời gian augment cho 3000h dữ liệu

| Augment     | T/g mỗi 1s audio (ms) | Tổng thời gian cho 3000h (giây) | Thời gian (phút) | Thời gian (giờ) | Ghi chú                     |
| ----------- | --------------------- | ------------------------------- | ---------------- | --------------- | --------------------------- |
| `vol`       | 0.20 ms               | 2,160                           | 36 min           | 0.6 h           | ✅ rất nhanh                 |
| `noise`     | 0.45 ms               | 4,860                           | 81 min           | 1.35 h          | ✅ nhanh                     |
| `telephony` | 1.21 ms               | 13,068                          | 218 min          | 3.6 h           | ✅ chấp nhận được            |
| `reverb`    | 22.43 ms              | 242,244                         | 4,037 min        | 67.3 h          | ⚠️ tương đối chậm           |
| `pitch`     | 593.78 ms             | 6,409,824                       | 106,830 min      | 1,780.5 h       | 🐢 rất chậm – cần tối ưu    |
| `speed`     | 760.36 ms             | 8,208,768                       | 136,813 min      | 2,280.2 h       | 🐢 cực chậm – phải chọn lọc |

---

### ✅ Gợi ý chiến lược áp dụng cho 3000h

| Augment     | Nên áp dụng toàn bộ? | Gợi ý                                                        |
| ----------- | -------------------- | ------------------------------------------------------------ |
| `vol`       | ✅ Có                 | Không cần giới hạn – áp dụng offline toàn bộ                 |
| `noise`     | ✅ Có                 | Dễ dàng preload hoặc generate ngẫu nhiên                     |
| `telephony` | ✅ Có                 | Có thể thực hiện batch song song bằng multiprocessing        |
| `reverb`    | ⚠️ Có giới hạn       | Chỉ áp dụng 10–20%, hoặc dùng IR ngắn, hoặc async queue      |
| `pitch`     | ❌ Tránh              | Chỉ apply với 5–10% hoặc khi thực sự cần diversity           |
| `speed`     | ❌ Tránh              | Apply trên subset nhỏ (VD: 5–10%) hoặc online trong training |



--------------

🎧 File: E_huongds_D_2025-04-01_H_080932_182_CLID_0904945499___000025238___left___000028746.wav | Duration: 3.51 sec
   🧪 vol        → 0.49 ms
   🧪 speed      → 460.22 ms
   🧪 telephony  → 3.29 ms

🎧 File: E_huongds_D_2025-04-01_H_080932_182_CLID_0904945499___000027062___right___000029802.wav | Duration: 2.74 sec
   🧪 vol        → 0.33 ms
   🧪 speed      → 7023.92 ms
   🧪 telephony  → 3.94 ms

🎧 File: E_huongds_D_2025-04-01_H_080932_182_CLID_0904945499___000030006___right___000031818.wav | Duration: 1.81 sec
   🧪 vol        → 0.52 ms
   🧪 speed      → 6565.06 ms
   🧪 telephony  → 3.95 ms

🎧 File: E_huongds_D_2025-04-01_H_080932_182_CLID_0904945499___000030326___left___000031082.wav | Duration: 0.76 sec
   🧪 vol        → 0.38 ms
   🧪 speed      → 6716.65 ms
   🧪 telephony  → 1.27 ms

🎧 File: E_huongds_D_2025-04-01_H_080932_182_CLID_0904945499___000030838___left___000034986.wav | Duration: 4.15 sec
   🧪 vol        → 0.60 ms
   🧪 speed      → 295.91 ms
   🧪 telephony  → 3.11 ms

🎧 File: E_huongds_D_2025-04-01_H_080932_182_CLID_0904945499___000035926___right___000036810.wav | Duration: 0.88 sec
   🧪 vol        → 0.20 ms
   🧪 speed      → 1720.24 ms
   🧪 telephony  → 1.06 ms

🎧 File: E_huongds_D_2025-04-01_H_080932_182_CLID_0904945499___000037174___left___000042058.wav | Duration: 4.88 sec
   🧪 vol        → 1.58 ms
   🧪 speed      → 40.57 ms
   🧪 telephony  → 3.40 ms

🎧 File: E_huongds_D_2025-04-01_H_080932_182_CLID_0904945499___000044246___left___000045322.wav | Duration: 1.08 sec
   🧪 vol        → 0.21 ms
   🧪 speed      → 22.34 ms
   🧪 telephony  → 0.94 ms

🎧 File: E_huongds_D_2025-04-01_H_080932_182_CLID_0904945499___000045110___left___000048682.wav | Duration: 3.57 sec
   🧪 vol        → 1.54 ms
   🧪 speed      → 7657.95 ms
   🧪 telephony  → 5.45 ms

🎧 File: E_huongds_D_2025-04-01_H_080932_182_CLID_0904945499___000051862___right___000053002.wav | Duration: 1.14 sec
   🧪 vol        → 0.30 ms
   🧪 speed      → 7720.30 ms
   🧪 telephony  → 2.21 ms

📊 Tổng hợp:
🔹 vol       :
   • Tổng thời gian xử lý  : 0.03 sec
   • Tổng độ dài audio     : 46.86 sec
   • T/g mỗi 1s audio      : 0.69 ms/sec
   • Trung bình mỗi file   : 1.62 ms
🔹 speed     :
   • Tổng thời gian xử lý  : 56.49 sec
   • Tổng độ dài audio     : 46.86 sec
   • T/g mỗi 1s audio      : 1205.44 ms/sec
   • Trung bình mỗi file   : 2824.59 ms
🔹 telephony :
   • Tổng thời gian xử lý  : 0.05 sec
   • Tổng độ dài audio     : 46.86 sec
   • T/g mỗi 1s audio      : 1.16 ms/sec
   • Trung bình mỗi file   : 2.72 ms
