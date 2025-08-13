## 0. Tập dữ liệu 
- Tập train và valid cũ
    - Dữ liệu cuộc gọi của vpb: bao gồm cả agent (người gọi) và user(người nghe)
    - Độ dài voice của agent khoảng gấp 5 lần của user 
    - Giọng của agent cơ bản ít người, dễ nghe, còn user thì đa dạng khó nghe 

- Tập voice user mới 
    - khoảng 2994 sample của user voice (ko có agent voice)
    - Chia ra và mix 1 phần vào tập train cũ, mix 1 phần vào tập valid cũ 


## Tập train mới - model gốc 
📊 Tổng số mẫu: 2303
🎯 WER trung bình (sample avg): 33.91%
🌐 WER toàn cục   (global):     23.90%

## Tập valid mới - model gốc 
📊 Tổng số mẫu: 587
🎯 WER trung bình (sample avg): 36.70%
🌐 WER toàn cục   (global):     27.43%


## Tập train mix và tập valid mix - model gốc 
📊 Tổng số mẫu: 5712
🎯 WER trung bình (sample avg): 28.07%
🌐 WER toàn cục   (global):     18.87%

📊 Tổng số mẫu: 967
🎯 WER trung bình (sample avg): 31.68%
🌐 WER toàn cục   (global):     22.83%



## 3. Tập train, tập test mix với model đã finetune (với tập train cũ)
🧪 Đánh giá mô hình trước khi fine-tune:
🎯 Dev WER (CTC): 36.64%
🌐 Global WER           : 31.08%
🕒 Evaluate time: 44.52s (avg decode/sample: 0.04s)

🧪 Đánh giá mô hình trên tập train:
🎯 Dev WER (CTC): 20.04%
🌐 Global WER           : 15.33%
🕒 Evaluate time: 258.66s (avg decode/sample: 0.04s)

## 4. Tập valid mix với model-finetune mới nhất (tập train cũ -> tập train mới)
💾 Đã lưu checkpoint: checkpoints_vpb_ctc/epoch20.pt
🎯 Dev WER (CTC): 33.85%
🌐 Global WER           : 29.54%


-----------

Quá trình fine-tune với tập train + valid cũ 

🧪 Đánh giá mô hình trước khi fine-tune:
🎯 Dev WER (CTC): 33.96%
🌐 Global WER           : 21.89%
🕒 Evaluate time: 13.65s (avg decode/sample: 0.03s)

🧪 Đánh giá mô hình trên tập train:
🎯 Dev WER (CTC): 31.97%
🌐 Global WER           : 20.00%
🕒 Evaluate time: 122.89s (avg decode/sample: 0.03s)


✅ Epoch 10 hoàn tất trong 20m54s
💾 Đã lưu checkpoint: checkpoints_vpb_ctc/epoch10.pt
🎯 Dev WER (CTC): 18.31%
🌐 Global WER           : 11.55%
🕒 Evaluate time: 17.31s (avg decode/sample: 0.04s)
