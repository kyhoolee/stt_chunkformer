Dựa vào số lượng sample của các dataset từ kết quả bạn gửi, ta có thể chia đều tương đối cho 4 máy xử lý dữ liệu song song. Tổng số sample theo manifest:

| Dataset         | Total Samples |
| --------------- | ------------- |
| fpt\_fosd       | 25,915        |
| infore          | 14,935        |
| lsvsc           | 56,823        |
| speech\_massive | 5,120         |
| vais1000        | 1,000         |
| vietmed         | 2,858         |
| vivos           | 12,420        |
| vlsp2020        | 56,172        |
| **Tổng cộng**   | **175,243**   |

Ta chia đều khoảng **43,800 samples** cho mỗi máy.

---

### ✅ Gợi ý chia máy:

#### Máy 1: \~43,300

```bash
python preprocess_entry.py --mode full --datasets lsvsc
```

#### Máy 2: \~43,300

```bash
python preprocess_entry.py --mode full --datasets vlsp2020
```

#### Máy 3: \~44,000

```bash
python preprocess_entry.py --mode full --datasets fpt_fosd vivos
```

#### Máy 4: \~44,600

```bash
nohup python preprocess_entry.py --mode full --datasets infore speech_massive vais1000 vietmed > log_vp4.txt 2>&1 & 

nohup python -m chunkformer_vpb.preprocess._5_1_small_ds_parallel --mode full --datasets infore speech_massive vais1000 vietmed --num_workers 8 > log_vp4.txt 2>&1 &


nohup bash -c 'PYTHONWARNINGS=ignore python -m chunkformer_vpb.preprocess._5_1_small_ds_parallel \
    --mode full --datasets infore speech_massive vais1000 vietmed --num_workers 16 \
    > log_vp4.txt 2>&1' &


```

---

### ✅ Notes:

* Giả sử file chạy là `preprocess_entry.py`, bạn thay bằng tên thực tế như `_5_1_small_ds_parallel.py` nếu cần.
* Có thể thêm `--num_workers 16` nếu bạn muốn tối ưu theo số core máy.
* Bạn có thể redirect log output nếu chạy ngầm bằng `nohup`:

  ```bash
  nohup python preprocess_entry.py --mode full --datasets lsvsc > log_lsvsc.txt 2>&1 &
  ```

---


tar -xf manifests.tar
tar -xf voice_data_8k.tar


# Lấy 10 file đầu tiên trong folder
ls -1 audio | head -n 10

# Lấy 10 file cuối cùng trong folder
ls -1 audio/infore/train/origin | tail -n 10
ls -1 audio/infore/train/vol | tail -n 10
ls -1 audio/infore/train/telephony | tail -n 10
ls -1 audio/infore/train/speed | tail -n 10
ls -1 audio/infore/train/noise | tail -n 10



python -m chunkformer_vpb.preprocess._5_2_big_ds_parallel --mode debug --dataset vi_voice

python -m chunkformer_vpb.preprocess._5_2_big_ds_parallel --mode benchmark


pkill -f "python -m chunkformer_vpb.preprocess._5_1_small_ds_parallel"

pkill -f "python -m chunkformer_vpb.preprocess._6_1_small_ds_no_queue"


nohup bash -c 'PYTHONWARNINGS=ignore python -m chunkformer_vpb.preprocess._6_1_small_ds_no_queue \
    --mode full --datasets infore speech_massive vais1000 vietmed --num_workers 16 \
    > log_vp4.txt 2>&1' &

nohup bash -c 'PYTHONWARNINGS=ignore python -m chunkformer_vpb.preprocess._6_1_small_ds_no_queue \
    --mode full --datasets fpt_fosd vivos --num_workers 16 \
    > log_vp3.txt 2>&1' &

nohup bash -c 'PYTHONWARNINGS=ignore python -m chunkformer_vpb.preprocess._6_1_small_ds_no_queue \
    --mode full --datasets vlsp2020 --num_workers 16 \
    > log_vp2.txt 2>&1' &

nohup bash -c 'PYTHONWARNINGS=ignore python -m chunkformer_vpb.preprocess._6_1_small_ds_no_queue \
    --mode full --datasets lsvsc --num_workers 16 \
    > log_vp1.txt 2>&1' &


PYTHONUNBUFFERED=1 python -m chunkformer_vpb.preprocess._6_1_small_ds_no_queue \
    --mode full --datasets vietmed --num_workers 8 --verbose 2


# 1) vi_voice
nohup env PYTHONWARNINGS=ignore python -m chunkformer_vpb.preprocess._6_2_big_ds_no_queue \
    --dataset vi_voice \
    --mode full \
    --num_workers 16 \
    --log_every 2000 \
    > log_vi_voice.txt 2>&1 &

# 2) viet_bud500
nohup env PYTHONWARNINGS=ignore python -m chunkformer_vpb.preprocess._6_2_big_ds_no_queue \
    --dataset viet_bud500 \
    --mode full \
    --num_workers 16 \
    --log_every 2000 \
    > log_viet_bud500.txt 2>&1 &

# 3) vietspeech
nohup env PYTHONWARNINGS=ignore python -m chunkformer_vpb.preprocess._6_2_big_ds_no_queue \
    --dataset vietspeech \
    --mode full \
    --num_workers 16 \
    --log_every 2000 \
    > log_vietspeech.txt 2>&1 &
