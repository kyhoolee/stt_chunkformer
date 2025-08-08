Dưới đây là **tóm tắt đầy đủ, rõ ràng và áp dụng thực chiến** cho việc sử dụng **`multiprocessing` trong Python**, đặc biệt là các pattern dùng biến đúng cách trong môi trường nhiều tiến trình.

---

## ✅ 1. Tổng quan kiến trúc `multiprocessing`

Trong Python, khi bạn dùng `multiprocessing.Pool`, **mỗi process là một bản sao độc lập của process chính**:

* Không chia sẻ bộ nhớ trực tiếp (như thread),
* Mỗi process có **biến toàn cục riêng biệt**,
* Dữ liệu chỉ truyền qua:

  * **Argument khi gọi hàm**,
  * Hoặc **Queue / Pipe / SharedMemory (nâng cao)**.

---

## ✅ 2. Cách dùng phổ biến: `Pool` + `init_worker()`

```python
from multiprocessing import Pool

global_thing = None  # Biến toàn cục dùng trong từng process

def init_worker():
    global global_thing
    global_thing = SomeHeavyObject()

def worker_fn(item):
    global global_thing
    return global_thing.process(item)

if __name__ == "__main__":
    with Pool(processes=4, initializer=init_worker) as pool:
        results = pool.map(worker_fn, items)
```

### ✳️ Ý nghĩa:

* `init_worker()` sẽ chạy **1 lần duy nhất khi process khởi tạo**.
* `global_thing` trở thành **singleton object dùng lại cho toàn bộ tác vụ trong process đó**.

---

## ✅ 3. Các loại biến trong multiprocess

| Biến                               | Ý nghĩa                                                          | Cách dùng                                |
| ---------------------------------- | ---------------------------------------------------------------- | ---------------------------------------- |
| **Biến global trong main process** | Không dùng được trong worker                                     | Không có tác dụng trong process con      |
| **Biến global trong process con**  | Dùng được nếu khởi tạo qua `init_worker()`                       | Khai báo ở file level + gán trong `init` |
| **Biến truyền qua args**           | Từng `sample` truyền vào qua `map()`                             | Dùng tuple/list làm đối số hàm xử lý     |
| **Biến chia sẻ giữa process**      | Phức tạp – dùng `multiprocessing.Manager`, `Queue`, `Value`, ... | Ít dùng, chỉ khi cần đồng bộ/phối hợp    |

---

## ✅ 4. Khi nào nên dùng `init_worker + global`?

| Tình huống                                             | Nên dùng `init_worker()`? |
| ------------------------------------------------------ | ------------------------- |
| Bạn có object khởi tạo tốn kém (model, resampler, ...) | ✅ **Rất nên**             |
| Object chỉ cần dùng 1 lần rồi bỏ                       | ❌ Không cần               |
| Object cần reset lại mỗi lần xử lý                     | ❌ Không nên reuse         |
| Bạn dùng 1 model cho hàng ngàn lần xử lý               | ✅ Cần reuse               |

---

## ✅ 5. Lưu ý hiệu năng

* `init_worker()` cực kỳ hữu ích để **tránh khởi tạo lặp lại**, ví dụ:

  * Model ML,
  * Resampler,
  * Heavy config/parser,...
* Không nên tạo object nặng **bên trong hàm `process_one_sample()`** nếu có thể reuse.

---

## ✅ 6. Debug multiprocessing dễ hơn bằng cách

| Cách                                | Ý nghĩa                                          |
| ----------------------------------- | ------------------------------------------------ |
| `torch.set_num_threads(1)`          | Tránh conflict giữa `torch` và `multiprocessing` |
| Sử dụng `tqdm()` trong `debug`      | Xem được progress                                |
| Giới hạn số process khi debug       | Dễ trace hơn: `num_workers=1`                    |
| Ghi log từng process vào file riêng | Nếu log quá lộn xộn do concurrent prints         |

---

## 🧠 Tổng kết công thức

```python
# 1. Global scope
global_model = None

# 2. Worker init
def init_worker():
    global global_model
    global_model = load_model()

# 3. Worker function
def worker_fn(x):
    global global_model
    return global_model.infer(x)

# 4. Run
with Pool(processes=4, initializer=init_worker) as pool:
    results = pool.map(worker_fn, inputs)
```

---

Nếu bạn có use case cụ thể (VD: NLP, Audio, Blockchain, Vision...), mình có thể thiết kế cấu trúc multiprocessing riêng cho phù hợp 💡.
