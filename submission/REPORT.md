# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Đức Thắng  **MSSV**: 2A202602605  **Ngày**: 2026-10-07
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4 16 GB (14.6 GB khả dụng, sm_75, fp16 — không có bf16)
**Adapter (HF Hub, public)**: https://huggingface.co/thangnd11/lab21-qwen35-4b-triage-lora  **Repo**: https://github.com/thangws4/Day21-Track3-NguyenDucThang-2A202602605-Finetuning-Lab

> Mọi con số dưới đây lấy từ `results/` (artefact chấm điểm) và `submission/logs/run.log`
> (log đầy đủ của lượt chạy NB1→NB5). Commit lab: `d27c1c0`. Chạy đầy đủ, **không** dùng
> `EVAL_LIMIT` (`baselines_frozen.json`: `eval_limit: null`, `smoke_mode: false`).

---

## 0. Lựa chọn thí nghiệm và lý do

| | Lựa chọn | Lý do |
|---|---|---|
| Base model | `unsloth/Qwen3.5-4B` (mặc định của tier T4) | Model lớn nhất train LoRA 16-bit vừa T4 (peak 8.78 GB, `runs.csv`). Giữ mặc định để phép so sánh với mốc (b) dùng **cùng một base**, và mask đã được lab kiểm chứng cho đúng template này. |
| Dataset | Corpus mặc định: 250 ticket CSKH tiếng Việt → JSON 4 trường (`intent`, `urgency`, `product`, `sentiment`) | Nhãn khách quan nên chấm tự động được từng trường, không cần LLM judge. Checksum tập eval giữ nguyên (`make verify`: "eval sets unmodified"). |
| Prompt (b) | `OPTIMIZED_PROMPT` gốc, **không sửa** (SHA `719e74d3b6232053`) | (b) đã thắng (a) rõ rệt; không có lý do làm yếu, cũng không chỉnh sau khi thấy kết quả. |

**Ghi chú về quá trình chạy (khai báo trung thực).** Lượt chạy đầu tiên bị mất khi runtime
Colab bị thu hồi giữa NB4. Lượt được nộp là lượt thứ hai, chạy lại **toàn bộ từ NB1** trên
runtime mới. Hai thay đổi chỉ là về hạ tầng, không đụng tới mã hay cấu hình của lab:
(1) chạy `scripts/colab_run.py` ở chế độ nền (`nohup`), log ghi ra `run.log`;
(2) `results/` và `adapters/` được symlink vào Google Drive để không mất khi runtime bị ngắt.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH → JSON triage (`data/train_seed.jsonl`) |
| Train / val | 225 / 25 (seed 42) |
| Eval | target 50 mẫu · regression 15 câu (`baselines_frozen.json`) |
| `max_length` | 1024 (giá trị của tier) — p95 đo được là **98**, p99 100, max 101; gợi ý của lab là 256 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / **30** cho cả 4 run (`runs.csv`, cột `max_steps`) |
| Batch hiệu dụng | 1 × 16 = 16 (< 32) |
| Precision | fp16 + GradScaler (T4 không có bf16) |

**Về `max_length` lệch với p95.** p95 = 98 nên mức gợi ý là 256; tier đặt 1024. Mình giữ 1024
vì trên corpus này giá trị đó **không ảnh hưởng gì**: chuỗi dài nhất chỉ có 101 token nên
không mẫu nào bị cắt, và `per_device_batch = 1` với packing tắt nên không phát sinh padding.
`max_length` chỉ là trần cắt. Nếu đổi sang dataset có câu trả lời dài hơn thì phải đo lại p95.

**Template có giữ khối `<think>` không?** **Có.** `template_check.json`: `ok: true`, `verdict:
"reasoning preserved — safe to train on traces"`. Chuỗi render vẫn giữ nguyên
`<think>\nbuoc 1: kiem tra. buoc 2: tra loi.\n</think>`. Tuy vậy, mọi câu trả lời train đều là JSON
trần, nên khối `<think>` trong vùng supervised luôn rỗng. Vì vậy `valid_trace_rate = 0.0`
(`verdict.json`) là đúng kỳ vọng, không phải lỗi.

**Kiến trúc (NB3, `run.log`).** `layer_types` của Qwen3.5-4B: 32 lớp gồm **24 `linear_attention`**
và **8 `full_attention`** (`full_attention_interval = 4`). `resolve_target_modules("text-linear")`
gắn LoRA vào 12 loại module của text decoder: `down_proj, gate_proj, in_proj_a, in_proj_b,
in_proj_qkv, in_proj_z, k_proj, o_proj, out_proj, q_proj, up_proj, v_proj`. Trong đó có các
projection của lớp linear-attention (`in_proj_*`, `out_proj`), và không có vision tower.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` (1 mẫu) | **0.4149** (39/94 token) |
| Toàn tập train (NB3) | 9014 / 20951 token = 43.0% |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn **được** tính loss (`mask_proof.json → supervised_preview`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đoạn **bị che** (`masked_preview`): system prompt, toàn bộ ticket của user, và `<|im_start|>assistant\n<think>\n\n`.

Đối chứng với `MASK_MODE=everything` (log NB1): 94/94 token (100%) bị tính loss, bao gồm
`<|im_start|>system … <|im_start|>user Alo shop, mình đặt balo laptop …`. Đó chính là bug khiến
model học viết lại câu hỏi. Mask `assistant-only` chỉ giữ phần đóng `</think>`, JSON và `<|im_end|>`,
nên model học được cả cách **dừng** (EOS).

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train) và bản fine-tune (NB5)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7911 | 0.000 | 3358.5 |
| (b) base + optimized prompt | **0.765** | **0.7911** | 1.000 | 1036.2 |
| (c) LoRA fine-tune (`correct`, naive prompt) | **0.970** | **0.6111** | 1.000 | 1460.6 |

*(a), (b): `baselines_frozen.json`; (c): `verdict.json → comparison`. n = 50 cho target.*

**(b) có thật sự mạnh hơn (a) không?** **Có**, rất rõ: target 0.000 → 0.765, format 0.000 → 1.000.
Prompt ngây thơ ("Phân loại ticket sau.") không nói rõ phải trả JSON, nên base model trả văn xuôi.
Format = 0 kéo target = 0. Câu trả lời dài cũng giải thích latency của (a) cao gấp ~3 lần (b).
Regression của (a) và (b) bằng nhau (0.7911) vì bài regression không dùng system prompt.

**Có sửa `OPTIMIZED_PROMPT` không?** Không. SHA `719e74d3b6232053` khớp bản gốc (`make verify`:
"baseline (b) prompt unmodified"). Mốc được đóng băng lúc NB2 ghi `baselines_frozen.json`, trước NB3.

---

## 4. Giải phẫu cấu hình sai (NB4, chấm ở NB5 §4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | train s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6265 | **0.970** | 405.8 | 8.78 |
| `attn_only` | q,v | 283 *(matched, α=566)* | 32,456,704 | 1e-4 | 0.5365 | **0.970** | 280.3 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.000** | 417.6 | 8.78 |
| `qlora` | text-linear (4-bit base) | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.940** | 505.2 | 3.86 |

*Nguồn: `runs.csv` (cột train loss = `final_loss`, là **trung bình** loss trên cả 30 step) và
`autopsy.json` (target, format). Format: 1.0 / 1.0 / 0.0 / 1.0. Cả 4 run cùng 30 step
(`make verify`: "all runs share ONE step budget"). Ngân sách `attn_only` lệch 0.025% so với `correct`.*

Mỗi run đối chứng chỉ đổi **một** biến so với `correct`:
`attn_only` đổi vị trí (q,v), với rank được nâng để giữ nguyên số tham số;
`wrong_lr` đổi LR (1e-4 → 1e-5);
`qlora` đổi độ chính xác của base (16-bit → 4-bit).

**Xếp hạng theo target:** `correct` = `attn_only` (0.970) > `qlora` (0.940) > `wrong_lr` (0.000).
**Xếp hạng theo train loss:** `attn_only` (0.5365) < `correct` (0.6265) < `qlora` (0.7058) < `wrong_lr` (1.5702).
**Hai thứ tự khác nhau ở vị trí đầu.** Theo loss, `attn_only` là cấu hình tốt nhất. Theo tác vụ, nó chỉ **hoà**.

Loss từng step (`submission/logs/run.log`):

| step | correct | attn_only | wrong_lr | qlora |
|---|---|---|---|---|
| 5 | 2.163 | 2.163 | 2.163 | 2.155 |
| 10 | 1.382 | 0.8234 | 2.066 | 1.731 |
| 15 | 0.1422 | 0.1459 | 1.606 | 0.2408 |
| 30 | 0.02571 | 0.02369 | 1.119 | 0.02622 |

**4.1 — Rank so với vị trí.** Với cùng ngân sách ~32.46 M tham số, `attn_only` (q,v, r = 283)
**hoà** `correct` (all-linear, r = 16) trên tập target: cả hai đạt 0.970, format 1.0. Thứ tự theo
train loss lại cho thấy `attn_only` "tốt hơn" (0.5365 so với 0.6265). Bảng loss từng step giải
thích vì sao: khác biệt nằm ở **tốc độ giảm loss sớm** (step 10: 0.82 so với 1.38). Đến step 15 hai
đường đã trùng nhau, và loss cuối gần như bằng nhau (0.024 so với 0.026). Nghĩa là `final_loss`
trung bình thưởng cho run nào hội tụ nhanh hơn, chứ không đo năng lực trên tác vụ. Với một tác vụ
hẹp, gần như chỉ là định dạng như triage JSON 4 trường, mình **không đo được** lợi thế của việc gắn
vào mọi lớp linear so với chỉ gắn q,v khi ngân sách đã bằng nhau. Vậy trên bài này, vị trí không
phải đòn bẩy. Ngân sách tham số và (như mục 4.2 cho thấy) LR mới quyết định. Kết luận "all-linear
thắng" của deck có thể đúng trên tác vụ khó hơn, nhưng số đo của mình không ủng hộ nó ở đây, và
mình không báo cáo một thứ tự mà dữ liệu không cho thấy. Ngoài lề: `attn_only` train nhanh hơn
(280 s so với 406 s) và sinh nhanh hơn (930.7 ms so với 1460.6 ms, `autopsy.json`) vì chỉ có 2 loại
module mang adapter.

**4.2 — `wrong_lr`.** Chỉ khác một con số (1e-5 thay vì 1e-4), nhưng sau 30 step loss vẫn là
**1.119**, trong khi `correct` đã xuống **0.026**, tức chênh khoảng 43 lần. Đường loss của
`wrong_lr` gần như phẳng (2.163 → 2.066 → 1.606 → 1.119), và token accuracy cuối chỉ 0.79. Hệ quả
trên tác vụ là target **0.000**, format **0.000**, latency 5468.6 ms: adapter chưa học được gì đủ để
thay prompt, nên với naive prompt nó cư xử giống hệt base (a) (target 0, format 0, trả lời dài).
Nếu chỉ nhìn đường loss mà không biết LR, mình sẽ kết luận sai rằng "LoRA r = 16 không đủ sức chứa"
hoặc "dữ liệu quá ít/nhiễu", rồi đi tăng rank hay thu thêm dữ liệu. Thực tế chỉ cần LR đúng thang
(≈ 10× LR full-FT) là cùng cấu hình đạt 0.970. Đây mới là **đòn bẩy lớn nhất** đo được trong lab.

**4.3 — `qlora`.** QLoRA giảm peak VRAM từ **8.78 GB xuống 3.86 GB** (−56%, tiết kiệm 4.92 GB).
Cái giá là target giảm **0.03** (0.970 → 0.940), train chậm hơn **24%** (505.2 s so với 405.8 s, do
dequantize mỗi bước), và latency suy luận cao hơn (1887.3 ms so với 1460.6 ms). Loss cuối gần như
không khác (0.02622 so với 0.02571), nên lại thêm một ví dụ loss không phản ánh chênh lệch trên tác
vụ. Số đo **ủng hộ một phần** khuyến nghị "đừng dùng QLoRA cho Qwen3.5": chất lượng có giảm nhưng
nhỏ (1.5 trên 50 mẫu, nằm trong vùng nhiễu của tập 50 mẫu). Lý do mạnh hơn là trên T4 16 GB, bản
16-bit đã vừa (8.78 GB), nên QLoRA đổi chất lượng và tốc độ lấy bộ nhớ mà mình không cần. QLoRA chỉ
đáng dùng khi bản 16-bit không vừa, ví dụ 9B trên T4.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **FAILED**
`target Δ = +0.205` · `regression Δ = −0.180` · `valid_trace_rate = 0.0`
Lý do trong `verdict.json`: *"general capability regressed by 0.180 (tolerance 0.020)."*

**Diễn giải.** Fine-tune **thắng rõ ở đúng tác vụ được huấn luyện**: target 0.970 so với 0.765 của
(b), format 1.0, và đạt được mà **không cần** prompt dài kèm few-shot. Nhưng nó **thua cổng hồi quy**
vì làm hỏng năng lực chung: keyword-recall trên 15 câu hỏi phổ thông giảm từ 0.7911 xuống 0.6111.
Mức giảm 0.180 gấp 9 lần ngưỡng cho phép 0.02.

Nguyên nhân hợp lý nhất là **quên thảm hoạ do dữ liệu một chiều**. 225 mẫu train đều cùng một dạng
(ticket → JSON 4 khoá), không có mẫu nào là hội thoại tự do. Sau 2 epoch, loss từng step xuống 0.026
và token accuracy 0.996, tức adapter đã gần như ghi nhớ hoàn toàn định dạng đó. Adapter được gắn vào
mọi projection của 32 lớp (gồm cả MLP), nên nó tác động lên cả đường sinh văn bản tự do. Bằng chứng
gián tiếp: NB5 chấm regression **không có system prompt**, vậy mà model vẫn bị kéo khỏi câu trả lời
tự nhiên.

Cần thận trọng với độ lớn con số. Tập regression chỉ có 15 câu, nên 0.18 tương đương khoảng 2.7 câu
về mặt keyword-recall. Đây là thước đo thô và có nhiễu. Nhưng mức giảm lớn gấp nhiều lần ngưỡng, nên
mình không coi đây là nhiễu. Mình cũng **không** nới ngưỡng, sửa tập eval hay đổi prompt (b).

Theo thứ tự chẩn đoán của NB5: format (1.0) và target (0.97) đều tốt, nên không phải lỗi mask hay
template. Target có tăng, nên LR đúng. Vấn đề nằm ở bước 2: **regression tụt, tức quên thảm hoạ**.
Cách sửa đúng là trộn thêm 1–5% dữ liệu phổ thông (replay) vào tập train (deck §6.3), có thể kết hợp
giảm xuống 1 epoch. Kết luận về mặt sản phẩm: bản fine-tune này **chưa nên deploy** như một model
dùng chung. Nếu triển khai, nó chỉ phù hợp làm adapter **chuyên dụng** gắn khi request là triage
(hot-swap ở mục 8), còn câu hỏi phổ thông đi qua base không có adapter.

---

## 6. Định tính — có cả ca THUA

Nguồn: `results/qualitative.json` (đã sắp từ tệ đến tốt; 44/50 mẫu đạt 1.0, 6/50 mẫu đạt 0.75)
cùng nhãn trong `data/eval_target.jsonl`. NB2 không lưu output (b) theo từng mẫu, nên cột so sánh
là **nhãn đúng**. Điểm tổng của (b) là 0.765.

| # | i | Ticket (rút gọn) | Nhãn đúng | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | 0 | "…chuột không dây … Cho tôi trả lại. **Gấp.** Shop hỗ trợ tốt." | doi_tra · **cao** · tich_cuc | doi_tra · cao · tich_cuc | ✅ đúng 4/4. "Gấp" → cao, và tách được sentiment tích cực dù đang đòi trả hàng |
| 2 | 47 | "…ốp lưng … Shipper không gọi. **Hỏi cho biết thôi.** Shop hỗ trợ tốt." | van_chuyen · **thap** · tich_cuc | van_chuyen · thap · tich_cuc | ✅ đúng 4/4. Nhận ra cue "thấp" ("hỏi cho biết thôi") |
| 3 | 48 | "…ốp lưng … Giá bao nhiêu. Mong shop phản hồi." | hoi_thong_tin · trung_binh · trung_tinh | hoi_thong_tin · trung_binh · trung_tinh | ✅ đúng 4/4 |
| 4 | 3 | "…bình giữ nhiệt … Chưa thấy tiền. **Khi nào tiện.** Cảm ơn shop nhiều." | hoan_tien · **thap** · tich_cuc | hoan_tien · **trung_binh** · … | ❌ **FT thua** (0.75): sai urgency |
| 5 | 39 | "…nồi chiên không dầu … Hoàn tiền. **Khi nào tiện.** Quá tệ." | hoan_tien · **thap** · tieu_cuc | hoan_tien · **trung_binh** · … | ❌ **FT thua** (0.75): sai urgency |
| 6 | 41 | "…đèn bàn LED … Giao hàng chậm. **Khi nào tiện.** Cảm ơn shop nhiều." | van_chuyen · **thap** · tich_cuc | van_chuyen · **trung_binh** · … | ❌ **FT thua** (0.75): sai urgency |

**Mẫu chung ở các ca FT thua.** Cả **6/6** lỗi (i = 3, 5, 12, 39, 41, 46) giống hệt nhau: ticket chứa
cụm **"Khi nào tiện."**, nhãn là `urgency = thap`, model đoán `trung_binh`. Ba trường còn lại đều
đúng ở cả 6 ca. Điều này không đến từ việc thiếu dữ liệu: trong `train_seed.jsonl` cụm này xuất hiện
**35 lần, cả 35 lần đều là `thap`**, và cả 6 ticket có cụm này trong tập eval đều là `thap`. Model
vẫn nhận đúng các cue "thấp" khác (ca #2), nên đây là một lỗi hệ thống của một cue cụ thể. Giả thuyết
của mình: "Khi nào tiện" về nghĩa đen là một câu hỏi thời gian. Prior của base có thể xếp nó vào mức
"trung bình", và 30 step chưa đủ để ghi đè prior đó. Muốn kiểm chứng cần lưu output (b) theo từng mẫu
để xem base đã prompt có mắc cùng lỗi không, và thử thêm 1 epoch. Lỗi tập trung một chỗ như vậy
cũng là lý do target dừng ở 0.97 thay vì 1.0.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Câu hỏi của lab là: bản fine-tune có thắng base model đã được prompt tử tế không, và
mình có phát hiện được nếu nó không thắng không? Câu trả lời có hai mặt, và chính cổng bốn nhóm cho
phép thấy cả hai. Trên tác vụ đích, fine-tune thắng thật: target tăng từ 0.765 lên 0.970, format
tuyệt đối, và đạt được với một prompt ngắn. Nếu chỉ đo target hay perplexity, mình sẽ kết luận "ship
được". Nhưng nhóm regression cho thấy cái giá: năng lực chung giảm 0.180, gấp 9 lần ngưỡng. Lý do
là 225 mẫu cùng một khuôn, train tới loss 0.026, đã kéo cả mô hình về phía JSON. Vì vậy phán quyết
FAILED là đúng, và mình **không** deploy bản này như model dùng chung.

Trong lab này, đòn bẩy thật sự là **mask và learning rate**, không phải vị trí hay rank. Mask sai thì
mọi thứ sau đều vô nghĩa. Mask đúng (41% token supervised) cho format 1.0 ngay sau 30 step. LR sai
thang biến cùng một cấu hình từ 0.970 thành 0.000. Ngược lại, khi giữ ngân sách tham số bằng nhau,
đổi vị trí từ all-linear sang q,v không tạo ra khác biệt đo được trên tác vụ này. QLoRA chỉ đổi 0.03
target lấy 4.9 GB VRAM mà T4 không thiếu. Bước tiếp theo hợp lý là thêm 1–5% dữ liệu replay để kéo
regression về trong ngưỡng, rồi chấm lại trên **cùng** mốc đã đóng băng.

**Ba điều tôi học được** (cụ thể, từ số đo của chính lượt chạy này):
1. **Loss giảm không đồng nghĩa model tốt hơn.** Trong thí nghiệm, `attn_only` có train loss
   **0.5365**, thấp hơn `correct` (**0.6265**), nhưng cả hai đều đạt target **0.970**
   (`runs.csv`, `autopsy.json`). Chênh lệch loss chỉ đến từ 10 step đầu (step 10: 0.8234 so với
   1.382), còn đến step 30 hai đường gần như trùng nhau (0.02369 so với 0.02571). Vì vậy tôi xếp hạng
   cấu hình bằng điểm trên tác vụ, không bằng loss.
2. **Kiểm tra loss mask là bước bắt buộc** để đảm bảo mô hình chỉ học phần câu trả lời mong muốn.
   Với `assistant-only`, chỉ **41.5%** token (39/94, `mask_proof.json`) được tính loss, gồm JSON và
   `<|im_end|>`. Với `everything`, con số là **100%** (94/94), tức model học luôn cả system prompt
   và ticket. Nhờ mask đúng, bản fine-tune đạt format **1.000** chỉ sau 30 step.
3. **Fine-tuning có thể cải thiện mạnh tác vụ chuyên biệt nhưng đồng thời làm giảm khả năng tổng quát
   của model**, vì vậy cần đánh giá nhiều tiêu chí trước khi triển khai. Target tăng **+0.205**
   (0.765 → 0.970) nhưng regression giảm **−0.180** (0.7911 → 0.6111), gấp 9 lần ngưỡng 0.02
   (`verdict.json`). Nếu chỉ nhìn target, tôi đã kết luận sai là nên ship.

**Nếu có thêm 2 giờ nữa, tôi sẽ:** thiết kế trước một thử nghiệm replay 1–5% dữ liệu phổ thông
trong tập train, giữ nguyên eval cũ làm mốc đóng băng, rồi thêm một tập đánh giá mới độc lập để
kiểm tra regression có về trong ngưỡng 0.02 không. Tôi cũng sẽ đo nhiều seed và sai số quanh target
thay vì chỉ so một con số duy nhất. Với 50 mẫu target, chênh 0.03 giữa `correct` và `qlora` chỉ tương
đương 1.5 mẫu. Ngoài ra, tôi sẽ sửa NB2 để lưu output (b) theo từng mẫu, rồi kiểm tra xem base đã
prompt có mắc cùng lỗi "Khi nào tiện" không. Tôi sẽ không chỉnh tiêu chí để cứu verdict.

---

## 8. Bonus — NB6: merge + hot-swap (B1)

`results/merge_check.json`: target **trước merge 0.97 → sau merge 0.97**, Δ = **0.0** (ngưỡng 0.01, n = 50).
Merge `W = W₀ + (α/r)·BA` bảo toàn điểm chính xác.

**Hot-swap ≥ 2 adapter.** Phần §3 của NB6 bị lỗi trong lượt chạy (`submission/logs/nb6.log`):
`ValueError: We need an offload_dir…` khi `load_adapter`. Nguyên nhân là notebook chỉ `del merged`,
còn biến `model` (PeftModel trước merge) vẫn giữ base trên GPU, nên khi `load_base` lần hai, VRAM
chứa hai bản 4B và accelerate đòi offload. Mình chạy lại **đúng logic §3** trong một process mới
(`submission/logs/hotswap.py`, output trong `submission/logs/hotswap.log`). Kết quả: nạp
**3 adapter (`correct`, `attn_only`, `qlora`) lên cùng một base** và chuyển bằng `set_adapter`. Với
ticket "…chuột không dây … Cho tôi trả lại. Gấp. Shop hỗ trợ tốt.", cả ba adapter đều trả về
`{"intent": "doi_tra", "urgency": "cao", "product": "chuột không dây", "sentiment": "tich_cuc"}`,
khớp nhãn. Lưu ý: adapter `qlora` được nạp lên base 16-bit đúng như NB6 gốc làm, không phải base
4-bit mà nó đã được train.
