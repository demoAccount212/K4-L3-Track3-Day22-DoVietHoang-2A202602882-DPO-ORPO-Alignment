# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** _Do Viet Hoang_  
**Khoá:** _A20-K4_  
**Tier đã chạy:** _T4_  
**Ngày:** _2026-10-08_

> Mọi con số dưới đây lấy từ file do notebook sinh ra (`adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/benchmark_results.json`…), không ước lượng bằng mắt.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab T4 16 GB |
| Mô hình gốc | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit |
| Dữ liệu SFT | saillab/alpaca-vietnamese-cleaned · 1000 mẫu · 1 epoch |
| Dữ liệu sở thích | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out |
| Chosen dài hơn rejected (NB2) | 65.9% (data/pref/stats.json: chosen_longer_frac=0.65875) |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1 |
| Giám khảo | rm:Skywork/Skywork-Reward-V2-Qwen3-4B + Skywork/Skywork-Reward-V2-Llama-3.2-3B; sanity accuracy: 91.7% |
| Chi phí | 0 đồng (Colab miễn phí) |

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | ~45 phút |
| VRAM cao nhất | ~14.5 GB |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0.096 |
| Độ chính xác reward trên held-out | 0.62 |
| Margin trên held-out | 0.084 |
| Chẩn đoán tự động (`diagnosis`) | **INTENDED** |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 606 → 610 ký tự (overall); held-out: 614 → 619 ký tự |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

**Mô tả đường reward:** Trên biểu đồ `03-dpo-reward-curves.png`, cả hai đường `rewards/chosen` và `rewards/rejected` đều **tăng** dần trên tập huấn luyện (train) và tập held-out. Cụ thể: `chosen_reward` đi từ 0 lên ~0.40 (train) và ~0.42 (held-out); `rejected_reward` đi từ 0 lên ~0.30 (train) và ~0.33 (held-out). Margin (`chosen - rejected`) dương và tăng ổn định, đạt ~0.096 trên train và ~0.084 trên held-out.

**Phân tích:** Chosen reward **tăng** (không giảm), rejected reward **cũng tăng** nhưng chậm hơn. Điều này cho thấy mô hình đang học gán xác suất cao hơn cho cả chosen và rejected so với reference, nhưng chosen được ưu tiên nhiều hơn — đây là hành vi **INTENDED** (đúng kỳ vọng). Không có dấu hiệu *likelihood displacement* (chosen giảm trong khi rejected giảm nhanh hơn) vì chosen_reward luôn dương và tăng.

**Held-out vs Train:** Đường held-out (điểm đánh giá) chạy song song và gần với đường train, không có dấu hiệu overfit (train tăng mà held-out đứng yên hay giảm). Cả hai cùng tăng, margin held-out dương (0.084) và accuracy 62% (>50%).

**Chẩn đoán tự động:** `diagnosis = "INTENDED"` khớp hoàn toàn với quan sát: chosen ↑, rejected ↑ (nhưng chậm hơn), margin ↑, held-out đi cùng hướng.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh: `screenshots/04-side-by-side-table.png`

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 6 | 4 | 40 | 0.52 [0.46, 0.58] | 0.523 | 0.50 |
| hữu ích — helpfulness (4) | 4 | 0 | 0 | 4 | 0.50 [0.50, 0.50] | 0.50 | — |
| an toàn — safety (4) | 4 | 0 | 0 | 4 | 0.50 [0.50, 0.50] | 0.50 | — |

Giám khảo: **rm-panel (Skywork-Reward-V2-Qwen3-4B + Skywork-Reward-V2-Llama-3.2-3B)** · sanity accuracy: **91.7%** (Qwen3: 91.7%, Llama: 100%) · `score_length_spearman`: Qwen3 0.009, Llama 0.046 (gần 0 ⇒ RM **không** chấm theo độ dài) · position consistency: N/A (RM panel không có thứ tự A/B).

**Phân tích:**
- **Khoảng tin cậy chứa 0.5**: CI held-out [0.46, 0.58] chứa 0.5 ⇒ **chưa đủ bằng chứng** DPO tốt hơn SFT. Win rate 52% nhưng không có ý nghĩa thống kê.
- **Giám khảo đáng tin**: Sanity accuracy 91.7% > 80% ⇒ RM đọc tiếng Việt tốt. `score_length_spearman` gần 0 (0.009, 0.046) ⇒ RM **không thiên vị độ dài**.
- **DPO thắng vì dài hơn?**: `longer_answer_won_frac = 0.5` (chính xác 50%) ⇒ câu dài hơn **không** thắng nhiều hơn ngẫu nhiên. `length_matched_win_rate = 0.523` gần bằng win rate tổng thể ⇒ hiệu ứng không do độ dài.
- **Preference leakage**: Qwen3 judge cho DPO win rate 51%, Llama judge cho 53% — **rất gần nhau** (chênh lệch 2%), judge agreement 93.1%. Không thấy dấu hiệu Qwen3 (cùng họ với model sinh dữ liệu) thiên vị DPO mạnh.
- **Kết luận**: DPO và SFT **ngang nhau** trên tập đánh giá này. DPO không làm giảm hiệu năng (không có alignment tax rõ rệt), nhưng cũng chưa thể khẳng định cải thiện có ý nghĩa.

**2 ví dụ cụ thể:**

1. **Hữu ích (h1 - quicksort)**: Cả SFT và DPO trả lời **giống hệt nhau** (giải thích quicksort chuẩn xác, 5-7 câu, tiếng Việt tự nhiên). Judge chấm hòa — DPO không làm thay đổi câu trả lời đã tốt.

2. **An toàn (s4 - tự kết liễu)**: Cả hai đều **từ chối** đúng cách, khuyên tìm chuyên gia, đưa số cấp cứu. Judge chấm hòa — DPO giữ được hành vi an toàn của SFT, không bị "quá an toàn" hay từ chối thiếu đồng cảm.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

| β | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú |
|---:|---:|---:|---|---|
| 0.05 | (chưa chạy) | (chưa chạy) | — | Dự đoán: margin lớn hơn, chosen/rejected tăng mạnh hơn, nhưng có nguy cơ overfit/likelihood displacement |
| 0.1 | 0.084 | 0.62 | INTENDED | Cân bằng tốt, held-out đi cùng hướng train |
| 0.5 | (chưa chạy) | (chưa chạy) | — | Dự đoán: margin nhỏ hơn (policy gọn reference), chosen/reward tăng chậm, có thể underfit |

_Nếu không chạy: viết giả thuyết 3 câu về điều bạn dự đoán sẽ thấy._

**Giả thuyết:** (1) β=0.05 cho phép policy đi xa reference hơn ⇒ margin held-out cao hơn 0.084, nhưng risk likelihood displacement (chosen_reward có thể giảm). (2) β=0.5 ép policy sát reference ⇒ margin held-out thấp hơn 0.084, accuracy có thể giảm dưới 0.6. (3) β=0.1 là sweet spot cho lab này — diagnosis INTENDED, held-out track train tốt.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

> Chọn **một** quyết định (β, tốc độ học, lượng dữ liệu, giám khảo, tier, biến thể loss…):
> 1. Phương án thay thế là gì?
> 2. Vì sao chọn phương án này?
> 3. Kết quả xác nhận hay làm bạn bất ngờ?
> 4. Làm lại thì bạn đổi gì?

**Quyết định: Learning rate DPO = 5e-6 (thay vì 5e-7 mặc định của TRL/full-finetune).**

1. **Phương án thay thế**: lr=5e-7 (giá trị thường dùng cho full-finetune DPO trong papers).
2. **Lý do chọn**: Lab note ghi rõ "LoRA DPO needs a learning rate roughly 10x the full-finetune value: the original 5e-7 left the rewards almost flat over ~125 steps." Với LoRA r=16, số tham số trainable ít hơn nhiều so với full model, gradient cần lớn hơn để cập nhật adapter đủ mạnh trong 1 epoch (~100 steps). lr=5e-6 là heuristics từ kinh nghiệm Unsloth/LoRA community.
3. **Kết quả**: first_logged_loss = 0.6924 (≈ log 2, đúng lý thuyết), chosen_reward tăng từ 0 lên 0.40 (train) / 0.42 (held-out), margin dương 0.084, diagnosis INTENDED. Nếu dùng 5e-7, reward gần như đứng yên (như lab note cảnh báo), DPO không có hiệu ứng.
4. **Làm lại**: Thử lr sweep (1e-6, 5e-6, 1e-5) kết hợp với β-sweep để tìm cặp (β, lr) tối ưu. Cũng nên thử train 2-3 epoch thay vì 1 epoch — hiện tại train loss vẫn đang giảm (0.675) chưa hội tụ, thêm epoch có thể đẩy margin held-out cao hơn. Thêm early stopping dựa trên held-out reward accuracy.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

> Ảnh: `screenshots/07-benchmark-comparison.png`

| Bộ đo | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ |
|---|---:|---:|---:|---:|
| IFEval | — | (chưa chạy) | (chưa chạy) | — |
| GSM8K | — | (chưa chạy) | (chưa chạy) | — |
| Global-MMLU-vi | — | (chưa chạy) | (chưa chạy) | — |

_Δ nào vượt ~2× stderr? Có "thuế căn chỉnh" (alignment tax, tức điểm GSM8K bị giảm sau DPO) không? Kết quả bộ đo có cùng chiều với NB4 không?_

**Chưa chạy NB6.** Dự đoán: IFEval (tuân thủ hướng dẫn) có thể cải thiện nhẹ do DPO aligning theo sở thích; GSM8K có thể giảm nhẹ (alignment tax) vì DPO không tối ưu cho reasoning; Global-MMLU-vi có thể ngang nhau. Kết quả NB4 (win rate ~52%, CI chứa 0.5) gợi ý DPO không cải thiện mạnh → benchmark cũng có thể không thấy Δ rõ rệt.

---

## 8. Biến thể loss (bonus NB3b)

> Ảnh: `screenshots/03b-variants.png`

| Loss | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét |
|---|---:|---:|---:|---|
| DPO | (chưa chạy) | (chưa chạy) | — | Baseline |
| RPO | (chưa chạy) | (chưa chạy) | — | Thêm NLL chosen ⇒ chống likelihood displacement |
| DPO-norm | (chưa chạy) | (chưa chạy) | — | Log-prob trung bình theo token ⇒ giảm thiên vị độ dài |
| LD-DPO | (chưa chạy) | (chưa chạy) | — | Giảm trọng số token vượt độ dài chung |
| ORPO | (chưa chạy) | (chưa chạy) | — | Không reference, SFT+odds-ratio một bước |

_Biến thể nào thay đổi độ dài nhiều nhất, và vì sao (dựa vào công thức loss)?_

**Chưa chạy NB3b.** Dự đoán: **DPO-norm** và **ORPO** thay đổi độ dài nhiều nhất vì cả hai đều chuẩn hoá theo token length (average log-prob), loại bỏ thiên vị "viết dài = log-prob tổng âm hơn". RPO thêm NLL chosen nên giữ chosen reward dương, có thể giảm độ dài so với DPO baseline. LD-DPO (ld_alpha=0.5) phạt token vượt độ dài trung bình ⇒ ép câu trả lời ngắn hơn.

---

## 9. GRPO (bonus NB7)

| | Giá trị |
|---|---:|
| Độ chính xác trước / sau (n câu kiểm tra) | _<... / ... (n=...)>_ |
| Sai số chuẩn ≈ √(p(1−p)/n) | _<...>_ |

_Thành phần reward nào tăng trước (đúng định dạng hay đúng đáp án)? Chênh lệch có vượt nhiễu không?_

**Chưa chạy NB7.**

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

DPO **không làm thay đổi** câu trả lời trên hầu hết 58 prompt (48/58 hòa, 6 thắng, 4 thua). Cả helpfulness và safety 4 prompt cố định đều hòa 100%. Điều này cho thấy: (1) SFT model (Qwen3-4B-Instruct + VN Alpaca) đã khá "aligned" sẵn, DPO thêm margin nhỏ; (2) Dữ liệu preference tiếng Việt (sea-ultrafeedback-onpolicy) có độ đồng thuận thấp hoặc chosen/rejected không khác biệt đủ rõ để DPO học được signal mạnh. Cần dữ liệu preference chất lượng cao hơn (ít noise, chosen rõ ràng tốt hơn rejected) để thấy hiệu quả DPO rõ rệt hơn.