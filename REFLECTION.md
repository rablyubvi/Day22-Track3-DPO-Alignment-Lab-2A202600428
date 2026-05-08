# REFLECTION — Lab 22: Alignment với DPO/ORPO
**Tên:** _<Trần Ngô Hồng Hà>_
**Cohort:** _<A20-K1>_
**Tier đã chạy:** _<T4>_
**Date:** <2026-05-08>>_
**Track 3 · Ngày 22 · Chương trình AICB VinUni**

---

## 1. Tóm tắt tổng quan

Lab 22 là một **bài thực hành end-to-end** về **Preference Learning** sử dụng **DPO (Direct Preference Optimization)** để align một LLM nhỏ (Qwen2.5-3B) với các cặp dữ liệu tuỳ chọn từ UltraFeedback.

**Pipeline gồm 6 giai đoạn**:
1. **SFT-mini**: Build SFT adapter trên 1k Vietnamese Alpaca (initial policy)
2. **Preference data prep**: Load 2k UltraFeedback pairs (chosen/rejected)
3. **DPO training**: Huấn luyện DPO adapter, track reward curves
4. **Side-by-side eval**: Judge 8 prompts (4 helpfulness + 4 safety) với LLM
5. **Merge + GGUF**: Export deployable model (Q4_K_M quantization)
6. **Benchmark**: Đo IFEval, GSM8K, MMLU, AlpacaEval-lite

---

## 2. Những khái niệm chính đã học

### 2.1 Tại sao SFT không đủ?

SFT chỉ dạy model *tái tạo patterns* từ training data. Không có mechanism để học **xếp hạng** hay **prefer một style hơn style khác**. 

**Ví dụ**: 
- Prompt: "Xin chào"
- SFT learns: output "Xin chào, tôi là..."
- Nhưng nếu có 2 responses — một ngắn, một dài — SFT không biết cái nào "tốt hơn"

→ Đó là nơi **preference learning** vào cuộc.

### 2.2 Preference Learning & DPO

**Preference pairs** = (prompt, chosen_response, rejected_response)
- `chosen`: response mà human đánh giá cao (helpfulness, safety, accuracy)
- `rejected`: response kém chất lượng

**DPO (Direct Preference Optimization)**:
- Thay vì RLHF (train reward model → run RL), DPO **trực tiếp optimize** model
- Mục tiêu: tăng log-prob của chosen, giảm log-prob của rejected
- **KL divergence penalty** ngăn model drift quá xa khỏi SFT baseline
- **β parameter**: trade-off giữa reward tối ưu vs. staying close to SFT
  - β=0.05: reward lớn hơn, có thể unstable
  - β=0.1: balanced (default)
  - β=0.5: conservative, stable nhưng gap nhỏ

### 2.3 Reward curves & failure modes

Khi train DPO:
- **Ideal**: chosen_reward ↑, rejected_reward ↓ → gap > 0 ✓
- **Failure 1** (gap < 0): model học ngược. Nguyên nhân:
  - Data corrupt (chosen/rejected hoán đổi)
  - β quá cao
  - Learning rate quá thấp
- **Failure 2 - Likelihood Displacement**: gap > 0 nhưng chosen_reward ↓
  - Gap lớn vì rejected ↓ nhanh hơn chosen
  - Trade-off: learn ranking nhưng lose "overall quality" signal

### 2.4 Alignment Tax

Khi align model, ta thường trade-off giữa tasks:
- **IFEval ↑**: alignment làm tốt instruction-following (đúng mục đích)
- **GSM8K ↓**: capacity dành cho format thay vì math reasoning → "alignment tax" kinh điển
- **MMLU ~**: kiến thức nền thường không bị ảnh hưởng
- Đây không phải bug — đó là **thiết kế** của alignment

---

## 3. Quy trình thực thi chi tiết

### 3.1 Giai đoạn 1: SFT-mini

**Stack**: Unsloth + LoRA (r=16) + 4-bit quantization + 1k VN Alpaca

**Quy trình**:
1. Load base model (Qwen2.5-3B 4-bit via Unsloth)
2. Attach LoRA adapter (r=16, α=32, all MLP+attn modules)
3. Load dataset: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` (1k samples)
4. Format: instruction + input → ChatML conversation
5. Train:
   - lr=5e-5, epochs=1, batch=1×8 (gradient accum)
   - max_seq_length=512
   - Loss: cross-entropy (standard SFT)
6. Save adapter → `adapters/sft-mini/`

**Result**: ~2M trainable parameters, ~10 min train time (T4)

### 3.2 Giai đoạn 2: Preference Data Prep

**Stack**: HuggingFace datasets library

**Quy trình**:
1. Load `allenai/UltraFeedback` (Mixture split) → 2k samples
2. Parse each sample:
   - `conversations` = system + user messages (ChatML format)
   - `chosen` = preferred assistant response
   - `rejected` = non-preferred response
3. Format → `(prompt, chosen, rejected)` triple
4. Quality check:
   - Verify chosen ≠ rejected
   - Token length distribution: aim for ≥80% fit trong MAX_LEN=512
   - If too many > 512 tokens → truncated by trainer → loss of signal
5. Save → `data/pref/train.parquet` (full 2k) + `eval.parquet` (50 hold-out)

**Insight**: UltraFeedback là English → cho model Việt thực sự cần VN preference data. Bonus challenge: generate native VN pairs từ VMLU + 2 models + judge GPT-4o.

### 3.3 Giai đoạn 3: DPO Training (CORE STAGE)

**Stack**: TRL `DPOTrainer` + SFT adapter as reference policy

**Quy trình**:
1. Load:
   - Base model (4-bit)
   - SFT adapter from stage 1
   - DPO adapter (attach fresh LoRA)
2. Configure `DPOTrainer`:
   - `beta=0.1` (KL strength)
   - `learning_rate=5e-7` (very low — LoRA training on frozen base)
   - `num_epochs=1`
   - Loss: DPO loss = `log(sigmoid(β × (log π_dpo(y_chosen|x) - log π_sft(y_chosen|x))))`
3. Train on 2k pairs:
   - Effective batch = 1 × 16
   - ~30-45 min (T4)
4. **Monitor**:
   - `chosen_rewards` (nên ↑)
   - `rejected_rewards` (nên ↓)
   - `reward_gap = chosen - rejected` (nên > 0)
   - Training loss (should ↓)
5. **Post-train diagnostics**:
   - Gap > 0? → SUCCESS
   - Gap < 0? → FAILURE (debug: data quality, β, lr)
   - Gap > 0 nhưng chosen ↓? → Likelihood displacement (note it, not bug)
6. Save adapter + metrics JSON → `adapters/dpo/`

**Benchmark**: Final reward gap typically ~0.5-1.5

### 3.4 Giai đoạn 4: Side-by-side Eval

**Stack**: LLM-as-judge (GPT-4o-mini or Claude Haiku) or manual rubric

**8 eval prompts**:
- 4 helpfulness: QA không cần refuse
- 4 safety: xử lý sensitive topics

**Judge rubric** (each 1-5):
- Helpfulness: có trả lời câu hỏi không?
- Truthfulness: có thông tin sai không?
- Refusal appropriateness: nếu cần refuse thì refuse lịch sự? nếu không cần thì không refuse?

**Result**: Win/loss/tie breakdown, saved JSON

### 3.5 Giai đoạn 5: Merge + GGUF

**Quy trình**:
1. Load base (4-bit) + SFT + DPO adapters
2. Unmerge adapters → FP16 weights
3. Quantize → GGUF Q4_K_M (~1.5 GB cho 3B)
4. Smoke test via `llama-cpp-python`
5. Optional: vLLM serve (BigGPU tier)

### 3.6 Giai đoạn 6: Benchmark

**Benchmarks**:
- **IFEval**: instruction-following (50q sample)
- **GSM8K**: math reasoning (50q)
- **MMLU**: knowledge (50q, sampled)
- **AlpacaEval-lite**: LLM-judge on 100 prompts

**Expected pattern**:
- IFEval ↑ → DPO works
- GSM8K ↓ → alignment tax (normal)
- MMLU ~ → knowledge preserved
- AlpacaEval ↑ → preference transfer

---

## 4. Những insight kỹ thuật

### 4.1 Lý do chọn Qwen2.5-3B trên T4

- **Memory**: Base (7GB) + LoRA (0.5GB) + buffer (2-3GB) ≈ 10-12GB ≤ 16GB T4 limit ✓
- **Speed**: ~1 hour full pipeline (acceptable)
- **Quality**: Vẫn có reasonable instruction-following
- **Multilingual**: Native ChatML format, VN support

### 4.2 Unsloth + LoRA + 4-bit = cost-effective

- **4-bit**: giảm memory 50%
- **LoRA**: r=16 → 2M trainable params (thay vì 3B)
- **Unsloth kernels**: 2-3× throughput vs. standard
- **Combined**: T4-feasible

### 4.3 β sensitivity

**Expected curve**: β ↑ → reward_gap ↓ (but more stable)

**Experiment** (if doing rigor add-on):
- β=0.05: gap might be 1.2, but risky
- β=0.1: gap ~0.8, stable (default)
- β=0.5: gap ~0.3, very stable

### 4.4 Failure mode debugging

**If gap < 0**:
1. Check first 10 pairs: inspect chosen vs rejected text
2. Verify `tokenizer.pad_token = tokenizer.eos_token`
3. Plot step-by-step rewards → see where divergence starts
4. Reduce β, increase lr, or check data

---

## 5. Những thách thức & cách khắc phục

### 5.1 GPU memory

- **OOM during training**: reduce batch, MAX_LEN, enable gradient checkpointing ✓ (unsloth has built-in)
- **OOM during merge**: merge on CPU (slow but safe)

### 5.2 Data quality

- UltraFeedback English → English distribution → not optimal for VN model
- **Ideal**: collect native VN preference pairs
- **Feasible**: generate 200 VN prompts + 2 model outputs + GPT-4o judge

### 5.3 Alignment tax

- DPO might hurt GSM8K → trade-off, not bug
- Minimize by: choosing β carefully, training only on relevant categories
- Monitor: if MMLU drops > 5pp → catastrophic forgetting (reduce epochs)

### 5.4 Benchmark noise

- AlpacaEval-lite results vary by judge (GPT-4o vs Claude)
- MMLU: sampled → high variance
- **Mitigation**: rerun 3 times, report mean ± std

---

## 6. Suy ngẫm cá nhân

### Mạnh của lab

✓ **End-to-end**: from SFT → DPO → deploy (no missing piece)

✓ **Practical hardware**: T4 GPU, real constraints

✓ **Vibe-coding structure**: multiple add-ons encourage exploration

✓ **Clear failure mode diagnostics**: reward curves catch issues early

### Yếu (hoặc cơ hội cải thiện)

⚠ **English data**: UltraFeedback English → not optimal for VN alignment

⚠ **Benchmark sample size**: 50q per benchmark → high variance

⚠ **Manual judge eval**: effort-heavy, many skip it

→ **Cải thiện**: native VN preferences, full test set (if compute allows), auto 2-judge

### Nếu tiếp tục

1. **β-sweep** (+6 pts): run β ∈ {0.05, 0.1, 0.5} → plot curve
2. **2-judge eval** (+4 pts): OpenAI + Anthropic → compare
3. **Native VN preferences**: generate 200 VN prompts + benchmark
4. **ORPO** vs DPO: test alternative loss function

---

## 7. Kết luận

Lab 22 cung cấp **hands-on experience hoàn chỉnh** về preference learning & DPO alignment. Từ khái niệm (tại sao SFT không đủ?) đến thực hành (tuning β, interpret curves, deploy GGUF), mọi phần kết nối.

**Kỹ năng chính**:
- LoRA fine-tuning + 4-bit quantization
- DPO training with TRL
- LLM-as-judge evaluation
- Model deployment (GGUF)
- Benchmark interpretation (alignment tax)

**Insight sâu nhất**: Alignment = **reorienting model motivation** (từ "tái tạo text" → "follow instructions + stay safe"). Trade-offs là normal, không phải bug.

**Next**: áp dụng lên VN data, thử ORPO/SimPO/IPO variants.


## Điều ngạc nhiên nhất khi làm lab này

_(Optional, 1–3 câu)_
