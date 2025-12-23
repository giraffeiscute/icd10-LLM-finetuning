# 運用大型語言模型微調技術提升 ICD-10 診斷編碼預測準確率
# Enhancing ICD-10 Diagnostic Coding Accuracy via LLM Fine-Tuning

## 📖 專案簡介 (Introduction)

本專案旨在探索如何透過 **大型語言模型 (LLM)** 的微調技術，從病患的臨床資訊（如病歷、用藥紀錄）中精準推估 **ICD-10 診斷代碼**。

傳統方法（如 BERT）在處理複雜醫療脈絡時往往面臨瓶頸（F1 Score 約 0.25），且缺乏對中文醫療專業知識的理解。本研究利用 **MIMIC-IV** 資料庫，結合 **強化學習 (GRPO)** 與 **監督式微調 (SFT)**，致力於提升醫療編碼工作的自動化效率與準確性。

### 核心目標
* **精準預測**：從非結構化臨床文本中提取特徵，準確預測 ICD-10 代碼。
* **效率提升**：減少人工編碼時間，輔助醫療行政流程。
* **技術探索**：驗證 GRPO (Group Relative Policy Optimization) 與監督式微調 (Supervised Fine-Tuning) 在多標籤分類任務中的優勢。

---

##  專案結構 (Project Structure)

本倉庫的檔案結構組織如下：

```text
.
├── Qwen3 baseline/           # Qwen3 基礎模型的 Baseline 測試代碼與結果
├── RL GRPO/                  # 本專案核心：使用 GRPO 演算法進行強化學習微調的實作代碼
├── SFT/                      # Supervised Fine-Tuning (監督式微調) 相關實驗代碼
├── medgemma baseline/        # 使用 Google MedGemma 模型作為對照組的測試代碼
├── gemini API baseline/      # 使用 Gemini API 進行 Zero-shot/Few-shot 測試的基準代碼
├── data preprocessing/       # MIMIC-IV 原始資料清洗與前處理腳本
├── train data construction/  # 建構訓練集、測試集與格式化數據的腳本
├── result/                   # 存放訓練過程的 Log、模型權重檢查點與評估圖表
├── baseline_summary.ipynb    # 彙整各 Baseline 模型表現的分析筆記本
├── note_icd_data.jsonl       # 處理後的臨床筆記與 ICD 對應數據
├── requirements.txt          # 專案依賴套件清單
└── ...
```
## 🚀 方法論 (Methodology)

本研究採用了多階段的優化策略，並重點比較了不同模型與訓練方法的效益：

### 1. 強化學習微調 (GRPO)
採用 DeepSeek 提出的 **GRPO (Group Relative Policy Optimization)** 演算法取代傳統的 PPO。

* **優勢**：大幅降低 GPU 運算資源需求，特別適合需要精確推理的複雜多標籤分類任務。
* **機制**：透過群組相對策略優化，加速模型收斂並改善預測品質。

### 2. 提示工程 (Prompt Engineering)
設計特定的 System Prompt 讓 LLM 扮演醫療編碼專家：

* **Chain of Thought (CoT)**：要求模型逐步推理（例如：「解釋每種藥物通常用於治療哪些疾病」），再輸出最終編碼。
* **格式控制**：確保輸出符合標準化格式以便於解析。

### 3. 實驗數據集
* **來源**：MIMIC-IV (MIT-LCP)
* **內容**：涵蓋重症監護病患的病史、主訴、詳細用藥紀錄及專業人員標註的 ICD-10 代碼。

---

## 📊 實驗結果 (Results)

我們比較了不同模型規模與方法在 ICD-10 預測任務上的表現 (F1 Score)：

| Model / Method | 表現 (Average F1 Score) | 備註 |
| :--- | :--- | :--- |
| **MedGemma 4B** | ~0.1064 | 針對醫療優化的基礎模型，但實際實驗之表現有限 |
| **Qwen3 4B** | ~0.1813 | 通用模型 Baseline |
| **Qwen3 14B** | ~0.2539 | 較大參數模型表現更佳 |
| **Qwen3 14B + SFT** | ~0.2734 | SFT 後模型推理格式相當穩定 |
| **Qwen3 14B + SFT + RL** | ~0.2958 | RL 後模型醫學知識推理表現更加提升 |


---


## 🛠️ 安裝與使用 (Getting Started)

### 環境需求
```bash
pip install -r requirements.txt
```
### 🛠️ 數據準備

請確保您擁有 **MIMIC-IV** 的存取權限，並將原始檔案放入 `data preprocessing` 指定的路徑中執行清洗腳本。

---

## 📝 引用與致謝 (Citation)

本專案使用了 **MIMIC-IV** 資料庫，並結合了高效的微調與強化學習技術。

* **Data Citation**: Johnson, A., Bulgarelli, L., Pollard, T., ... Mark, R. (2023). MIMIC-IV. PhysioNet.
* **Technical References**:
    * **Unsloth**: Used for efficient **SFT** (Supervised Fine-Tuning) and memory optimization during training.
    * **DeepSeek GRPO Algorithm**: Used for Group Relative Policy Optimization to enhance reasoning capabilities.
* **Last updated**: 2025/12
