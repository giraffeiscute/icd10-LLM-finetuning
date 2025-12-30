# 運用大型語言模型微調技術提升 ICD-10 診斷編碼預測準確率
> [!NOTE]
> **[繁體中文](#繁體中文)** | **[English Version](#english-version)**

<a name="繁體中文"></a>
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
├── SFT/                      # 本專案核心：Supervised Fine-Tuning (監督式微調) 相關實驗代碼
├── medgemma baseline/        # 使用 Google MedGemma 模型作為對照組的測試代碼
├── gemini API baseline/      # 使用 Gemini API 進行測試的基準代碼
├── data preprocessing/       # MIMIC-IV 原始資料清洗與前處理腳本
├── train data construction/  # 使用Gemini 2.5 Flash的模範回答建構訓練集的腳本 
├── results/                   # 存放所有模型回答跟實驗結果
├── baseline_summary.ipynb    # 彙整各 Baseline 模型表現的分析筆記本
├── note_icd_data.jsonl       # 處理後的臨床筆記與 ICD 對應數據
├── requirements.txt          # 專案依賴套件清單
├── requirements_sft.txt      # sft 專案依賴套件清單
└── ...
```
##  方法論 (Methodology)

### 1. 監督式微調與知識蒸餾 (SFT & Knowledge Distillation)
在進行強化學習之前，我們先透過知識蒸餾建立高品質的基礎能力：

* **數據合成**：利用 **Gemini 2.5 Flash** 作為教師模型，針對醫療案例生成具備詳細推理邏輯（Chain of Thought）的模範回答。
* **模型蒸餾**：將合成的推理數據輸入 **Qwen 14B** 進行監督式微調 (SFT)，使其吸收教師模型的醫療推理路徑與診斷邏輯。
* **技術實現**：採用 Unsloth 框架。透過其優化的核心與 Triton 算子，我們在微調過程中成功減少了約 70% 的顯存佔用，並提升了 2 倍以上的訓練速度。

### 2. 強化學習微調 (GRPO)
在 SFT 基礎上，採用 DeepSeek 提出的 **GRPO (Group Relative Policy Optimization)** 演算法：

* **優勢**：不同於傳統 PPO，GRPO 不需要額外的 Critic Network（評論家模型），節省了大量的運算資源，極其適合在有限顯存下處理具有多項候選代碼的複雜任務。
* **機制**：針對診斷代碼的 格式正確性、代碼匹配精準度 以及 推理邏輯的一致性 設定獎勵函數，強制模型在輸出時不僅提供正確編碼，還必須提供合理的臨床證據支持。

### 3. 提示工程 (Prompt Engineering)
設計專用的 System Prompt 將模型定位為「專業醫療編碼稽核員」：

* **CoT 引導**：強制模型必須先分析病史與用藥，最後再列出 ICD-10 Code。

* **輸出約束**：定義標準化的 JSON 或標籤格式，確保模型預測結果可被下游行政系統直接解析。

### 4. 實驗數據集
* **來源**：MIMIC-IV (MIT-LCP)
* **內容**：涵蓋重症監護病患的病史、主訴、詳細用藥紀錄及專業人員標註的標準 ICD-10 代碼。

---

##  實驗結果 (Results)

我們比較了不同模型規模與方法在 ICD-10 預測任務上的表現 (F1 Score)：

| Model / Method | 表現 (Average F1 Score) | 備註 |
| :--- | :--- | :--- |
| **MedGemma 4B** | ~0.1064 | 針對醫療優化的基礎模型，但實際實驗之表現有限 |
| **Qwen3 4B** | ~0.1813 | 通用模型 Baseline |
| **Qwen3 14B** | ~0.2748 | 較大參數模型表現更佳 |
| **Qwen3 14B + SFT** | ~0.2970 | SFT 後模型推理格式相當穩定 |
| **Qwen3 14B + SFT + RL** | ~0.3127 | RL 後模型醫學知識推理表現更加提升 |
| *Gemini 2.5 Flash* | *~0.3357* | *Teacher 模型 (知識來源)* |

下圖更詳細地展示了 Qwen-14B 在不同訓練階段（Base, SFT, RL）與 Gemini Pro 對照組，在不同預測數量 (Top-K) 下的 F1 Score 表現趨勢：

![不同模型與微調階段的 F1 Score 比較圖](./graph/14B%20GRPO%20vs%20SFT%20vs%20base%20vs%20gemini_v0.png)
*圖 1：不同模型在 Top-K 預測中的 F1 Score 表現比較。可見經過 RL (GRPO) 微調後的模型，在各項指標上均優於 SFT 與 Base 版本，並逐步逼近 Gemini Pro 的表現。*

 ** 結果分析**：
* **優化成效顯著**：
    Qwen3 14B 在加入 **RL (GRPO)** 強化學習後，F1 分數較原始 Base 版本提升了約 **13.8%**。這證明了強化學習能有效校準模型在處理複雜醫療任務時的決策品質。
* **知識蒸餾與格式穩定**：
    **SFT 階段**成功將 Gemini 的推理能力蒸餾至 Qwen 模型中，有效解決了醫療編碼中常見的輸出格式不穩定問題，為後續的 RL 優化奠定了堅實基礎。
* **RL 誘導之推理能力提升**：
    透過 RL 階段的獎勵機制（Reward Function），模型不僅學會遵守正確格式，更能深層理解**醫學病歷文本**與 **ICD-10 標準代碼**間的邏輯關聯，使推理表現進一步逼近 Gemini 2.5 Flash。
* **性能差距縮小**：
    經過優化的 Qwen3 14B 表現已有效接近商業強大模型。這證明了「**SFT + GRPO**」的訓練流程在醫療編碼等特定垂直領域中，具有極高的落地應用潛力。
---


##  安裝與使用 (Getting Started)

### 環境需求
```bash
pip install -r requirements.txt
pip install -r requirements_sft.txt
```
### 🛠️ 數據準備

請確保您擁有 **MIMIC-IV** 的存取權限，並將原始檔案放入 `data preprocessing` 指定的路徑中執行清洗腳本。

---

##  引用與致謝 (Citation)

本專案使用了 **MIMIC-IV** 資料庫，並結合了高效的微調與強化學習技術。

* **Data Citation**: Johnson, A., Bulgarelli, L., Pollard, T., ... Mark, R. (2023). MIMIC-IV. PhysioNet.
* **Technical References**:
    * **Unsloth**: Used for efficient **SFT** (Supervised Fine-Tuning) and memory optimization during training.
    * **DeepSeek GRPO Algorithm**: Used for Group Relative Policy Optimization to enhance reasoning capabilities.
* **Last updated**: 2025/12


<a name="english-version"></a>
# Enhancing ICD-10 Diagnostic Coding Accuracy via LLM Fine-Tuning

## 📖 Project Overview (Introduction)

This project explores how **large language model (LLM)** fine-tuning techniques can be used to accurately predict **ICD-10 diagnostic codes** from patients’ clinical information (such as clinical notes and medication records).

Traditional approaches (e.g., BERT-based models) often struggle with complex medical contexts, achieving limited performance (F1 score around 0.25), and lack sufficient understanding of Chinese medical terminology. Leveraging the **MIMIC-IV** database, this study integrates **Reinforcement Learning (GRPO)** and **Supervised Fine-Tuning (SFT)** to improve the automation efficiency and accuracy of medical coding.

### Core Objectives

* **Accurate Prediction**: Extract meaningful features from unstructured clinical text to precisely predict ICD-10 codes.
* **Efficiency Improvement**: Reduce manual coding time and support medical administrative workflows.
* **Technical Exploration**: Validate the effectiveness of GRPO (Group Relative Policy Optimization) and Supervised Fine-Tuning in multi-label classification tasks.

---

## 📂 Project Structure

The repository is organized as follows:

```text
.
├── Qwen3 baseline/            # Baseline experiments and results using the Qwen3 base model
├── RL GRPO/                   # Core component: reinforcement learning fine-tuning with the GRPO algorithm
├── SFT/                       # Core component: Supervised Fine-Tuning (SFT) experiments
├── medgemma baseline/         # Baseline experiments using Google MedGemma as a comparison model
├── gemini API baseline/       # Baseline experiments using the Gemini API
├── data preprocessing/        # Scripts for cleaning and preprocessing raw MIMIC-IV data
├── train data construction/   # Scripts for constructing training data using exemplar answers from Gemini 2.5 Flash
├── result/                    # Model outputs and experimental results
├── baseline_summary.ipynb     # Notebook summarizing and analyzing baseline model performance
├── note_icd_data.jsonl        # Processed clinical notes paired with ICD-10 codes
├── requirements.txt           # Project dependencies
├── requirements_sft.txt       # Dependencies specific to SFT experiments
└── ...
```

---

## 🧠 Methodology

### 1. Supervised Fine-Tuning & Knowledge Distillation (SFT)

Before reinforcement learning, we first establish strong baseline capabilities through knowledge distillation:

* **Data Synthesis**: Use **Gemini 2.5 Flash** as a teacher model to generate exemplar answers with detailed reasoning chains (Chain of Thought) for medical cases.
* **Model Distillation**: Feed the synthesized reasoning data into **Qwen 14B** for Supervised Fine-Tuning (SFT), enabling the model to absorb the teacher’s medical reasoning pathways and diagnostic logic.
* **Technical Implementation**: The **Unsloth** framework is adopted. With its optimized core and Triton kernels, GPU memory usage is reduced by approximately 70%, while training speed is increased by more than 2×.

### 2. Reinforcement Learning Fine-Tuning (GRPO)

Building on the SFT model, we further apply **GRPO (Group Relative Policy Optimization)** proposed by DeepSeek:

* **Advantages**: Unlike traditional PPO, GRPO does not require an additional critic network, significantly reducing computational overhead. This makes it particularly suitable for complex tasks with multiple candidate codes under limited GPU memory.
* **Mechanism**: Reward functions are designed to enforce **output format correctness**, **code matching accuracy**, and **consistency of clinical reasoning**, ensuring that the model not only outputs correct ICD-10 codes but also provides clinically plausible justifications.

### 3. Prompt Engineering

A dedicated system prompt is designed to position the model as a *professional medical coding auditor*:

* **Chain-of-Thought Guidance**: The model is required to first analyze patient history and medications, and only then output the ICD-10 codes.
* **Output Constraints**: Standardized JSON or tagged formats are enforced so that predictions can be directly parsed by downstream administrative systems.

### 4. Experimental Dataset

* **Source**: MIMIC-IV (MIT-LCP)
* **Content**: Includes ICU patient histories, chief complaints, detailed medication records, and professionally annotated ICD-10 codes.

---

## 📊 Experimental Results

We compare different model sizes and methods on the ICD-10 prediction task using the F1 score:

| Model / Method           | Performance (Average F1 Score) | Notes                                                             |
| :----------------------- | :----------------------------- | :---------------------------------------------------------------- |
| **MedGemma 4B**          | ~0.1064                        | Medical-optimized base model, but limited performance in practice |
| **Qwen3 4B**             | ~0.1813                        | General-purpose baseline model                                    |
| **Qwen3 14B**            | ~0.2514                        | Larger model achieves better performance                          |
| **Qwen3 14B + SFT**      | ~0.2775                        | More stable reasoning format after SFT                            |
| **Qwen3 14B + SFT + RL** | ~0.2958                        | Further improvement in medical reasoning after RL                 |
| *Gemini 2.5 Flash*       | *~0.3255*                      | *Teacher model (knowledge source)*                                |

![Comparison of F1 Scores across different models and fine-tuning stages](./graph/14B%20SFT%20vs%2014B%20base%20vs%20gemini_v0.png)
*Figure 1: Comparison of F1 Score performance across different models in Top-K predictions. It is evident that the model fine-tuned with RL (GRPO) outperforms both the SFT and Base versions across all metrics, progressively approaching the performance of Gemini Pro.*

### 💡 Result Analysis

* **Significant Optimization Impact**
    After incorporating **RL (GRPO)**, the F1 score of Qwen3 14B improved by approximately **13.8%** compared to the original Base version. This demonstrates that reinforcement learning can effectively calibrate the model's decision-making quality when handling complex and high-stakes medical tasks.

* **Knowledge Distillation & Structural Stability**
    The **SFT phase** successfully distilled reasoning capabilities from Gemini into the Qwen model. This stage was critical in resolving output format instability—a common challenge in automated medical coding—thereby establishing a stable foundation for subsequent RL optimization.

* **RL-Driven Reasoning Enhancement**
    Through the implementation of specialized **Reward Functions** during the RL phase, the model did not merely learn to adhere to formatting constraints; it developed a deeper cognitive grasp of the logical correlations between **unstructured medical record texts** and **standardized ICD-10 codes**. This advancement brought its reasoning performance significantly closer to that of Gemini 2.5 Flash.

* **Bridging the Performance Gap**
    The results indicate that the optimized Qwen3 14B has effectively narrowed the performance gap with state-of-the-art commercial models. This validates that the **"SFT + GRPO"** training pipeline holds immense potential for practical, privacy-compliant deployment in specialized vertical domains such as healthcare and medical insurance.
---

## 🚀 Getting Started

### Environment Setup

```bash
pip install -r requirements.txt
pip install -r requirements_sft.txt
```

### 🛠️ Data Preparation

Please ensure that you have authorized access to **MIMIC-IV**, and place the raw files into the designated paths under `data preprocessing` before running the cleaning scripts.

---

## 📚 Citation & Acknowledgements

This project utilizes the **MIMIC-IV** database and integrates efficient fine-tuning and reinforcement learning techniques.

* **Data Citation**: Johnson, A., Bulgarelli, L., Pollard, T., ... Mark, R. (2023). *MIMIC-IV*. PhysioNet.

* **Technical References**:

  * **Unsloth**: Used for efficient **SFT (Supervised Fine-Tuning)** and GPU memory optimization during training.
  * **DeepSeek GRPO Algorithm**: Used for Group Relative Policy Optimization to enhance reasoning capabilities.

* **Last updated**: 2025/12
