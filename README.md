# LLM-based Medical Claim Optimization with SFT and Reinforcement Learning

> [!NOTE]
> **[繁體中文](#繁體中文)** | **[English Version](#english-version)**

<a name="繁體中文"></a>

# 基於 SFT 與強化學習的醫療診斷報告解讀與理賠流程優化

## 專案簡介

本專案導入大型語言模型（Large Language Model, LLM）微調技術，建構一套面向醫療理賠場景的 **自動化診斷報告解讀流程**。

專案結合 **監督式微調（Supervised Fine-Tuning, SFT）**、**強化學習微調（Reinforcement Learning, RL）**、**Prompt Engineering** 與 **知識蒸餾（Knowledge Distillation）**，使模型能夠從非結構化醫療診斷文本中理解病患病史、用藥紀錄與臨床描述，並輸出可用於後續理賠審核流程的 ICD-10 診斷代碼與分類結果。

相較於傳統模型在醫療文本理解與多標籤分類任務上的限制，本專案透過 **SFT + GRPO** 的訓練流程強化模型的醫療推理能力、輸出格式穩定性與判斷一致性，最終將分類表現提升約 **15%**。專案目標不僅是提升 ICD-10 預測準確率，也希望降低人工審核負擔，使模型結果能更穩定地銜接下游醫療理賠與人工覆核流程。

---

## 專案背景

醫療理賠與診斷編碼流程通常需要人工閱讀大量病歷、診斷報告與用藥紀錄，判斷病患狀況是否對應特定 ICD-10 診斷代碼或理賠分類。這類任務具有以下挑戰：

1. **醫療文本高度非結構化**
   臨床筆記、診斷描述與用藥紀錄常包含大量縮寫、上下文依賴與專業術語。

2. **多標籤分類難度高**
   單一病患可能同時對應多個 ICD-10 code，模型需要理解不同疾病、症狀與治療紀錄之間的關係。

3. **模型輸出難以直接落地**
   一般 LLM 雖能生成文字說明，但若輸出格式不穩定，將難以直接接入後續理賠審核或人工覆核流程。

4. **人工審核成本高**
   若模型無法穩定提供結構化判斷依據，仍需大量人工重新整理與確認，無法有效提升流程效率。

因此，本專案以醫療診斷報告自動解讀為核心任務，透過 LLM 微調與推論流程設計，提升模型在醫療分類、診斷編碼與理賠輔助決策上的可用性。

---

## 核心目標

* **醫療診斷報告自動解讀**
  從病歷、主訴、用藥紀錄與臨床描述中提取關鍵醫療資訊，判斷可能對應的 ICD-10 診斷代碼。

* **理賠分類與審核輔助**
  將模型輸出轉化為可用於後續理賠金額分類、人工審核與行政流程的結構化結果。

* **提升分類準確率**
  結合 SFT 與 RL 強化模型在醫療多標籤分類任務中的判斷準確度，最終分類表現提升約 **15%**。

* **提升生成式 AI 推論服務品質**
  透過 Prompt Engineering、知識蒸餾與輸出格式約束，強化模型在輸出格式、判斷準確度與推理一致性上的表現。

* **降低人工審核負擔**
  使模型結果能以標準化格式銜接下游人工覆核流程，提升醫療理賠自動化程度。

---

## 專案結構

```text
.
├── Qwen3 baseline/           # Qwen3 基礎模型的 baseline 測試代碼與結果
├── RL GRPO/                  # 使用 GRPO 演算法進行強化學習微調的實作代碼
├── SFT/                      # Supervised Fine-Tuning 監督式微調相關實驗代碼
├── medgemma baseline/        # 使用 Google MedGemma 模型作為對照組的測試代碼
├── gemini API baseline/      # 使用 Gemini API 進行測試的基準代碼
├── data preprocessing/       # MIMIC-IV 原始資料清洗與前處理腳本
├── train data construction/  # 使用 Gemini 2.5 Flash 建構模範回答與訓練集的腳本
├── results/                  # 存放模型回答與實驗結果
├── baseline_summary.ipynb    # 彙整各 baseline 模型表現的分析 notebook
├── note_icd_data.jsonl       # 處理後的臨床筆記與 ICD-10 對應數據
├── requirements.txt          # 專案依賴套件清單
├── requirements_sft.txt      # SFT 實驗依賴套件清單
└── ...
```

---

## 技術方法

### 1. 監督式微調與知識蒸餾

在強化學習之前，本專案先透過知識蒸餾與監督式微調建立模型的基礎醫療推理能力。

* **教師模型生成訓練資料**
  使用 **Gemini 2.5 Flash** 作為教師模型，根據臨床案例生成具備推理邏輯與標準化輸出的模範回答。

* **醫療推理能力蒸餾**
  將教師模型生成的診斷推理路徑、ICD-10 判斷依據與標準輸出格式蒸餾至 **Qwen3 14B** 模型中。

* **SFT 建立穩定輸出格式**
  透過監督式微調，使模型能穩定遵循指定格式輸出診斷代碼、推理依據與分類結果，為後續 RL 優化建立基礎。

* **高效微調實作**
  採用 **Unsloth** 框架進行訓練優化，降低顯存佔用並提升訓練效率，使較大規模模型能在有限硬體資源下完成微調。

---

### 2. 強化學習微調

在 SFT 模型基礎上，本專案進一步導入 **Group Relative Policy Optimization（GRPO）** 進行強化學習微調。

GRPO 不需要額外訓練 Critic Network，能降低強化學習微調的計算成本，適合應用於醫療診斷代碼預測這類具有多候選答案、多標籤輸出與複雜判斷依據的任務。

本專案設計的獎勵方向包括：

* **ICD-10 code 匹配準確度**
  鼓勵模型輸出與標準標註更一致的診斷代碼。

* **輸出格式正確性**
  確保模型輸出能被下游系統或人工審核流程直接解析。

* **醫療推理一致性**
  鼓勵模型根據病史、主訴、用藥紀錄與臨床描述給出合理判斷，而不是只輸出無解釋的代碼。

* **理賠流程可用性**
  使模型輸出能支援後續理賠分類、人工覆核與行政決策流程。

---

### 3. Prompt Engineering 與輸出格式約束

本專案透過 Prompt Engineering 將模型定位為「專業醫療編碼與理賠審核輔助員」，要求模型依序完成：

1. 閱讀病患臨床資訊
2. 分析病史、主訴與用藥紀錄
3. 判斷可能對應的診斷類別
4. 輸出 ICD-10 code
5. 給出可供人工審核的判斷依據
6. 以標準化格式輸出結果

透過固定輸出格式，模型結果能更容易被後續流程解析，降低人工重新整理與格式修正成本。

---

### 4. 實驗數據集

* **資料來源**：MIMIC-IV
* **資料內容**：重症監護病患的病史、主訴、用藥紀錄與標準 ICD-10 診斷代碼
* **任務形式**：根據非結構化臨床文本預測多個 ICD-10 code，並產生可供審核的診斷推理依據

---

## 實驗結果

本專案比較了不同模型與微調策略在 ICD-10 診斷代碼預測任務上的表現。

| Model / Method           | Performance（Average F1 Score） | Notes                      |
| :----------------------- | :---------------------------- | :------------------------- |
| **MedGemma 4B**          | ~0.1064                       | 醫療優化模型 baseline，但在本任務中表現有限 |
| **Qwen3 4B**             | ~0.1813                       | 通用模型 baseline              |
| **Qwen3 14B**            | ~0.2748                       | 較大參數模型具備更好的醫療文本理解能力        |
| **Qwen3 14B + SFT**      | ~0.2970                       | SFT 後輸出格式與診斷推理更穩定          |
| **Qwen3 14B + SFT + RL** | ~0.3127                       | RL 後醫療推理與分類表現進一步提升         |
| *Gemini 2.5 Flash*       | *~0.3357*                     | 教師模型與知識蒸餾來源                |

![不同模型與微調階段的 F1 Score 比較圖](./graph/14B%20GRPO%20vs%20SFT%20vs%20base%20vs%20gemini_v0.png)

*圖 1：不同模型在 Top-K 預測中的 F1 Score 表現比較。經過 SFT 與 RL 微調後，Qwen3 14B 的分類表現逐步提升，並接近 Gemini 2.5 Flash 教師模型。*

---

## 結果分析

### 1. SFT + RL 有效提升醫療分類表現

相較於原始 Qwen3 14B，加入 **SFT + RL** 後模型在 ICD-10 多標籤分類任務上的表現明顯提升。實驗中 F1 Score 從約 **0.2748** 提升至 **0.3127**，顯示強化學習能有效改善模型在醫療診斷報告解讀與分類判斷上的穩定性。

### 2. 知識蒸餾提升模型醫療推理能力

透過 Gemini 2.5 Flash 生成的高品質模範回答，SFT 階段成功將教師模型的診斷推理邏輯蒸餾至 Qwen3 14B，使模型能更穩定地理解病歷描述、用藥資訊與 ICD-10 code 之間的關聯。

### 3. Prompt Engineering 強化輸出可用性

透過系統提示詞與格式約束，模型不再只是生成自由文字，而是能輸出更接近下游流程需求的結構化結果。這使模型輸出能更容易銜接人工覆核、理賠分類與行政處理流程。

### 4. 強化學習改善判斷一致性

RL 階段透過獎勵函數進一步校準模型，使模型在輸出格式、code 匹配與推理依據上更一致。這對醫療理賠場景尤其重要，因為模型不僅需要「答對」，也需要提供可被審核人員理解與追溯的判斷依據。

---

## 專案貢獻

* 建立一套結合 **SFT、RL、Prompt Engineering 與知識蒸餾** 的醫療診斷報告解讀流程。
* 實現 ICD-10 診斷代碼自動預測，支援後續醫療理賠分類與人工審核流程。
* 透過 Gemini 2.5 Flash 教師模型生成高品質推理樣本，提升 Qwen3 14B 的醫療文本理解能力。
* 導入 GRPO 強化學習微調，提升模型在輸出格式、判斷準確度與推理一致性上的表現。
* 將模型結果轉化為可被下游流程解析的結構化輸出，提升生成式 AI 在醫療理賠場景中的可用性。
* 最終將模型分類表現提升約 **15%**，降低人工審核負擔並提升理賠流程效率。

---

## 引用與致謝

本專案使用 MIMIC-IV 資料庫，並結合 LLM 微調、知識蒸餾與強化學習方法進行醫療文本分類研究。

* **Data Citation**: Johnson, A., Bulgarelli, L., Pollard, T., ... Mark, R. (2023). MIMIC-IV. PhysioNet.
* **Technical References**:

  * **Unsloth**: 用於高效 SFT 與顯存優化。
  * **DeepSeek GRPO Algorithm**: 用於 Group Relative Policy Optimization 強化學習微調。
  * **Gemini 2.5 Flash**: 作為教師模型生成推理樣本與知識蒸餾資料。


---

<a name="english-version"></a>

# LLM-based Medical Claim Optimization with SFT and Reinforcement Learning

## Project Overview

This project develops an automated medical report interpretation workflow for medical claim optimization using large language model（LLM）fine-tuning. It integrates **Supervised Fine-Tuning（SFT）**, **Reinforcement Learning（RL）**, **Prompt Engineering**, and **Knowledge Distillation** to enable LLMs to interpret unstructured clinical notes, medication records, and diagnostic descriptions, and generate ICD-10 diagnostic codes and classification outputs that can support downstream claim review workflows.

Compared with traditional models that struggle with complex medical contexts and multi-label classification, this project adopts an **SFT + GRPO** training pipeline to improve medical reasoning, output format stability, and decision consistency. The final model improves classification performance by approximately **15%**, while reducing manual review burden and improving the practical usability of AI-assisted claim processing.

---

## Project Background

Medical claim review and diagnostic coding often require human reviewers to read large volumes of clinical notes, diagnosis reports, and medication records to determine whether a case corresponds to specific ICD-10 diagnostic codes or claim categories. This process is challenging for several reasons:

1. **Highly unstructured medical text**
   Clinical notes and diagnosis reports often contain abbreviations, implicit context, and domain-specific terminology.

2. **Complex multi-label classification**
   A single patient may correspond to multiple ICD-10 codes, requiring the model to understand relationships among symptoms, diseases, treatments, and medications.

3. **Limited downstream usability of raw LLM outputs**
   Even when an LLM can generate medical explanations, unstable output formats make it difficult to connect the results to claim review systems or human audit workflows.

4. **High manual review cost**
   Without stable structured outputs and interpretable reasoning, human reviewers still need to manually verify, reformat, and correct model results.

To address these challenges, this project focuses on automated medical report interpretation and uses LLM fine-tuning to improve diagnostic coding, claim classification support, and downstream review efficiency.

---

## Core Objectives

* **Automated Medical Report Interpretation**
  Extract key medical information from clinical notes, chief complaints, medication records, and diagnostic descriptions.

* **Claim Classification and Review Support**
  Convert model outputs into structured results that can support claim amount classification, manual review, and administrative workflows.

* **Improved Classification Accuracy**
  Combine SFT and RL to enhance model performance on medical multi-label classification tasks, achieving an approximately **15%** improvement in classification performance.

* **Enhanced Generative AI Inference Quality**
  Use Prompt Engineering, Knowledge Distillation, and output formatting constraints to improve format stability, judgment accuracy, and reasoning consistency.

* **Reduced Manual Review Burden**
  Produce standardized outputs that can be directly connected to downstream review workflows, improving the automation level of medical claim processing.

---

## Project Structure

```text
.
├── Qwen3 baseline/           # Baseline experiments and results using the Qwen3 base model
├── RL GRPO/                  # Reinforcement learning fine-tuning with the GRPO algorithm
├── SFT/                      # Supervised Fine-Tuning experiments
├── medgemma baseline/        # Baseline experiments using Google MedGemma
├── gemini API baseline/      # Baseline experiments using the Gemini API
├── data preprocessing/       # Scripts for cleaning and preprocessing raw MIMIC-IV data
├── train data construction/  # Scripts for constructing training data using Gemini 2.5 Flash exemplar answers
├── results/                  # Model outputs and experimental results
├── baseline_summary.ipynb    # Notebook summarizing and analyzing baseline model performance
├── note_icd_data.jsonl       # Processed clinical notes paired with ICD-10 codes
├── requirements.txt          # Project dependencies
├── requirements_sft.txt      # Dependencies specific to SFT experiments
└── ...
```

---

## Methodology

### 1. Supervised Fine-Tuning and Knowledge Distillation

Before reinforcement learning, this project first builds strong medical reasoning capabilities through knowledge distillation and supervised fine-tuning.

* **Teacher-generated training data**
  **Gemini 2.5 Flash** is used as a teacher model to generate exemplar answers with diagnostic reasoning, ICD-10 decision logic, and standardized output formats.

* **Medical reasoning distillation**
  The teacher model’s diagnostic reasoning paths, code selection logic, and structured outputs are distilled into **Qwen3 14B**.

* **Stable output formatting through SFT**
  Supervised fine-tuning enables the model to consistently follow the required output format, producing diagnostic codes, reasoning evidence, and classification results.

* **Efficient fine-tuning implementation**
  The **Unsloth** framework is used to optimize training efficiency, reduce GPU memory usage, and enable fine-tuning of larger models under limited hardware resources.

---

### 2. Reinforcement Learning Fine-Tuning

Building on the SFT model, this project further applies **Group Relative Policy Optimization（GRPO）** for reinforcement learning fine-tuning.

GRPO does not require an additional critic network, reducing the computational cost of RL fine-tuning. This makes it suitable for medical diagnostic coding tasks that involve multiple candidate codes, multi-label outputs, and complex reasoning criteria.

The reward design focuses on:

* **ICD-10 code matching accuracy**
  Encouraging the model to generate codes that better match the reference labels.

* **Output format correctness**
  Ensuring that outputs can be parsed by downstream systems or human review workflows.

* **Medical reasoning consistency**
  Encouraging the model to justify its predictions based on patient history, symptoms, medications, and clinical descriptions.

* **Claim workflow usability**
  Making model outputs more suitable for claim classification, manual review, and administrative decision-making.

---

### 3. Prompt Engineering and Output Constraints

The system prompt is designed to position the model as a professional medical coding and claim review assistant. The model is instructed to:

1. Read the patient’s clinical information
2. Analyze medical history, chief complaints, and medication records
3. Determine the corresponding diagnostic categories
4. Output ICD-10 codes
5. Provide reasoning evidence for human review
6. Return results in a standardized format

By enforcing structured outputs, the model results become easier to parse, verify, and integrate into downstream medical claim workflows.

---

### 4. Experimental Dataset

* **Source**: MIMIC-IV
* **Content**: ICU patient histories, chief complaints, medication records, and standard ICD-10 diagnostic codes
* **Task**: Predict multiple ICD-10 codes from unstructured clinical text and generate reasoning evidence for review

---

## Experimental Results

This project compares different models and fine-tuning strategies on the ICD-10 diagnostic coding task.

| Model / Method           | Performance（Average F1 Score） | Notes                                                                  |
| :----------------------- | :---------------------------- | :--------------------------------------------------------------------- |
| **MedGemma 4B**          | ~0.1064                       | Medical-optimized baseline model, but limited performance in this task |
| **Qwen3 4B**             | ~0.1813                       | General-purpose baseline model                                         |
| **Qwen3 14B**            | ~0.2748                       | Larger model with stronger medical text understanding                  |
| **Qwen3 14B + SFT**      | ~0.2970                       | More stable output format and diagnostic reasoning after SFT           |
| **Qwen3 14B + SFT + RL** | ~0.3127                       | Further improvement in medical reasoning and classification after RL   |
| *Gemini 2.5 Flash*       | *~0.3357*                     | Teacher model and knowledge source                                     |

![Comparison of F1 Scores across different models and fine-tuning stages](./graph/14B%20GRPO%20vs%20SFT%20vs%20base%20vs%20gemini_v0.png)

*Figure 1: Comparison of F1 Score performance across different models in Top-K prediction. After SFT and RL fine-tuning, Qwen3 14B shows consistent improvement and approaches the performance of the Gemini 2.5 Flash teacher model.*

---

## Result Analysis

### 1. SFT + RL improves medical classification performance

Compared with the original Qwen3 14B, the model fine-tuned with **SFT + RL** shows clear improvement on the ICD-10 multi-label classification task. The F1 Score increases from approximately **0.2748** to **0.3127**, indicating that reinforcement learning can improve the model’s stability in medical report interpretation and classification decisions.

### 2. Knowledge distillation improves medical reasoning

By using high-quality exemplar answers generated by Gemini 2.5 Flash, the SFT stage successfully distills diagnostic reasoning logic into Qwen3 14B. This helps the model better understand the relationship between clinical descriptions, medication records, and ICD-10 codes.

### 3. Prompt Engineering improves downstream usability

Through system prompts and output constraints, the model produces structured results instead of free-form text. This makes the outputs easier to connect to manual review, claim classification, and administrative processing workflows.

### 4. Reinforcement Learning improves decision consistency

During the RL stage, reward functions further calibrate the model’s behavior in terms of output format, code matching, and reasoning evidence. This is especially important in medical claim scenarios, where the model must not only generate correct predictions but also provide reviewable and traceable reasoning.

---

## Contributions

* Built an automated medical report interpretation workflow combining **SFT, RL, Prompt Engineering, and Knowledge Distillation**.
* Implemented ICD-10 diagnostic code prediction to support downstream medical claim classification and manual review workflows.
* Used Gemini 2.5 Flash as a teacher model to generate high-quality reasoning samples for knowledge distillation.
* Applied GRPO-based reinforcement learning fine-tuning to improve output format, judgment accuracy, and reasoning consistency.
* Converted model outputs into structured results that can be parsed by downstream workflows.
* Improved classification performance by approximately **15%**, reducing manual review burden and improving medical claim processing efficiency.

---

## Citation & Acknowledgements

This project uses the MIMIC-IV database and integrates LLM fine-tuning, knowledge distillation, and reinforcement learning for medical text classification.

* **Data Citation**: Johnson, A., Bulgarelli, L., Pollard, T., ... Mark, R. (2023). *MIMIC-IV*. PhysioNet.
* **Technical References**:

  * **Unsloth**: Used for efficient SFT and GPU memory optimization.
  * **DeepSeek GRPO Algorithm**: Used for Group Relative Policy Optimization.
  * **Gemini 2.5 Flash**: Used as the teacher model for reasoning sample generation and knowledge distillation.


