#!/usr/bin/env python
# coding: utf-8

# In[3]:


import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
import numpy as np


# In[5]:


def extract_icd_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "").replace(".", "")


# In[ ]:


import json
import re

SYSTEM_PROMPT = """你是一名專門根據患者用藥紀錄預測 ICD-10 編碼的醫療專家。
請始終嚴格遵守以下格式回答：

<reasoning>
逐步推理：解釋每種藥物通常用於治療哪些疾病，並將其對應到可能的 ICD-10 編碼。
在本段最後，決定最合理的 ICD-10 編碼。
</reasoning>
<answer>
僅輸出最終的 ICD-10 編碼（用逗號分隔）。不要添加任何額外文字。
</answer>

如果無法判定，僅在 <answer> 中輸出 "NA"。
"""


EXAMPLE_QUESTION = "Patient 12078629 used the following medications: Iso-Osmotic Dextrose, Bisacodyl, Influenza Vaccine Quadrivalent, Senna, Oseltamivir, Vial, Cefepime, Vancomycin, Lactated Ringers, Aspirin, Vitamin D, Loperamide, Benzonatate, Guaifenesin, 0.9% Sodium Chloride, Potassium Chloride, Potassium Acetate, Phosphorus, Insulin, Heparin, Cefpodoxime Proxetil, Sodium Chloride 0.9% Flush. What ICD-10 codes should be assigned? Please just tell me which codes in the end"
EXAMPLE_REASONING = """Iso-Osmotic Dextrose could imply either dehydration (E86.0) or hypoglycemia (E16.0), though it's tricky.
Bisacodyl, Senna, and Loperamide relate to bowel issues like constipation (K59.0) or diarrhea (R19.7).
Pneumonia and UTI could be linked to antibiotics (J18.9, N39.0).
Flu vaccines and Oseltamivir fit into specific codes like Z23, J10.x.
For potassium chloride and related electrolytes, I’m looking at E87.6 for hypokalemia and E83.3 for hypophosphatemia. For insulin, it’s E10.x for type 1 or E11.x for type 2 diabetes. For heparin, Z79.01 could apply for long-term use or possibly I82 for thrombosis. Aspirin might map to I25.10 for coronary artery disease prevention.
Aspirin likely maps to I10 or I25.10 for cardiovascular prevention, not pain/fever. Vitamin D is E55.9 for deficiency. Benzonatate and Guaifenesin both map to R05.9 for cough. For antibiotics, Cefepime and Vancomycin could be linked to A41.9 (sepsis) or J18.9 (pneumonia). Heparin may be Z79.01 or Z29 for prophylaxis.
Answer:"""
EXAMPLE_ANSWER = "J10.1, J11.1, A41.9, J18.9, N39.0, E87.6, E11.9, I10, I25.10, R19.7, E55.9"

chat_style_data = []

with open("patient_icd_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        chat_style_data.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": EXAMPLE_QUESTION},
                {"role": "assistant", "content": f"<reasoning>{EXAMPLE_REASONING}</reasoning><answer>{EXAMPLE_ANSWER}</answer>"},
                {"role": "user", "content": item["question"]}
            ],
            "answer": extract_icd_answer(item["answer"])  # ground truth ICD-10 codes
        })


# In[ ]:


# 用法
dataset = chat_style_data


# In[8]:


#reward中會用到的function
def extract_xml_answer(text: str) -> str: #提取出答案
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    answer = answer.strip()
    answer = answer.replace(".", "") # 移除 ICD-10 編碼中的小數點，例如 Z45.81 → Z4581
    return answer

def is_text(text: str) -> bool:
    """
    判斷輸入字串是否主要為文字敘述（而非 ICD code 等短編碼）

    text: 輸入的字串

    回傳:
        True  -> 主要是文字敘述
        False -> 主要不是文字（例如短編碼）
    """
    # 計算字串中英文字母或空白的總數
    letters_and_spaces = sum(c.isalpha() or c.isspace() for c in text)

    # 計算文字/空白比例，避免除以 0
    ratio = letters_and_spaces / max(1, len(text))  

    # 如果文字/空白比例超過 0.5，視為文字敘述
    if ratio > 0.5:  
        return True  
    else:
        return False

def compute_f1(pred_set: set[str], true_set: set[str]) -> float:
    """
    pred_set: 模型預測的 ICD-10 code 集合
    true_set: 真實答案的 ICD-10 code 集合
    """
    if not pred_set and not true_set:
        print("⚠️ Both prediction and ground truth are empty, F1=1.0")
        return 1.0
    tp = len(pred_set & true_set)   # true positive
    fp = len(pred_set - true_set)   # false positive
    fn = len(true_set - pred_set)   # false negative

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1

def strict_format_check(prediction: str, ground_truth: str) -> bool:
    """
    嚴格格式檢查，用於 reward 計算前的 F1 安全檢查
    回傳 True -> 可以安全計算 F1, False -> 無法計算 F1
    """
    try:
        # 將字串拆成集合，支援逗號或空格分隔
        def parse_codes(text):
            # 如果包含逗號，用逗號分隔；否則用空格分隔
            if ',' in text:
                return set(code.strip() for code in text.split(',') if code.strip())
            else:
                return set(code.strip() for code in text.split() if code.strip())
        
        pred_set = parse_codes(prediction)
        true_set = parse_codes(ground_truth)
        
        # 檢查格式是否符合 ICD-10（字母開頭 + 2~4 位數字）
        # 修復：調整正則表達式以符合實際的 ICD-10 格式
        icd_pattern = re.compile(r'^[A-Z][0-9A-Z]{2,6}([.][0-9A-Z]+)?$')
        
        # 檢查所有 code 是否符合格式
        for code in pred_set:
            # 保持原始格式進行檢查，不要移除點號
            if not icd_pattern.match(code):
                print(f'strict_format_check fail: {code[:20]}')
                return False
        
        # 嘗試計算 F1（檢查是否能計算，不真正使用結果）
        # 修復：將 * = compute*f1 改為正確的函數調用
        f1_score = compute_f1(pred_set, true_set)
        return True
        
    except Exception as e:
        print("Error in strict_format_check:", e)
        print('strict_format_check error')
        return False


# In[ ]:


def accuracy_func(prompts, completions, answer, **kwargs) -> list[float]:
    """
    計算 F1-score 作為 reward
    規則：
    1. 如果回答是文字敘述或格式不符 ICD-10，直接給 0
    2. 只要前三碼相同就算答對
    """

    # 模型輸出文字
    responses = [completion[0]['content'] for completion in completions]

    # 取得問題（debug 用）
    q = prompts[0][-1]['content']

    # 提取 XML 中的回覆
    extracted_responses = [extract_xml_answer(r) for r in responses]

    # Debug 輸出
    print('-'*20)
    print(f"Question:\n{q}")
    print(f"\nAnswer:\n{answer[0]}")
    print(f"\nResponse:\n{responses[0]}")
    print(f"\nExtracted:\n{extracted_responses[0]}")

    f1_scores = []

    for r, a in zip(extracted_responses, answer):

        # 移除答案前綴
        if a.startswith("ICD10 編碼："):
            a = a.replace("ICD10 編碼：", "")

        # 文字敘述檢查 + 嚴格格式檢查
        if is_text(r) or not strict_format_check(r, a):
            f1_scores.append(0.0)
            continue

        # 處理答案與模型輸出
        a_codes = [code.strip()[:3] for code in a.split() if code.strip()]
        r_codes = [code.strip()[:3] for code in r.split(',') if code.strip()]

        a_set = set(a_codes)
        r_set = set(r_codes)

        # 計算 F1-score 並乘 20
        f1_scores.append(compute_f1(r_set, a_set) * 30)

    # Debug 輸出
    print('--------------------')
    print("F1 scores:", f1_scores)
    print('--------------------')

    return f1_scores

def soft_format_reward_func(completions, **kwargs) -> list[float]: #只要不是一堆文字敘述就給分
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    rewards = []
    for r in extracted_responses:
        if is_text(r):  
            rewards.append(0.0)   # 主要是文字 → 0 分
        else:
            rewards.append(0.5)   # 主要是短編碼 → 0.5 分
    print('soft text rewards',rewards)
    return rewards

def strict_format_reward_func(completions, answer, **kwargs) -> list[float]: #要可以分解成icd10編碼才給分
    """Reward function that checks if the completion has a specific format."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]

    rewards = []
    for r, a in zip(extracted_responses, answer):
        # 移除答案前綴
        if a.startswith("ICD10 編碼："):
            a = a.replace("ICD10 編碼：", "")
        # 文字敘述檢查 + 嚴格格式檢查
        if strict_format_check(r,a):  
            rewards.append(1)   # 是icd10格式 → 1 分
        else:
            rewards.append(0.0)   # 不符合格式 → 0 分
    print('strict text rewards',rewards)
    return rewards


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    rewards = [count_xml(c) for c in contents]
    print("xml rewards:", rewards)
    return rewards


# In[ ]:


# 選擇模型名稱
model_name = "models/Qwen3-4B-Instruct-2507"

# 設定輸出資料夾與 run_name，方便 tensorboard / log 區分不同模型
if "Llama" in model_name:
    output_dir = "outputs/Llama-1B-GRPO"
    run_name = "Llama-1B-GRPO-gsm8k"
else:
    output_dir = "outputs/Qwen-4B-instruct"
    run_name = "Qwen-4B-GRPO-icd"
    
# GRPO 訓練參數設定
training_args = GRPOConfig(
    output_dir=output_dir,         # 模型 checkpoint 與 log 的輸出位置
    run_name=run_name,             # run 名稱，用於 tensorboard 區分
    learning_rate=2e-6,            # 基本學習率
    adam_beta1=0.9,                # Adam 優化器參數 beta1
    adam_beta2=0.99,               # Adam 優化器參數 beta2
    weight_decay=0.1,              # 權重衰減，避免 overfitting
    warmup_ratio=0.1,              # 前 10% step 做 learning rate warmup
    lr_scheduler_type='cosine',    # 餘弦退火學習率
    logging_steps=1,               # 每 step log 一次
    bf16=True,                     # 使用 bfloat16 訓練（省顯存，保持精度）
    per_device_train_batch_size=1, # 每個 GPU batch size
    gradient_accumulation_steps=4, # 梯度累積，等效 batch size = 1*3=3
    num_generations=4,             # 每個 prompt 生成多少個答案，供 reward function 打分
    max_prompt_length=500,        # prompt 最長 token 長度
    max_completion_length=850,     # 模型生成部分最長 token 長度
    num_train_epochs=1,            # 訓練 epoch 數
    save_steps=200,                # 每 100 steps 存 checkpoint
    max_grad_norm=0.1,             # 梯度裁剪上限，避免梯度爆炸
    report_to="tensorboard",       # log 到 tensorboard
    log_on_each_node=False,        # 多機訓練時是否每個節點都 log
)

# LoRA 低秩適應設定，用於節省記憶體與加速訓練
peft_config = LoraConfig(
    r=16,                          # LoRA rank
    lora_alpha=64,                 # LoRA scaling 係數
    target_modules=[               # 指定要插入 LoRA 的線性層
        "q_proj", "k_proj", "v_proj", "ao_proj", 
        "up_proj", "down_proj", "gate_proj"
    ],
    task_type="CAUSAL_LM",         # 任務類型：因果語言模型
    lora_dropout=0.05,             # LoRA dropout，避免 overfitting
)

# 載入模型，啟用 Flash Attention 2
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,    # 模型權重型別：bfloat16
    attn_implementation="flash_attention_2", # 高效注意力實作
    device_map="auto",             # 自動分配多 GPU
    max_memory={                   # 限制每張卡的顯存用量
        0: "40GB",
        1: "40GB"
    },
    offload_folder="offload",      # 超出顯存的 tensor 存放到硬碟
    offload_state_dict=True,       # state_dict 是否 offload
    trust_remote_code=True         # 允許模型 repo 的自定義程式碼
)

# 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # pad token 設為 eos token

# 初始化 GRPOTrainer
# - reward_funcs: 自訂獎勵函數清單
# - train_dataset: 訓練資料
# - peft_config: LoRA 配置（若不用 LoRA 可移除）
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        accuracy_func,             # 最終準確率獎勵
        xmlcount_reward_func,      # 檢查輸出是否有 <reasoning>/<answer> tags
        soft_format_reward_func,   # 判斷文字敘述程度 如果不是文字敘述就加分
        strict_format_reward_func # 嚴格格式符合獎勵
        # length_cosine_reward_func # 控制輸出長度獎勵（避免太短或太長）
    ],
    args=training_args,
    train_dataset=dataset,
    # callbacks=[CustomSaveCallback()], # 若要自定義 checkpoint 存法可打開
    peft_config=peft_config
)

# 開始訓練
trainer.train()

