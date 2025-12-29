import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
import numpy as np 
import csv

from functions import accuracy_func, soft_format_reward_func, strict_format_reward_func, extract_icd_answer

SYSTEM_PROMPT = """你是一名專門根據患者用藥與病史紀錄預測 ICD-10 編碼的醫學專家。  
請嚴格遵守以下作答格式：  

<reasoning>  
逐步推理：逐條說明病例中每個臨床資訊（症狀、診斷、檢查結果、既往病史、用藥等）可能對應的 ICD-10 編碼，並在最後總結出最合理的編碼集合。  
</reasoning>  
<answer>  
僅輸出最終的 ICD-10 編碼，使用英文逗號分隔，不要添加任何額外文字或解釋。  
</answer>  
"""
EXAMPLE_QUESTION = """
Sex: F
Service: MEDICINE
Allergies: Aspirin
Attending: ___
Chief Complaint: weakness, diarrhea
Major Surgical or Invasive Procedure: None

History of Present Illness:
Ms. ___ is a ___ year-old woman with PMH significant for chronic anemia, osteoporosis, hypertension, ataxia, and recent L5 fracture in the setting of recurrent falls who presents from home with fatigue and generalized weakness and diarrhea.  

The patient's recent history is notable for the follow:  
- On ___, she presented with 4 days of LBP s/p fall from standing at which time imaging revealed acute L5 fracture. She was evaluated by Spine team who recommended early mobilization, pain control, but no brace required. She was evaluated by ___, and discharged to ___.  
- She was discharged home with ___ on ___.  
- On ___, she again presented to ___ s/p fall from standing while trying to reach for a glass of water. She did have a occipital scalp hematoma, but imaging including ___, C-spine CT, and L hip X-ray were negative for acute process so patient was discharged home.  

She now represents with generalized fatigue and diarrhea. In the setting of opiates for her L5 fracture, the patient has had constipation (5 days with no BM) for which she took a "natural laxative" the evening prior to presentation. The patient had 2 bowel movements in the morning of presentation and one episode of incontinence with diarrhea while sleeping. In this setting, she felt very weak and called EMS and was brought to ___ ED.

What ICD-10 codes should be assigned? Please just tell me which codes in the end
"""
EXAMPLE_REASONING = """The patient has chronic anemia, documented and clinically relevant, which corresponds to D500.  
There is vitamin deficiency suggested, which fits E538.  
She also has visual impairment history, which maps to H548.  
Hypertension is a chronic condition, coded as I10.  
Diarrhea due to laxative use is coded as K521 (toxic gastroenteritis and colitis).  
Her recent fracture with underlying osteoporosis corresponds to M810.  
Ataxia is a chronic neurologic condition, coded as R270.  
There is adverse effect of drugs (laxatives/opioids), so T474X5A applies.  
Environmental factor of injury is Y92099 (unspecified place of occurrence).  
She also has aspirin allergy, captured by Z9181.  

Therefore, the most appropriate ICD-10 codes are:
"""

EXAMPLE_ANSWER = "D500 E538 H548 I10 K521 M810 R270 T474X5A Y92099 Z9181"


chat_style_data = []

# 設定你想讀取的最大行數（例如設定為 10 筆）
MAX_ROWS = 1000

with open("0.4_acc_result.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    
    # 使用 enumerate 來取得行數索引 (i)
    for i, row in enumerate(reader):
        # 如果達到設定的行數就停止迴圈
        if MAX_ROWS is not None and i >= MAX_ROWS:
            break
            
        # 原始的資料處理邏輯
        chat_style_data.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": EXAMPLE_QUESTION},
                {"role": "assistant", "content": f"<reasoning>{EXAMPLE_REASONING}</reasoning><answer>{EXAMPLE_ANSWER}</answer>"},
                {"role": "user", "content": row["question"]}
            ],
            "answer": row["ground_truths_ans"]
        })

print(f"已成功載入 {len(chat_style_data)} 筆資料。")

dataset = chat_style_data


# 選擇模型名稱
model_name = "models/Qwen3-14B-sft_v2"

# 設定輸出資料夾與 run_name，方便 tensorboard / log 區分不同模型
if "Llama" in model_name:
    output_dir = "outputs/Llama-1B-GRPO"
    run_name = "Llama-1B-GRPO-gsm8k"
else:
    output_dir = "outputs/Qwen-14B-sft_v2"
    run_name = "Qwen-14B-sft_v2"
    
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
    gradient_accumulation_steps=2, # 梯度累積，等效 batch size = 1*3=3
    num_generations=2,             # 每個 prompt 生成多少個答案，供 reward function 打分
    max_prompt_length=500,        # prompt 最長 token 長度
    max_completion_length=1600,     # 模型生成部分最長 token 長度
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

from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # 使用 NF4 分佈，對權重保留更好
    bnb_4bit_compute_dtype=torch.bfloat16, # 計算時轉回 bf16，保持精確度
    bnb_4bit_use_double_quant=True,  # 二次量化，再省約 0.4 bit/param
)

# 載入模型，啟用 Flash Attention 2
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,    # 注入量化設定
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
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
        strict_format_reward_func # 嚴格格式符合獎勵
    ],
    args=training_args,
    train_dataset=dataset,
    # callbacks=[CustomSaveCallback()], # 若要自定義 checkpoint 存法可打開
    peft_config=peft_config
)

# 開始訓練
trainer.train()


# 訓練完後合併並儲存 (注意：這通常需要較大的 CPU RAM)
from peft import PeftModel

# 1. 先把訓練好的模型存起來
trainer.save_model("./temp_lora_weights")

# 2. 載入原始模型 (不使用量化，以便合併)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cpu", # 合併建議在 CPU 或足夠顯存的 GPU
    trust_remote_code=True
)

# 3. 載入 LoRA 權重並合併
model_to_merge = PeftModel.from_pretrained(base_model, "./temp_lora_weights")
merged_model = model_to_merge.merge_and_unload()

# 4. 儲存完整模型
merged_model.save_pretrained("outputs/Qwen-14B-GRPO-sft")
tokenizer.save_pretrained("outputs/Qwen-14B-GRPO-sft")