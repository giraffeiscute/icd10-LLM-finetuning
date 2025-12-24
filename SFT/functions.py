import re
from datasets import load_dataset, Dataset
import numpy as np 
import os
import csv



def extract_icd_answer(text: str) -> str:
    """
    從模型的完整回覆或標準答案中提取 ICD-10 編碼。
    優先從 <answer> 標籤中提取,若無則從整個文本中提取。
    """
    # 使用正則表達式搜尋 <answer> 標籤，re.DOTALL 讓 . 可匹配換行符，re.IGNORECASE 忽略大小寫
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    # 如果找到 <answer> 標籤，則只在標籤內容中搜尋 ICD 編碼；否則在整個文本中搜尋
    search_text = answer_match.group(1) if answer_match else text
    
    # 匹配格式：字母 + 1-3位數字 + 可選的小數點和1-2位數字
    codes_list = re.findall(r'\b([A-Z]\d{1,6}(?:\.\d{1,2})?)\b', search_text, re.IGNORECASE)
    
    # 移除小數點、去重、排序、轉大寫，並過濾掉長度<=2的編碼
    cleaned_codes = sorted(set(
        code.replace('.', '').upper() 
        for code in codes_list 
        if len(code.replace('.', '')) > 2
    ))
    return " ".join(cleaned_codes)

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
     # 模型輸出文字
    # responses = [completion[0] for completion in completions]

    # # 取得問題（debug 用）
    # q = prompts[0]

    # 提取 XML 中的回覆
    extracted_responses = [extract_icd_answer(r) for r in responses]

    # --- CSV 紀錄邏輯 ---
    csv_filename = "model_generation_logs.csv"
    file_exists = os.path.isfile(csv_filename)
    
    # 我們只紀錄每一批次 (Batch) 的第一個範例來觀察，避免檔案過大
    # 或者你可以跑迴圈紀錄整個 Batch
    log_data = {
        "answer": answer[0],
        "response": responses[0],
        "extracted": extracted_responses[0]
    }

    with open(csv_filename, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=["answer", "response", "extracted"])
        # 如果檔案是新的，先寫入表頭 (Header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)
    # ------------------
    # Debug 輸出
    print('-'*20)
    print(f"Question:\n{q}")
    print(f"\nResponse:\n{responses[0]}")
    print(f"\nAnswer:\n{answer[0]}")
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
        r_codes = [code.strip()[:3] for code in r.split() if code.strip()] # 這裡從 split(',') 改成 split()

        a_set = set(a_codes)
        r_set = set(r_codes)

        score = compute_f1(r_set, a_set) * 30
        f1_scores.append(score)

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
    extracted_responses = [extract_icd_answer(r) for r in responses]

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


def length_cosine_reward_func(completions, answer, prompts, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_answers = [extract_icd_answer(r) for r in responses]

    rewards = []
    for i, r in enumerate(responses):  # 對於每一個模型輸出的回答 response（r）
        reasoning_text = r.split("<answer>")[0]  # 只取出 `<answer>` 之前的部分（即 reasoning 推理內容）
        
        # 使用 tokenizer 將 reasoning_text 分詞成 tokens，並取其長度作為 reasoning 的 token 長度
        reasoning_len = len(tokenizer(reasoning_text, return_tensors="pt")["input_ids"][0])
        
        is_correct = extracted_answers[i] == answer[i]  # 判斷模型的最終回答是否與正確答案相符（是否正確）

        # 設定最大長度為 16000，將推理長度 reasoning_len 映射到 [0, π] 的範圍（正規化）
        max_length = 786 
        x = np.clip(reasoning_len / max_length, 0, 1) * np.pi/2  # 確保比例不超過 1，避免過長失控

        if is_correct:
            # 如果答對了，就使用「遞減型餘弦」，鼓勵短的答案（reward 隨長度增加而減少）
            reward = 5 * np.cos(x)  # 最多給 +5，越長越接近 0
        else:
            # 如果答錯了，就使用「反向餘弦」，鼓勵多一點思考（reward 隨長度增加而上升）
            # reward = -5.0 * np.cos(x)  # 最多扣 -10，越長越接近 0（或甚至轉為正）
            reward = 0
        rewards.append(float(reward))  # 把這一筆的 reward 加入總 reward 清單
    return rewards  # 回傳所有 sample 的 reward 分數