import datasets
from datasets import load_dataset
import pandas as pd
from .conversation import get_conv_template
from functools import partial
import re

# tactic -> class_id 映射 
# Classes (by functionality):
# 0: simplification / simp-family
# 1: direct/equality/exact-style tactics
# 2: arithmetic / rewriting / normalization / algebraic simplifiers
# 3: introduction / pattern-matching / destructuring (cases, rcases, intros...)
# 4: application / proof-step control / forward reasoning (have, apply, refine...)
# 5: OTHER / miscellaneous

# Combined desired mapping — will be filtered by observed_tactics below
# tactic_to_class = {
#     # simplification family
#     "simp": 0, "simp_all": 0, "simpa": 0, "dsimp": 0, "field_simp": 0,

#     # exact / equality
#     "exact": 1, "exacts": 1, "rfl": 1, "symm": 1, "apply_fun": 1, "exact_mod_cast": 1,

#     # arithmetic / rewriting / normalization
#     "linarith": 2, "nlinarith": 2, "norm_num": 2, "norm_cast": 2, "omega": 2,
#     "rw": 2, "rwa": 2, "erw": 2, "linear_combination": 2, "trans": 2,
#     "ring": 2, "ring_nf": 2, "abel": 2, "gcongr": 2, "congr": 2, "convert": 2, "nth_rw": 2,

#     # intros / destructuring / case analysis
#     "intro": 3, "intros": 3, "rintro": 3, "rcases": 3, "cases": 3, "case": 3,
#     "constructor": 3, "obtain": 3, "induction": 3, "exists": 3, "split_ifs": 3,

#     # apply / refine / have / helpers
#     "have": 4, "haveI": 4, "refine": 4, "refine'": 4, "apply": 4, "use": 4, "specialize": 4,
#     "let": 4, "letI": 4, "assumption": 4, "trivial": 4, "contradiction": 4, "contrapose": 4, "contrapose!": 4,
#     "revert": 4, "clear": 4, "replace": 4, "change": 4, "choose": 4, "choose!": 4,

#     # misc / fallback
#     "all_goals": 5, "any_goals": 5, "tauto": 5, "infer_instance": 5, "pick_goal": 5, "group": 5,
#     "by_cases": 5, "decide": 5, "exfalso": 5, "absurd": 5, "continuity": 5, "positivity": 5,
#     "calc": 5, "show": 5, "suffices": 5, "split_ands": 5, "classical": 5, "OTHER": 5
# }

tactic_to_class = {

    # --------------------------------------------------
    # 0. Simplification (syntactic / rewriting-based)
    # --------------------------------------------------
    "simp": 0,
    "simp_all": 0,
    "simpa": 0,
    "dsimp": 0,
    "field_simp": 0,
    "norm_cast": 0,
    "unfold": 0,  # added: unfolding definitions

    # --------------------------------------------------
    # 1. Exact / Equality reasoning
    # --------------------------------------------------
    "exact": 1,
    "exacts": 1,
    "rfl": 1,
    "symm": 1,
    "exact_mod_cast": 1,
    "apply_fun": 1,
    "congr": 1,
    "gcongr": 1,
    "convert": 1,
    "trans": 1,
    "nth_rw": 1,
    "subst": 1,  # added: substitution

    # --------------------------------------------------
    # 2. Arithmetic / Algebraic normalization
    # --------------------------------------------------
    "linarith": 2,
    "nlinarith": 2,
    "norm_num": 2,
    "omega": 2,
    "ring": 2,
    "ring_nf": 2,
    "abel": 2,
    "linear_combination": 2,
    "rw": 2,
    "rwa": 2,
    "erw": 2,

    # --------------------------------------------------
    # 3. Structural introduction / case analysis
    # --------------------------------------------------
    "intro": 3,
    "intros": 3,
    "rintro": 3,
    "cases": 3,
    "cases'": 3,
    "case": 3,
    "rcases": 3,
    "constructor": 3,
    "induction": 3,
    "induction'": 3,
    "exists": 3,
    "split_ifs": 3,
    "ext": 3,
    "ext1": 3,       # added: variant of ext
    "funext": 3,     # added: function extensionality
    "obtain": 3,     # added: structural introduction / pattern match
    "fin_cases": 3,  # added: finite case analysis

    # --------------------------------------------------
    # 4. Proof construction / plumbing
    # --------------------------------------------------
    "apply": 4,
    "refine": 4,     # already in original, kept
    "refine'": 4,
    "have": 4,
    "haveI": 4,      # added: typeclass / context injection
    "use": 4,
    "specialize": 4,
    "assumption": 4,
    "trivial": 4,
    "revert": 4,
    "clear": 4,
    "replace": 4,
    "change": 4,
    "let": 4,
    "letI": 4,       # added: typeclass / context injection
    "choose": 4,
    "choose!": 4,


    # --------------------------------------------------
    # 5. Proof structuring / meta-level tactics
    # --------------------------------------------------
    "all_goals": 5,
    "any_goals": 5,
    "pick_goal": 5,
    "group": 5,
    "calc": 5,
    "show": 5,
    "suffices": 5,
    "split_ands": 5,
    "infer_instance": 5,
    "continuity": 5,
    "positivity": 5,

    # --------------------------------------------------
    # 6. Automated proof search / heavy automation
    # --------------------------------------------------
    "aesop": 6,
    "solve_by_elim": 6,
    "finish": 6,
    "first": 6,
    
    # --------------------------------------------------
    # 7. Logical / Classical reasoning / fallback
    # --------------------------------------------------
    "by_cases": 7,
    "tauto": 7,
    "decide": 7,
    "classical": 7,
    "contradiction": 7,
    "contrapose": 7,
    "contrapose!": 7,
    "exfalso": 7,
    "absurd": 7,
    "push_neg": 7,   # added: push negations
    "OTHER": 7,
}



def normalize_response(resp: str):
    if resp is None:
        return None

    # s = resp.strip().lower()
    s = resp.strip()

    # 去掉 `by`
    if s.startswith("by "):
        s = s[3:].strip()

    # 处理 `<;>` 前缀
    if s.startswith("<;> "):
        s = s[4:].strip()

    # 去掉 config 参数
    s = re.sub(r"\(config :=.*?\)", "", s)

    # 统一 simp/ exact
    s = re.sub(r"simp(\?| only)?(\s*\[.*?\])?", "simp", s)
    s = re.sub(r"exact(\?)?(\s*\[.*?\])?", "exact", s)

    return s.strip()


def extract_lean4_tactic_class(response: str, tactic_to_class=tactic_to_class):
    """
    输入 response 字符串，输出对应 tactic 类别 ID。
    先 normalize_response 统一化，再匹配已知 tactic，如果没有匹配到则归为 OTHER。
    """
    if not response:
        return tactic_to_class["OTHER"]

    s = normalize_response(response)

    # Find occurrences of any known tactic token in the normalized string.
    # We prefer the earliest occurrence in the string when multiple tactics appear.
    matches = []
    # sort keys by length descending to prefer longer tokens (e.g. 'simp_all' before 'simp')
    keys = sorted([k for k in tactic_to_class.keys() if k != "OTHER"], key=lambda x: -len(x))
    for k in keys:
        # match whole word or underscore-separated token
        pattern = r"\b" + re.escape(k) + r"(?=\s|\[|$)"
        m = re.search(pattern, s)
        if m:
            matches.append((m.start(), k))

    if matches:
        # pick the match with the smallest start position (earliest in string)
        chosen = min(matches, key=lambda x: x[0])[1]
        return tactic_to_class.get(chosen, tactic_to_class["OTHER"])
    else:
        print("No known tactic found in response:", response)
        # no known tactic found -> OTHER
        return tactic_to_class["OTHER"]

def add_tactic_label_2028(example):
    tactic = extract_lean4_tactic_class(example["response"], tactic_to_class)
    example["tactic"] = tactic
    return example

def get_dataset(dataset_name, local_data_dir=None):

    if dataset_name in ["gsm8k"]:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train", name="main")
    elif dataset_name in ["lighteval/MATH"]:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train", name="all")
    elif dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train_sft")
    elif dataset_name == "state_tactic_pairs":
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset("parquet", data_files={"train": dataset_name + ".parquet"}, split="train")
    elif dataset_name == "ag_news":
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name)
    else:
        dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
        dataset = load_dataset(dataset_name, split="train")

    return dataset

def process_sft_dataset(dataset_name, dataset, dataset_sample):
    # ===================== 新增：AG News =====================
    # if dataset_name == "ag_news":
    #     # AG News is a standard sequence classification dataset
    #     # No SFT-style processing is required
    #     print(f">> ===== Dataset {dataset_name} (no SFT processing), size={len(dataset)} =====")
    #     return dataset
    # ======================================================
    if dataset_name in ["lucasmccabe-lmi/CodeAlpaca-20k", "yahma/alpaca-cleaned", "FinGPT/fingpt-sentiment-train"]:
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output'], desc=f"Preprocessing {dataset_name} for unified format.")
    elif dataset_name in ["WizardLM/WizardLM_evol_instruct_70k"]:
        dataset = dataset.rename_column("output", "response")
    elif dataset_name in ["tatsu-lab/alpaca", "vicgalle/alpaca-gpt4", "gbharti/finance-alpaca"]:
        dataset = dataset.map(alpaca_format, remove_columns=['input', 'output', 'text'], desc=f"Preprocessing {dataset_name} for unified format.")
    elif dataset_name in ["TIGER-Lab/MathInstruct"]:
        df = pd.DataFrame(dataset)
        df = df.drop_duplicates(subset=['instruction'])
        dataset = datasets.Dataset.from_pandas(df)
        dataset = dataset.rename_column("output", "response")
        dataset = dataset.remove_columns(['source'])
    elif dataset_name in ["lighteval/MATH"]:
        dataset = dataset.rename_column("solution", "response")
        dataset = dataset.rename_column("problem", "instruction")
        dataset = dataset.remove_columns(['level', 'type'])
    elif dataset_name in ['gsm8k']:
        dataset = dataset.rename_column("question", "instruction")
        dataset = dataset.rename_column("answer", "response")
    elif dataset_name in ['medalpaca/medical_meadow_medical_flashcards']:       # TODO: 'lavita/ChatDoctor-HealthCareMagic-100k'. not sure whether to discard the instruction.
        dataset = dataset.remove_columns(['instruction'])
        dataset = dataset.rename_column("input", "instruction")
        dataset = dataset.rename_column("output", "response")
    elif dataset_name in ['state_tactic_pairs']:
        dataset = dataset.rename_column("input", "instruction")
        dataset = dataset.rename_column("tactic", "response")
        # ===== Extract Lean4 tactic labels =====
        dataset = dataset.map(
            add_tactic_label_2028,
            desc="Extracting Lean4 tactic labels"
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")
    dataset = dataset.shuffle(seed=2023)

    if dataset_sample:
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(num_sample))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    return dataset

def alpaca_format(example):
    if example['input'] == "":
        example["instruction"] = example["instruction"]
    else:
        example["instruction"] = example["instruction"] + " " + example['input']
    example["response"] = example['output']
    return example


def process_dpo_dataset(dataset_name, dataset, template_name, dataset_sample):
    if dataset_name in ["Anthropic/hh-rlhf"]:
        dataset = dataset.map(partial(split_hh, template_name=template_name), load_from_cache_file=False)
    elif dataset_name in ["HuggingFaceH4/ultrafeedback_binarized"]:
        dataset = dataset.map(partial(split_ultrafeedback, template_name=template_name), load_from_cache_file=False)
        dataset = dataset.remove_columns(['prompt_id', 'messages', 'score_chosen', 'score_rejected'])
    
    dataset = dataset.shuffle(seed=2023)
    if dataset_sample:
        num_sample = min(len(dataset), dataset_sample)
        dataset = dataset.select(range(num_sample))
    print(f">> ===== After processing, Dataset {dataset_name} has {len(dataset)} examples. =====")
    print(f">> ===== Data Example =====")
    print(dataset[0])
    print(f">> {'='*50}")
    return dataset
    
def find_common_prefix(str1, str2):
    prefix = ""
    for i in range(min(len(str1), len(str2))):
        if str1[i] == str2[i]:
            prefix += str1[i]
        else:
            break
    return prefix

def split_ultrafeedback(example, template_name="vicuna_v1.1"):
    conv_template = get_conv_template(template_name)

    conv_template.append_message(conv_template.roles[0], example["prompt"])
    conv_template.append_message(conv_template.roles[1], None)
    example["prompt"] = conv_template.get_prompt()
    example["chosen"] = " " + example["chosen"][1]["content"]       # There might need a space in the front.
    example["rejected"] = " " + example["rejected"][1]["content"]
    return example

def split_hh(example, template_name="vicuna_v1.1"):
    common_prefix = find_common_prefix(example["chosen"], example["rejected"])

    conv_template = get_conv_template(template_name)

    sentence = common_prefix
    human_prefix_len = len("\n\nHuman: ")
    assistant_prefix_len = len("\n\nAssistant: ")
    sentence = sentence[human_prefix_len:]
    turn = "user"
    while True:
        if turn == "user":
            index = sentence.find("\n\nAssistant: ")
            if index == -1:
                break
            else:
                conv_template.append_message(conv_template.roles[0], sentence[:index])
                turn = "assistant"
                sentence = sentence[index + assistant_prefix_len :]
        elif turn == "assistant":
            index = sentence.find("\n\nHuman: ")
            if index == -1:
                break
            else:
                conv_template.append_message(conv_template.roles[1], sentence[:index])
                turn = "user"
                sentence = sentence[index + human_prefix_len :]
    conv_template.append_message(conv_template.roles[1], None)
    example["prompt"] = conv_template.get_prompt()
    example["chosen"] = example["chosen"][len(common_prefix) - 1 :]     # -1 to include the space in the front.
    example["rejected"] = example["rejected"][len(common_prefix) - 1 :]
    return example