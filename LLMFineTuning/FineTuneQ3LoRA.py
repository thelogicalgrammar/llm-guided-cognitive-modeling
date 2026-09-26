import torch
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model
from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe
import pandas as pd
import os
from transformers import TrainerCallback
from datasets import Dataset, concatenate_datasets
from collections import defaultdict, Counter
import re

# ------- Create dataset from dataframe, filled based on text with features -------
def get_dataset(df):

    # extract Features from the data and specifically text for each participant
    features = []
    for _, row in df.iterrows():
        if not isinstance(row['text'], str):
            raise SystemExit(f"participant {row['participant']}: the text column holds "
                             f"{type(row['text']).__name__}, not a transcript")
        text = row['text'].strip()

        # extract all answers in <<...>> for this participant
        choices = re.findall(r'<<(.*?)>>', text)

        # extract the rest of the sentence after 'labeled'
        list_segment = re.search(r"labeled\s+([^.]+)", text)
        if list_segment is None:
            raise SystemExit(f"participant {row['participant']}: no 'labeled ...' phrase, so the "
                             f"options cannot be read. First 200 characters:\n{text[:200]}")
        list_segment = list_segment.group(1)

        # get all capital letters
        options = re.findall(r"\b[A-Z]\b", list_segment)
        if not options:
            raise SystemExit(f"participant {row['participant']}: no option letters in {list_segment!r}")

        # define mapping of character to number (e.g., 'E':0, 'M':1)
        choice_arm_mapping = {opt: i for i, opt in enumerate(options)}

        unknown = set(choices) - set(choice_arm_mapping)
        if unknown:
            raise SystemExit(f"participant {row['participant']}: chose {sorted(unknown)}, which are "
                             f"not among the options {options}")

        # tokenize the letters. '<<' is one token and the letter the next, so the letter carries a
        # distinct id from the same letter in running text; if that ever stops holding, the mask
        # below would silently mark the wrong positions.
        option_ids = []
        for opt in options:
            pieces = tokenizer(f"<<{opt}", add_special_tokens=False).input_ids
            if len(pieces) < 2:
                raise SystemExit(f"'<<{opt}' tokenises to {len(pieces)} token(s), so the option id "
                                 f"cannot be read off position 1")
            option_ids.append(pieces[1])
        if len(set(option_ids)) != len(options):
            raise SystemExit(f"options {options} do not have distinct token ids ({option_ids}), so "
                             f"choices could not be told apart")

        # define mapping of token_id to number
        option_token_mapping = {opt: i for i, opt in enumerate(option_ids)}

        # transform choices into 0 and 1
        bin_choices = [choice_arm_mapping[c] for c in choices]

        # tokenize the whole text
        tokenized = tokenizer(text, add_special_tokens=False, truncation=False)

        #  get the input_ids and attention mask
        input_ids = tokenized["input_ids"]

        # create default labels, -100 is don't compute nll over it
        completion_mask = [0] * len(input_ids)

        # set the token id in the labels where there are human choices
        for pos, token in enumerate(input_ids):
            if token in option_ids:
                completion_mask[pos] = 1

        # Build labels here rather than leaving it to trl. trl only folds a completion_mask into the
        # labels when completion_only_loss is true, and that resolves to false for a dataset without
        # prompt/completion columns, which is ours: the mask would be dropped and the loss taken over
        # every token of the transcript instead of the choices. A dataset that supplies labels is
        # used as given, which is what we want and does not depend on that default.
        labels = [token if keep else -100 for token, keep in zip(input_ids, completion_mask)]

        # create the features
        features.append({
            # model data
            "participant": row['participant'],
            "input_ids": input_ids,
            "labels": labels,

            # metadata
            "option_ids": option_ids,
            "option_mapping": option_token_mapping,
            "human_choices": bin_choices
        })

    # transform features to a dataframe
    features = pd.DataFrame(features)

    # convert to HF Dataset
    dataset = Dataset.from_pandas(features[["input_ids", "labels"]])

    return dataset

# ---------- DEFINE CONFIGURATIONS ----------
model_path = os.environ.get("COGMOD_BASE_MODEL", "/path/to/your/Models/Qwen3-Coder-Next")
data_path = os.environ.get("COGMOD_DATA_PATH", "/path/to/your/Data/").rstrip("/") + "/"
# the same default as run_FineTuneQ3LoRAjob.sh, which used to differ: the job created
# $PROJECT/FineTuneNew while this wrote into the data directory, so the results of a run were not
# where anything looked for them
output_dir = (os.environ.get("COGMOD_FINETUNE_OUT")
              or f"{os.environ.get('PROJECT', data_path.rstrip('/'))}/FineTuneNew").rstrip("/") + "/"
max_seq_length = 17555 # Adjust based on your needs
seed=3407

# A short run to check that the expert adapters train at all: peft can attach adapters to the
# fused MoE parameters through target_parameters and never update them, which leaves lora_B at
# its zero initialisation and produces a LoRA that does nothing to the experts. That is what
# happened to the published adapter, and it is invisible until the weights are inspected.
smoke_steps = int(os.environ.get("COGMOD_SMOKE_STEPS", 0))

# COGMOD_CONFIG_CHECK runs everything that does not need a GPU and then stops. A login node has no
# GPU, so bf16 has to give way there; every other argument is validated exactly as the real run has
# it. See the check itself, below the data.
config_check = bool(os.environ.get("COGMOD_CONFIG_CHECK"))

class ExpertAdapterCheck(TrainerCallback):
    """After a few optimizer steps, report whether the expert adapters have moved off zero"""
    def __init__(self, check_at=5):
        self.check_at = check_at

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step != self.check_at or model is None:
            return
        expert = [(name, parameter.detach().float().norm().item())
                  for name, parameter in model.named_parameters()
                  if "experts" in name and "lora_B" in name]
        other = [(name, parameter.detach().float().norm().item())
                 for name, parameter in model.named_parameters()
                 if "experts" not in name and "lora_B" in name]
        moved = sum(1 for _, norm in expert if norm > 0)
        print(f"\n[expert adapter check, step {state.global_step}] "
              f"{moved} of {len(expert)} expert lora_B tensors are non-zero; "
              f"{sum(1 for _, n in other if n > 0)} of {len(other)} elsewhere", flush=True)
        if expert and moved == 0:
            print("[expert adapter check] the expert adapters are not being trained: peft is not "
                  "propagating gradients to the parameters targeted through target_parameters. "
                  "Fitting will continue, but the experts will not be fine-tuned.", flush=True)

# ---------- DEFINE SFT, LORA CONFIG ----------
sft_config = SFTConfig(
    # ---- Data Preprocessing
    packing = True,
    padding_free = True,
    dataset_text_field=None,
    max_length = max_seq_length,
    shuffle_dataset=True,
    seed=seed,
    bf16=not config_check,            # a login node has no GPU to run bf16 on
    use_cpu=config_check,
    # ---- Regiment
    per_device_train_batch_size = 1,  # one batch at a time to avoid OOM
    per_device_eval_batch_size = 1,   # same for evaluation
    gradient_accumulation_steps = 8, # set to 8 after debugging
    gradient_checkpointing=True,
    warmup_steps = 2,              # each step is 8 forward & backward, and 1 optimizer call
    num_train_epochs = 3,
    # ----- Kernel
    prediction_loss_only = True,
    # trl defaults loss_type to "chunked_nll", which patches model.forward assuming it is a bound
    # method. apply_liger_kernel_to_qwen3_moe below has already replaced it with a partial for the
    # fused cross-entropy, so the patch raises AttributeError on __func__. trl's own documentation
    # says chunked_nll is incompatible with liger and defaults to "nll" when it applies liger
    # itself; we apply liger directly, so say so here. The fused kernel does this work anyway.
    loss_type = "nll",
    # ----- Optimizer
    # the paged optimiser needs CUDA, so the GPU-free check uses a plain one and reports whether
    # bitsandbytes (which this one needs) imports at all
    optim = "adamw_torch" if config_check else "paged_adamw_8bit",
    learning_rate = 2e-4,
    # --- Logging
    logging_strategy="steps",
    logging_steps=1,
    report_to="none",
    # counting tokens seen is only a logging convenience, and the comment left here said it errored;
    # with packing and padding_free there is no attention mask for it to count, so leave it off
    include_num_input_tokens_seen=False,
    # --- Evaluation ("no" for a smoke run: see the comment on max_steps below)
    eval_strategy="no" if smoke_steps else "steps",
    eval_steps=10,                 # evaluate every 100 steps
    # ---- Saving
    save_strategy="no" if smoke_steps else "steps",
    save_steps=10,
    save_total_limit=8,
    output_dir=output_dir,
    # a smoke run exists to see gradients reach the expert adapters, so it stops after a few steps
    # and skips the periodic evaluations and checkpoints, which would cost more than the steps do
    **({"max_steps": smoke_steps} if smoke_steps else {}),
)

# ---------- LOAD TOKENIZER ----------
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True
)

# ---------- LOAD DATA ----------
# Deliberately before the model: loading 80B parameters takes about ten minutes, and a problem in
# the data used to surface only after paying for it.
experiments = [
    {'name': 'TB', 'experiment': 'exp1', 'split': 'Train'},
    {'name': 'TB', 'experiment': 'exp2', 'split': 'Train'},
    {'name': 'HSo', 'experiment': 'exp0', 'split': 'Train'},
    {'name': 'HW', 'experiment': 'exp0', 'split': 'Train'},
    {'name': 'DB', 'experiment': 'exp0', 'split': 'Train'},
]

train_dats = []
eval_dats = []
for exp in experiments:

    train = pd.read_csv(f"{data_path}{exp['name']}/Train_text_{exp['experiment']}.csv")
    eval = pd.read_csv(f"{data_path}{exp['name']}/Val_text_{exp['experiment']}.csv")

    # an empty text export is the failure this run cannot survive, and it says nothing by itself
    for split, frame in (("Train", train), ("Val", eval)):
        if len(frame) == 0:
            raise SystemExit(f"{data_path}{exp['name']}/{split}_text_{exp['experiment']}.csv has no "
                             f"rows. Re-run loadAndSplitData.py; it reports empty text exports.")

    # get the dataset for this experiment
    train_dats.append(get_dataset(train))
    eval_dats.append(get_dataset(eval))

train_data = concatenate_datasets(train_dats)
eval_data = concatenate_datasets(eval_dats)

print(f"train sequences: {len(train_data)}   eval sequences: {len(eval_data)}", flush=True)
marked = sum(sum(1 for label in row if label != -100) for row in train_data["labels"])
tokens = sum(len(row) for row in train_data["input_ids"])
print(f"tokens: {tokens}, of which {marked} ({marked / tokens:.1%}) are human choices the loss is "
      f"computed on", flush=True)
if marked == 0:
    raise SystemExit("no tokens are marked as choices: the option letters were not found in the "
                     "tokenised text, so training would have nothing to learn from.")
if marked == tokens:
    raise SystemExit("every token would contribute to the loss, so the choice mask was lost")

# The configuration, the tokeniser and the data are all checked above without a GPU, so a mistake in
# any of them is found on a login node in a minute rather than after loading the model:
#   COGMOD_CONFIG_CHECK=1 COGMOD_SMOKE_STEPS=8 python LLMFineTuning/FineTuneQ3LoRA.py
if config_check:
    print(f"\nconfig and data OK. max_steps={getattr(sft_config, 'max_steps', None)}, "
          f"eval_strategy={sft_config.eval_strategy}, save_strategy={sft_config.save_strategy} "
          f"(bf16 was disabled for this check only)")
    for package in ("bitsandbytes", "liger_kernel", "peft", "accelerate"):
        try:                                              # not every package carries a __version__
            print(f"  {package}: {getattr(__import__(package), '__version__', 'importable')}")
        except ImportError as problem:                    # a dependency the run needs is missing
            print(f"  {package}: NOT IMPORTABLE ({problem})")
    raise SystemExit(0)

# ---------- LOAD MODEL & APPLY CONFIG ----------
apply_liger_kernel_to_qwen3_moe(
    rope=True,
    swiglu=True,
    cross_entropy=False,
    fused_linear_cross_entropy=True,
    rms_norm=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    local_files_only=True,
    device_map="auto",
    low_cpu_mem_usage=True,
    attn_implementation="kernels-community/flash-attn3",    # Use flash-attention for speed and avoiding cross-attention
)

# ----- BASELINE EVALUATION PERFORMANCE -----
# the loss of the unadapted model, for comparison with the fine-tuned one. This is a full pass over
# the validation set, so a smoke run skips it.
os.makedirs(output_dir, exist_ok=True)
if not smoke_steps:
    baseline_trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=sft_config
    )

    baseline_metrics = baseline_trainer.evaluate()
    pd.DataFrame([baseline_metrics]).to_csv(f"{output_dir}baseline_metrics.csv", index=False)

# ------- DEFINE LORA -------
# load config
config = AutoConfig.from_pretrained(model_path)

# define standard rank, and compute appropriate rank for each expert to avoid LoRA bloat
base_r = 16
num_experts = getattr(config, "num_local_experts", None) or config.num_experts
effective_r = max(1, base_r // num_experts)
print("effective_r:",effective_r)
lora_config = LoraConfig(
    r=base_r,
    lora_alpha=32,
    target_modules=[
        # There are 48 Layers in Qwen3-Coder-Next 80B
        # --- Classical Self-Attention, only appear 12 times ---
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # --- Linear Attention/DeltaNet Modules, appears 32 times ---
        "in_proj_qkvz",                   # Catches the fused QKVP linear attention
        "in_proj_ba",                     # Catches the secondary linear attention projection
        "out_proj",                       # Catches the attention output projections
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    target_parameters=[
        "gate_up_proj",
        "down_proj",
    ],
    rank_pattern={ # Chokes rank to 1 safely
        r".*mlp\.experts\.down_proj": effective_r,
        r".*mlp\.experts\.gate_up_proj": effective_r,
    },
    alpha_pattern={ # Ensure alpha is twice the size of effective_r
        r".*mlp\.experts\.down_proj": effective_r * 2,
        r".*mlp\.experts\.gate_up_proj": effective_r * 2,
    },
    bias="none",
    task_type="CAUSAL_LM"
)

# merge model
model = get_peft_model(model, lora_config)

# ------ CHECK IF TRAINABLE PARAMETERS IS PROPERLY SET (SHOULD BE 1% -  LOW NUMBER)
model.print_trainable_parameters()

# count the adapter tensors on the experts before training, so a silent failure is visible later
expert_adapters = [n for n, _ in model.named_parameters() if "experts" in n and "lora_" in n]
print(f"adapter tensors on the experts: {len(expert_adapters)}", flush=True)

# ---------- CREATE SFTTRAINER and TRAIN ----------
trainer = SFTTrainer(
    model = model,
    train_dataset = train_data,
    eval_dataset = eval_data,
    args = sft_config,
    callbacks = [ExpertAdapterCheck(check_at=min(5, smoke_steps) if smoke_steps else 5)],
)

# start training process, resuming only if a checkpoint was asked for and exists
resume = os.environ.get("COGMOD_RESUME_FROM")
trainer.train(resume_from_checkpoint=resume if resume and os.path.isdir(resume) else None)

# Run final evaluation on last LoRA state and store it. The adapter is saved either way, so a smoke
# run leaves something that checkMerge.py can inspect.
if not smoke_steps:
    trainer.evaluate()
model.save_pretrained(f"{output_dir}{'Smoke_LoRA' if smoke_steps else 'Final_LoRA'}")

# store log history
pd.DataFrame(trainer.state.log_history).to_csv(f"{output_dir}SFTTrainer_logs.csv", index=False)

# and report the expert adapters one last time: zero norms mean the experts were never adapted
final = [(n, p.detach().float().norm().item()) for n, p in model.named_parameters()
         if "experts" in n and "lora_B" in n]
print(f"after training: {sum(1 for _, v in final if v > 0)} of {len(final)} expert lora_B tensors "
      f"are non-zero", flush=True)