import argparse
import os
import random

model_dir = "~/workspace/models/"
task_list = {"arc-c", "arc-e", "boolq", "obqa", "piqa", "siqa", "hellaswag", "winogrande"}
model_list = {"mistralai/Mistral-7B-v0.1"}
peft_methods0 = {"lora", "dora", "qlora", "loraplus"}
peft_methods1 = {"rslora", "mixlora", "mixdora", "qmixlora"}

config_command = f"python launch.py gen"
run_command = f"python launch.py run"

config_suffix = {
    "lora": "--template lora",
    "dora": "--template lora --use_dora",
    "qlora": "--template lora",
    "loraplus": "--template lora --loraplus_lr_ratio 20.0",
    "rslora": "--template lora --use_rslora",
    "mixlora": "--template mixlora",
    "mixdora": "--template mixlora --use_dora",
    "qmixlora": "--template mixlora",
}

run_suffix = {
    "qlora": "--quantize 4bit",
    "qmixlora": "--quantize 4bit",
}


def sys_call(method, tasks, model):
    name_prefix = f"{model.split('/')[-1]}_{method}_{tasks}"

    config = (f"{config_command} --tasks {tasks} --adapter_name {name_prefix}"
              f" --file_name {name_prefix}.json {config_suffix.get(method, '')}")
    run = (f"{run_command} --base_model {model_dir}{model} --config {name_prefix}.json"
           f" --cuda_device {args.cuda} --log_file {name_prefix}.log"
           f" --overwrite false --attn_impl eager {run_suffix.get(method, '')}")

    os.system(config)
    os.system(run)
    # print(config)
    # print(run)


def main(args):
    peft_methods = None
    if args.cuda == 0:
        peft_methods = peft_methods0
    elif args.cuda == 1:
        peft_methods = peft_methods1

    for model in model_list:
        for method in peft_methods:
            for task in task_list:
                sys_call(method, task, model)
    # model = "THUDM/chatglm3-6b"
    # method = "lora"
    # for task in task_list:
    #     sys_call(method, task, model)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch run tasks")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device number")
    parser.add_argument("--run", action="store_true", help="Run the tasks")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the tasks")
    parser.add_argument("--multi-task", action="store_true", help="Run multiple tasks at once")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
