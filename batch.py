import argparse
import os
import random

model_dir = "~/workspace/models/"
task_list = {"arc-c", "piqa", "hellaswag"}
model_list = {"mistralai/Mistral-7B-v0.1"}
peft_methods0 = {"dora"}
peft_methods1 = {"qlora", "loraplus"}
peft_methods2 = {"rslora", "mixlora"}
peft_methods3 = {"mixdora", "qmixlora"}

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


def call_gen(peft_method, tasks, name_prefix):
    config = (f"{config_command} --tasks {tasks} --adapter_name {name_prefix}"
              f" --file_name {name_prefix}.json {config_suffix.get(peft_method, '')}")
    os.system(config)
    print(config)


def call_run(peft_method, model, name_prefix):
    run = (f"{run_command} --base_model {model_dir}{model} --config {name_prefix}.json"
           f" --cuda_device {args.cuda} --log_file {name_prefix}.log"
           f" --overwrite false --attn_impl eager {run_suffix.get(peft_method, '')}")
    os.system(run)
    print(run)


def main(args):
    peft_methods = peft_methods0

    for model in model_list:
        for method in peft_methods:
            for task in task_list:
                name_prefix = f"{model.split('/')[-1]}_{method}_{task}"
                if args.run:
                    call_run(method, model, name_prefix)
                elif args.gen:
                    call_gen(method, task, name_prefix)
                else:
                    call_gen(method, task, name_prefix)
                    call_run(method, model, name_prefix)


    # model = "THUDM/chatglm3-6b"
    # method = "lora"
    # for task in task_list:
    #     sys_call(method, task, model)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch run tasks")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device number")
    parser.add_argument("--run", action="store_true", help="Run the tasks")
    parser.add_argument("--gen", action="store_true", help="Generate the config")
    parser.add_argument("--multi-task", action="store_true", help="Run multiple tasks at once")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
