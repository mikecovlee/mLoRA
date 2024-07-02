import argparse
import os

task_list = {"arc-c", "arc-e", "boolq", "glue:rte"}
# task_list = {"arc-c", "arc-e", "boolq", "glue:rte"}
model_list = {"meta-llama/Llama-2-7b-hf"}


def main(args):
    for model in model_list:
        if args.multi_task:
            tasks = ";".join(task_list)
            config = f"python launch.py gen --template mixlora --tasks \"{tasks}\""
            command = f"python launch.py run --base_model {model}"
            cuda_config = f"CUDA_VISIBLE_DEVICES={args.cuda} "
            command = cuda_config + command

            os.system(config)
            os.system(command)
        else:
            filename = f"{task}.json"
            for task in task_list:
                # Generate config
                config = f"python launch.py gen --template mixlora --tasks {task} --filename {filename}"

                # Generate command.
                command = ""
                if args.run:
                    # Run the model.
                    command = f"python launch.py run --base_model {model} --config {filename}"
                elif args.evaluate:
                    # Evaluate the model.
                    command = f"python launch.py evaluate --base_model {model}"
                cuda_config = f"CUDA_VISIBLE_DEVICES={args.cuda} "
                command = cuda_config + command

                # Run the command.
                os.system(config)
                os.system(command)


def parse_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Batch run tasks")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device number")
    parser.add_argument("--run", action="store_true", help="Run the tasks")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the tasks")
    parser.add_argument("--multi-task", action="store_true", help="Run multiple tasks at once")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
