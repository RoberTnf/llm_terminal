"""
Chat with a LLM from the terminal with file support.

python chat.py --help
"""

import glob
from typing import Dict, List

import click
import replicate


@click.command()
@click.option("-f", "--files", help="Regex for files", default=[], multiple=True)
@click.argument("text", nargs=-1)
def main(files: List[str], text: List[str]) -> None:
    fnames = []
    history = []
    for file in files:
        fnames += glob.glob(file)
    for fname in fnames:
        with open(fname, "r", encoding="utf-8") as fp:
            history.append(
                {
                    "role": "user",
                    "msg": f"file_name: {fname}\nContent: ```{fp.read().replace("", "").replace("}", "\\}")}```",
                }
            )

    if len(text) > 0:
        prompt = " ".join(text)
    else:
        prompt = input("> ")

    response = ask(prompt, history)
    while True:
        history.append({"role": "user", "msg": prompt})
        history.append({"role": "assistant", "msg": response})
        prompt = input("\n\nQ> ")
        response = ask(prompt, history)


def build_template_from_history(history: List[Dict[str, str]], prompt) -> str:
    """
    Builds template in llama format with files and history
    """
    template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant. The user is interested in linux, archlinux, python or other programming languages. Try to give short answers, unless the questions asks for details."
    for h in history:
        role, msg = h["role"], h["msg"]
        template += f"<|eot_id|><|start_header_id|>{role}<|end_header_id|>\n\n{msg}"
    template += f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return template


def ask(prompt: str, history: List[Dict[str, str]]) -> str:
    """
    Ask and stream the response. Returns the response.
    """
    template = build_template_from_history(history, prompt)
    response = ""
    print("")
    for event in replicate.stream(
        "meta/meta-llama-3-70b-instruct",
        input={
            "top_k": 50,
            "top_p": 0.9,
            "prompt": template,
            "max_tokens": 2048,
            "min_tokens": 0,
            "temperature": 0.6,
            "prompt_template": "{prompt}",
            "presence_penalty": 1.15,
            "frequency_penalty": 0.2,
        },
    ):
        print(str(event), end="")
        response += str(event)

    print()
    return response


if __name__ == "__main__":
    main()

