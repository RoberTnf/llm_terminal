"""
Chat with a LLM from the terminal with file support.

python chat.py --help
"""

import glob
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import click
import replicate
from datetime import datetime

FILENAME =  Path.home() / Path(f"shared/chat_llm/{datetime.now().isoformat()}.txt")



def create_commit_message():
    process = subprocess.run(["git diff --cached"], shell=True, stdout=subprocess.PIPE)
    if process.returncode != 0:
        raise ValueError("git diff failed")
    if len(process.stdout) == 0:
        raise ValueError("Empty diff")
    
    response = ask(prompt=f"Generate a detailed commit message, explaining the changes. At the beginning of the commit message write a short, 10 words max, summary of the changes as a commit title. \n{process.stdout}", history=[])

    subprocess.run(["git", "commit", "-e", "-m", f"{response}"], stdin=sys.stdin )

@click.command()
@click.option("-f", "--files", help="Regex for files", default=[], multiple=True)
@click.option("--commit", "-c", is_flag=True, help="Generate commit message")
@click.argument("text", nargs=-1)
def main(files: List[str], text: List[str], commit: bool) -> None:
    fnames = []
    history = []
    if commit:
        create_commit_message()
        return
    for file in files:
        fnames += glob.glob(file, recursive=True)
    for fname in fnames:
        if Path(fname).is_dir():
            continue
        with open(fname, "r", encoding="utf-8") as fp:
            msg = fp.read().replace("", "").replace("}", "\\}")
            history.append(
                {
                    "role": "user",
                    "msg": f"file_name: {fname}\nContent: ```{msg}```",
                }
            )
            with open(FILENAME, "a+") as fp2:
                fp2.write(f"f: {msg}")

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
    with open(Path(__file__).parent / Path("system_prompt.txt"), "r") as fp:
        system_prompt = fp.read()
    template = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}"
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

    with open(FILENAME, "a+") as fp:
        fp.write(f"\nQ> {prompt}")
        fp.write(f"\nA> {response}")

    return response


if __name__ == "__main__":
    main()

