# LLM_terminal

Expects an environmental variable `REPLICATE_API_TOKEN`.

## How to run?

```bash
poetry install
poetry run chat.py --help
```

## Example:

````bash
> chat -f chat.py "Explain this file to me. Be succint."

This is a Python script that allows you to chat with a Large Language Model (LLM) from the terminal, with the option to include files in the conversation. It uses the `replicate` library to interact with the LLM and the `click` library for command-line arguments.```
````
