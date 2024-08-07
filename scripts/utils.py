"""Utility functions and prompt variables for create_db and query
"""

import yaml
from tqdm import tqdm


def load_config(yaml_path):
    """Load YAML file with parameter values.

    Args:
        file_path: Path to configuration file (string).
    """
    with open(yaml_path, encoding="utf-8") as file_obj:
        configuration = yaml.load(file_obj, Loader=yaml.SafeLoader)

    return configuration


def set_progress(pbar, update, description):
    """Update tqdm bar progress and set description

    Args:
        bar: tqdm progress bar (tqdm object).
        description: Description text (string).
        update: update increase (int).
    """
    pbar.update(update)
    pbar.set_description(description)


# prompt strings
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

SUMMARY_PROMPT_TEMPLATE = '''<<SYS>>Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary. Your answers should only write the summary once and not have any text after the answer is done.

Reference example:
Current summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

New lines of conversation:
Human: Why do you think artificial intelligence is a force for good?
AI: Because artificial intelligence will help humans reach their full potential.

New summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.

Do not repeat the above example. Only use it as reference for creating the new summary.<</SYS>>

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:'''

CONDENSE_QUESTION_PROMPT_TEMPLATE = '''<<SYS>>Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. Your answers should only write the summary once and not have any text after the answer is done.

Reference example #1:

Chat History:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

Follow up input:
Why is it a force for good?

Standalone question:
Why is artificial intelligence is a force for good?

Reference example #2:

Chat History:
The human greets AI. The AI asks human how is it going.

Follow up input:
Now explain me how USAID work?

Standalone question:
How does USAID work?

Do not repeat the above examples. Only use them as reference for creating the standalone question.<</SYS>>

Chat History:
{chat_history}

Follow Up Input:
{question}

Standalone question:'''

QA_PROMPT_TEMPLATE = '''[INST]<<SYS>>You are a helpful and respectful assistant. Answer the given question using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.

If you cannot find the answer in the context, say that question is out of context. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. <</SYS>>

CONTEXT:
{context}

Question:
{question}

Answer:[/INST]'''
