"""Prompt variables for create_db and query
"""

# prompt strings
SUMMARY_PROMPT_TEMPLATE = '''
"<|begin_of_text|><|start_header_id|>system<|end_header_id|>Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary. Your answers should only write the summary once and not have any text after the answer is done.

Reference example:
Current summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

New lines of conversation:
Human: Why do you think artificial intelligence is a force for good?
AI: Because artificial intelligence will help humans reach their full potential.

New summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.

Do not repeat the above example. Only use it as reference for creating the new summary.<|eot_id|><|start_header_id|>user<|end_header_id|>\
Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
'''


CONDENSE_QUESTION_PROMPT_TEMPLATE = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language. Your answers should only write the summary once and not have any text after the answer is done.

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

Do not repeat the above examples. Only use them as reference for creating the standalone question.<|eot_id|><|start_header_id|>user<|end_header_id|>

Chat History:
{chat_history}

Follow Up Input:
{question}

Standalone question:<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''


QA_PROMPT_TEMPLATE = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful and respectful assistant. Answer the given question using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.

If you cannot find the answer in the context, say that question is out of context. If a question does not make any sense, or is not factually coherent, explain why instead of #answering something not correct. If you don't know the answer to a question, please don't share false information.<|eot_id|><|start_header_id|>user<|end_header_id|>

CONTEXT:
{context}

Question:
{question}

Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
