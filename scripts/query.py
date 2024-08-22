"""LangChain multi-doc retriever with ChromaDB
 Local LLM
"""

import os
import sys
import textwrap
import yaml
import torch
from tqdm import tqdm
from langchain.vectorstores import Chroma
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains.conversation.memory import ConversationSummaryMemory
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, FalconForCausalLM
from transformers import TextIteratorStreamer, pipeline
from transformers import AutoModelForCausalLM


# add the directory for locally hosted models
# os.environ["HUGGINGFACE_HUB_CACHE"] = "/path/to/huggingface_cache"


def process_llm_response(llm_response, width, out_file):
    """Format LLM output and write to file

    Args:
        llm_response: LLM output containing response and sources.
        width: desired output text width.
        out_file: path to output file.
    """
    with open(out_file, "a", encoding="utf-8") as output:
        output.write("Question:\n")
        output.write(wrap_text_preserve_newlines(llm_response["question"], width))
        output.write("\nResponse:\n")
        output.write(wrap_text_preserve_newlines(llm_response["answer"], width))
        output.write("\n\nSources:\n\n")
        for source in llm_response["source_documents"]:
            if "page" in source.metadata:
                output.write(
                    f"Page number:{source.metadata['page']} Document:"
                    f"{source.metadata['source']} \n"
                )
            else:
                output.write(f"Document:{source.metadata['source']} \n")
        output.write("\n")
        output.write(width * "-")
        output.write("\n")


def _get_chat_history(chat_history) -> str:
    return chat_history


def wrap_text_preserve_newlines(text, width) -> str:
    """Format text to adjust width

    Args:
        text: text to format.
        width: width to adjust the text to.
    """
    # Split the input text into lines based on newline characters
    lines = text.split("\n")

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = "\n".join(wrapped_lines)

    return wrapped_text


def load_llm(config):
    """Load a pre-trained LLM model

    A pre-trained locally-stored model is loaded by providing:
        - A tokenizer, which prepares the inputs to be processed by the model.
        - A torch `dtype`, representing a `torch.Tensor` data type, that
        overrides the default, reducing the memory required to load the model by
        sacrificing precision.
        - A quantization argument (e.g., `load_in_4bit` or `load_in_8bit`).
        - A device map that specifies where each submodule should go (e.g., GPU
        devices).

    Args:
        config: Configuration data with parameter values.

    Raises:
        ValueError: If the model is not supported.
    """
    model_id = os.environ["HUGGINGFACE_HUB_CACHE"] + "/" + config["model_id"]

    if "llama2" in config["model_id"].lower():
        print(f'Using model {config["model_id"]}')
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            load_in_8bit=config["load_in_8bit"],
            device_map=config["lm_device_map"],
        )
    elif "falcon" in model_id.lower():
        print(f'Using model {config["model_id"]}')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = FalconForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            load_in_8bit=config["load_in_8bit"],
            device_map=config["lm_device_map"],
        )
    elif "llama3" in config["model_id"].lower():
        print(f'Using model {config["model_id"]}')
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            load_in_8bit=config["load_in_8bit"],
            device_map=config["lm_device_map"],
        )
    else:
        raise ValueError(
            f"Model in {config['model_id']} not supported, must be Llama-2"
            "Falcon, or Llama-3"
        )

    return model, tokenizer


def read_db(db, config):
    """Read a Chroma database and return a retriever

    A Chroma database is a collection of embeddings, which are high-dimensional
    vectors that capture the semantic meaning of text.
    The ChromaDB is read from memory (`persist_directory`), uses an embedding
    model, and other optional `model_kwargs` (e.g., {'device':'cuda'}).
    The function returns a vector retriever given a specific search algorithm
    `search_type` and optional arguments that can include the number of sources
    to return (`k`) and the amount of documents to pass to the search algorithm
    (`fetch_k`).

    Args:
        db: Chroma database
        config: Configuration data with parameter values.
    """
    # with HiddenPrints():
    instructor_embeddings = HuggingFaceInstructEmbeddings(
        model_name=config["model_name"], model_kwargs=config["model_kwargs"]
    )

    vectordb = Chroma(embedding_function=instructor_embeddings, persist_directory=db)
    retriever = vectordb.as_retriever(
        search_type=config["search_type"], search_kwargs=config["search_kwargs"]
    )
    return retriever


def create_memory(model, tokenizer, config):
    """Create memory to summarize conversations over time

    Args:
        model: LLM.
        tokenizer: Data structure containing tokenized text.
        config: Configuration data with parameter values.
    """
    if "llama3" in config["model_id"].lower():
        import utils_llama3 as utils
    else:
        import utils_llama2 as utils

    summary_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=config["pipeline_device_map"],
        max_new_tokens=config["max_new_tokens"],
        eos_token_id=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ],
    )
    summary_llm = HuggingFacePipeline(pipeline=summary_pipe)

    summary_prompt = PromptTemplate(
        template=utils.SUMMARY_PROMPT_TEMPLATE, input_variables=["summary", "new_lines"]
    )

    return ConversationSummaryMemory(
        llm=summary_llm,
        memory_key="chat_history",
        return_messages=True,
        prompt=summary_prompt,
    )


def qa_generator(model, tokenizer, config, retriever, ipynb=False):
    """Create question and answer chain with memory

    Args:
        model: LLM.
        tokenizer: Data structure containing tokenized text.
        config: Configuration data with parameter values.
        retriever: Returns documents from a database given a query.
        ipynb: Boolean to adjust return if functions is called from a jupyter
            notebook.
    """

    if "llama3" in config["model_id"].lower():
        import utils_llama3 as utils
    else:
        import utils_llama2 as utils

    condense_question_prompt = PromptTemplate(
        template=utils.CONDENSE_QUESTION_PROMPT_TEMPLATE,
        input_variables=["chat_history", "question"],
    )

    qa_prompt = PromptTemplate(
        template=utils.QA_PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    # Generating a standalone question
    question_generator_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=config["q_device_map"],
        eos_token_id=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ],
        max_length=config["q_max_length"],
    )
    question_generator_llm = HuggingFacePipeline(pipeline=question_generator_pipe)
    question_generator = LLMChain(
        llm=question_generator_llm, prompt=condense_question_prompt
    )

    # Generating the response
    streamer = TextIteratorStreamer(
        tokenizer,
        timeout=config["timeout"],
        skip_prompt=config["skip_prompt"],
        skip_special_tokens=config["skip_special_tokens"],
    )
    doc_chain_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=config["r_device_map"],
        eos_token_id=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ],
        max_length=config["r_max_length"],
        streamer=streamer,
    )
    doc_chain_llm = HuggingFacePipeline(pipeline=doc_chain_pipe)
    doc_chain = load_qa_chain(doc_chain_llm, chain_type="stuff", prompt=qa_prompt)

    # Chain for handling above define chains
    # if memory is passed no need to pass chat_history during inference
    qa_chain_with_mem = ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=doc_chain,
        get_chat_history=_get_chat_history,
        question_generator=question_generator,
        return_source_documents=config["return_source_documents"],
    )

    if ipynb:
        return qa_chain_with_mem, streamer

    return qa_chain_with_mem


def run_query(db, query_file, out_file, config_file):
    """Run retrieval-augmented generation queries using a LLM onto a Chroma
       database

    Args:
        db: Path to directory of the Chroma database.
        query_file: Path to the file containing the queries.
        out_file: Output file with answers to questions.
        config_file: Path to the file containing the parameter values.
    """

    prog = tqdm(range(100), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")

    set_progress(prog, 0, "Loading configuration data")
    config = load_config(config_file)

    set_progress(prog, 1, "Reading database")
    retriever = read_db(db, config)

    set_progress(prog, 9, "Set up LLM")
    model, tokenizer = load_llm(config)

    set_progress(prog, 30, "Create memory")
    memory = create_memory(model, tokenizer, config)

    set_progress(prog, 5, "Set up prompt")
    qa_chain_with_memory = qa_generator(model, tokenizer, config, retriever)

    set_progress(prog, 25, "Querying the database")
    with open(query_file, "r", encoding="utf-8") as file:
        questions = file.readlines()
        chunk = 30 / len(questions)

        for question in questions:
            generate_kwargs = {"question": question, "chat_history": memory.buffer}
            response = qa_chain_with_memory(generate_kwargs)
            process_llm_response(response, config["width"], out_file)
            memory.save_context({"input": question}, {"output": response["answer"]})
            set_progress(prog, chunk, "Querying the database")

    set_progress(prog, 0, "All queries processed")


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


if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 6:
        print("Error: wrong number of arguments")
        print("Usage: python query.py <db> <query_file> <output> [config file]")
        sys.exit(1)

    db_dir = sys.argv[1]
    query = sys.argv[2]
    output_file = sys.argv[3]
    CONF = "./scripts/config.yaml"

    if not os.path.exists(db_dir) or not os.path.isdir(db_dir):
        raise ValueError(
            f"Error: the db path provided '{db_dir}' doesn't exist"
            "and/or is not a directory."
        )

    if not os.path.exists(query):
        raise ValueError(f"Error: query file '{query}' does not exist.")

    if len(sys.argv) == 5:
        CONF = sys.argv[4]

    run_query(db_dir, query, output_file, CONF)
