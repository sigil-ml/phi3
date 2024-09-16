import os
from pathlib import Path

# from ctransformers import AutoModelForCausalLM, AutoTokenizer
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.summarize import load_summarize_chain

# from langchain.llms import CTransformers, GPT4All
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.llms.ctransformers import CTransformers

# from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def get_summary(llm, list_of_texts):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1200, chunk_overlap=200
    )

    docs = text_splitter.create_documents(
        list_of_texts
    )  # stuffs the lists of text into "Document" objects for LangChain

    prompt_template = """Summarize this content:
        {text}
        SUMMARY: 
        """
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a final refined summary\n"
        "Here's the existing summary: {existing_answer}\n "
        "Now add to it based on the following context (only if needed):\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "SUMMARY: "
    )
    refine_prompt = PromptTemplate.from_template(refine_template)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    chain = load_summarize_chain(
        llm,
        chain_type="refine",
        verbose=False,
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
    )

    return chain.invoke({"input_documents": docs})


if __name__ == "__main__":
    # llm = Llama(
    #     model_path="./Phi-3-mini-4k-instruct-q4.gguf",  # path to GGUF file
    #     n_ctx=4_000,  # The max sequence length to use - note that longer sequence lengths require much more resources
    #     n_threads=8,  # The number of CPU threads to use, tailor to your system and the resulting performance
    #     n_gpu_layers=0,  # The number of layers to offload to GPU, if you have GPU acceleration available. Set to 0 if no GPU acceleration is available on your system.
    # )
    # llm = AutoModelForCausalLM.from_pretrained(
    #     model_path_or_repo_id="./mistrallite.Q4_K_M.gguf",
    #     model_type="mistral",
    # )
    # llm = CTransformers(
    #     model="mistrallite.Q4_K_M.gguf",
    #     model_type="mistral",
    #     gpu_layers=50,
    #     model_kwargs={"device_map": "cuda:0"},
    # )
    # hf = HuggingFacePipeline.from_model_id(
    #     model_id="gpt2",
    #     task="text-generation",
    #     pipeline_kwargs={"max_new_tokens": 10},
    # )
    model_id = "amazon/MistralLite"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model_save_path = Path("./model_data/mistrallite")
    model.save_pretrained(str(model_save_path))
    tokenizer.save_pretrained(str(model_save_path))
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10, device=0
    )
    hf = HuggingFacePipeline(pipeline=pipe)

    # tokenizer = AutoTokenizer.from_pretrained("./mistrallite.Q4_K_M.gguf")
    config = {
        "max_new_tokens": 1500,
        "temperature": 0.7,
        "context_length": 6000,
    }
    # llm = CTransformers(
    #     model="mistrallite.Q4_K_M.gguf",
    #     model_type="mistral",
    #     config=config,
    #     threads=os.cpu_count(),
    #     n_gpu_layers=50,
    #     model_kwargs={"device": "cuda"},
    # )

    # llm = Ollama(model="mistrallite")

    doc1 = """
    Yes, I think we have enough analysis here that we can reasonably make this a leads question and answer it. I think the resolution we've converged on is:

    Self comes into scope at the { of an interface, impl, class, or mixin definition, and shadows any outer Self. (We could either say that Self is a member name, or that it's added to the scope directly, depending on whether we want T.Self to work as a way to name T.)
    .Self comes into scope at an as (in an impl or in an as expression), :!, or where, and names "the thing on the left" -- in impl T as X, T as X, and T:! X, .Self in X refers to T, and in A where B, the type T:! A constrained by the where clause is .Self in the constraints in B.
    Previously we've said that if there's more than one .Self in scope, they must all agree, in some loose sense (they can have different facet types so long as they have the same value). I'm not sure whether we still want that, or how we'd implement it if so -- maybe a shadowing rule for .Self would be more consistent.
        
    """

    data = [
        "This is a story about a dog that chased a mailman down the street.",
        "This is a story about a mailman that got chased by a dog while he was delivering mail.",
        "This is a story about a neighbor who watched a mailman get chased down the street by his neighbors dog!",
    ]

    # template = """Question: {question}

    # Answer: Let's think step by step."""
    # prompt = PromptTemplate.from_template(template)

    # chain = prompt | hf

    # question = "What is electroencephalography?"

    # print(chain.invoke({"question": question}))

    # pipe = pipeline("text-generation", model=llm)
    # print(llm.invoke("AI is going to", max_new_tokens=256))

    print(get_summary(hf, data))

    # prompt = "How to explain Internet to a medieval knight?"

    # # Simple inference example
    # prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"

    # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    # print(pipe(prompt, max_new_tokens=256))
