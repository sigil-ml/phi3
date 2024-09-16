"""
Triton compatible embedding model to load the all-mpnet-base-v2 model using SentenceTransformer
to ingest a list of documents & return their embeddings.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import triton_python_backend_utils as pb_utils  # pylint: disable=import-error
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class TritonPythonModel:
    """LLM for topic summarization"""

    outputs: Dict[str, str]
    model_name: str
    character_splitter: CharacterTextSplitter
    prompt_template: str
    refined_prompt_template: str
    refined_prompt: PromptTemplate
    prompt: PromptTemplate
    llm_cfg: dict
    device: str
    hf_model_id: str
    tokenizer: AutoTokenizer
    llm: AutoModelForCausalLM
    gpu_pipe: HuggingFacePipeline

    def initialize(self, args):
        """Load model into memory and set some necessary model specific variables"""

        model_config = json.loads(args["model_config"])

        self.outputs = {
            output["name"]: pb_utils.triton_string_to_numpy(output["data_type"])
            for output in model_config["output"]
        }
        self.model_name = args["model_name"]

        self.prompt_template = """Summarize this content:
        {text}
        SUMMARY: 
        """
        self.prompt = PromptTemplate.from_template(self.prompt_template)

        self.refined_prompt_template = (
            "Your job is to produce a final refined summary\n"
            "Here's the existing summary: {existing_answer}\n "
            "Now add to it based on the following context (only if needed):\n"
            "------------\n"
            "{text}\n"
            "------------\n"
            "SUMMARY: "
        )
        self.refined_prompt = PromptTemplate.from_template(self.refined_prompt_template)

        self.llm_cfg = {
            "max_new_tokens": 1500,
            "temperature": 0.7,
            "context_length": 6000,
        }

        self.model_id = "amazon/MistralLite"
        self.llm_path = Path("./model_data/mistrallite")

        try:
            self.char_splitter = CharacterTextSplitter(
                separator="\n", chunk_size=1200, chunk_overlap=200
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path)
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.llm_path, local_files_only=True
            )
        except ValueError:
            self.model = None
            self._log("Not instantiated due to missing file...")

        self.pipeline = pipeline(
            "text-generation",
            model=self.llm,
            tokenizer=self.tokenizer,
            max_new_tokens=1000,
            device=0,
        )
        self.gpu_pipe = HuggingFacePipeline(
            pipeline=self.pipeline,
            pipeline_kwargs=self.llm_cfg,
        )

        self.chain = load_summarize_chain(
            self.gpu_pipe,
            chain_type="refine",
            verbose=False,
            question_prompt=self.prompt,
            refine_prompt=self.refined_prompt,
            return_intermediate_steps=False,
            input_key="input_documents",
            output_key="output_text",
        )

    def execute(self, requests):
        """
        Decode & embed list of documents from a Triton request
        """

        responses = []
        for request in requests:
            documents = (
                pb_utils.get_input_tensor_by_name(request, "documents")
                .as_numpy()
                .tolist()
            )
            documents = [t.decode("UTF-8") for t in documents]
            documents = self.char_splitter.create_documents(documents)

            self._log("Summarizing documents")

            summarization_dict = self.chain.invoke({"input_documents": documents})
            summarization = summarization_dict["output_text"]
            summarization = np.array([summarization])

            self._log("... summarization complete!")

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "summarization",
                            summarization.astype(self.outputs["summarization"]),
                        )
                    ]
                )
            )
        return responses

    def finalize(self):
        """Clean up model on server shutdown"""
        print(f"Cleaning up {self.model_name}...", flush=True)

    def _log(self, message: str) -> None:
        """Log a message prepended with the current time"""
        cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        print(f"{cur_time} - {self.model_name} : {message}", flush=False)
