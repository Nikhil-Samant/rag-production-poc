from llama_index.core import PromptTemplate
from langfuse import Langfuse
from config import PROMPT_TEMPLATE, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY


def get_template():
    template = default_template
    if LANGFUSE_PUBLIC_KEY is not None and LANGFUSE_SECRET_KEY is not None:
        try:
            langfuse = Langfuse()
            prompt = langfuse.get_prompt(PROMPT_TEMPLATE)
            if prompt is not None:
                template = prompt.compile()
        except Exception as e:
            print("Error in getting prompt template - ", e)
        return PromptTemplate(template)
    return PromptTemplate(template)


default_template = """
Your name is RAG bot. You are a specialist in answering questions related to computer science papers. You have a knowledge
base of various computer science papers.
Here is some context related to the query:
    -----------------------------------------
    {context_str}
    -----------------------------------------

Considering the above information, please respond to the following inquiry with only the given context. Don't use any
other knowledge in the beginning. Only If you don't find the answer in the context, then use your wide knowledge outside
of this context

-----------------------------------------------------
Question: {query_str}.
-----------------------------------------------------

Ensure your response is understandable to 8th grade student. Return the response in markdown format.
"""
