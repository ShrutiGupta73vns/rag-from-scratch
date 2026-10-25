"""Step 5: question + retrieved chunks -> answer from the local LLM."""
import ollama

from src import config

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "I don't know based on the documents."
Cite the source of each fact like [file, page N].

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(question: str, results: list[dict]) -> str:
    """Label each chunk with its source so the LLM can cite it."""
    context = "\n\n".join(f"[{r['source']}, page {r['page']}]\n{r['text']}" for r in results)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def ask_llm(prompt: str) -> str:
    response = ollama.chat(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": config.TEMPERATURE},
    )
    return response["message"]["content"]


def generate(question: str, results: list[dict]) -> str:
    return ask_llm(build_prompt(question, results))
