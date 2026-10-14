import requests

from src.rag.retrieve import retrieve


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:latest"


def generate_answer(query, company="TCS", as_of=None, k=5):
    """
    Retrieve financial evidence and generate an answer using Ollama.
    """

    # Step 1: Retrieve relevant passages
    evidence = retrieve(
        query=query,
        company=company,
        as_of=as_of,
        k=100,
    )

    if not evidence:
        return {
            "answer": "I couldn't find relevant evidence to answer this question.",
            "citations": [],
        }

    # Step 2: Rerank passages using the cross-encoder
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    pairs = [
        (query, item.text)
        for item in evidence
    ]

    scores = reranker.predict(pairs)

    ranked_evidence = sorted(
        zip(scores, evidence),
        key=lambda item: float(item[0]),
        reverse=True,
    )

    selected = [
        item
        for _, item in ranked_evidence[:k]
    ]

    # Step 3: Build evidence context with citation metadata
    context_parts = []
    citations = []

    for item in selected:
        context_parts.append(
            f"Document: {item.doc_id}\n"
            f"Evidence: {item.text}"
        )

        citations.append({
            "doc_id": item.doc_id,
            "text": item.text,
        })

    context = "\n\n---\n\n".join(context_parts)

    # Step 4: Construct the LLM prompt
    prompt = f"""
You are VibeFinance, a financial document question-answering assistant.

Answer the user's question using ONLY the evidence provided below.

Rules:
1. Do not invent financial figures or facts.
2. If the evidence is insufficient, explicitly say so.
3. Preserve the units and reporting periods of financial figures.
4. Cite supporting evidence using its document ID.
5. Distinguish reported facts from your own calculations.

Evidence:
{context}

Question:
{query}

Provide a concise answer with citations.
"""

    # Step 5: Generate the answer using Ollama
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    result = response.json()

    return {
        "answer": result["response"].strip(),
        "citations": citations,
    }


if __name__ == "__main__":
    question = input("Ask VibeFinance a financial question: ")

    result = generate_answer(
        query=question,
        company="TCS",
        as_of="2026-10-15",
    )

    print("\nAnswer:")
    print(result["answer"])

    print("\nRetrieved citations:")

    for citation in result["citations"]:
        print(f"- {citation['doc_id']}")