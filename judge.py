import json
from ollama import chat


def judge_answer(question, answer, source_fragments, trace):
    judge_prompt = """
    You are evaluating the answer of a RAG system.

    Evaluate the answer ONLY using the provided source fragments.
    Do not use your general knowledge.

    Check whether:
    - the answer is correct according to the sources,
    - the answer is faithful to the sources,
    - the answer answers the user's question,
    - the sources are appropriate,
    - the system correctly refuses to answer when the sources do not contain enough information.

    Use scores from 0 to 3:
    0 - incorrect or unsupported
    1 - partially correct
    2 - mostly correct
    3 - correct and well supported

    Return only JSON with exactly these fields:
    {
        "correctness": 0,
        "faithfulness": 0,
        "source_quality": 0,
        "answer_completeness": 0,
        "hallucination_detected": false,
        "grounded_in_sources": false,
        "comment": ""
    }
    - Treat every factual claim that is not explicitly supported by the provided
    source fragments as unsupported, even if it is true in the real world.
    - Do not give credit for information based on general knowledge.
    - If the answer mentions laws, institutions, benefits, procedures, or facts
    that do not appear in the source fragments, mark this as a faithfulness issue.
    - hallucination_detected should be true when the answer contains factual
    claims not supported by the provided source fragments.
    """

    judge_input = {
        "question": question,
        "answer": answer,
        "source_fragments": source_fragments,
        "trace": trace
    }

    response = chat(
        model="qwen3.5:35b",
        messages=[
            {"role": "system", "content": judge_prompt},
            {
                "role": "user",
                "content": json.dumps(judge_input, ensure_ascii=False)
            }
        ],
        format="json",
        options={"num_ctx": 32768, "temperature": 0}
    )

    return json.loads(response.message.content)