import json
import weaviate
from sentence_transformers import SentenceTransformer
from ollama import chat

from judge import judge_answer
from tools import RAGTools

def main():
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051
    )
    try:
        collection = client.collections.get("StudyChunks")
        tools = RAGTools(collection, model)

        available_tools = {
            "search_vector": tools.search_vector,
            "get_chunk": tools.get_chunk,
            "get_document": tools.get_document,
            "get_neighboring_chunks": tools.get_neighboring_chunks,
            "search_by_metadata": tools.search_by_metadata
        }

        system_prompt = """
        Instructions:
        - Be helpful and answer questions concisely.
        - Use the available tools to retrieve relevant information from the documents before answering.
        - Base your answer only on information retrieved using the tools.
        - You may use multiple tools if more context is needed.
        - Cite each claim with the paragraph or article of the fragment it comes from.
        - Do not cite article or paragraph numbers, or give details, that are not in the retrieved fragments.
        - For broad questions, run several searches (e.g. one per type of benefit).
        - Polish student slang may appear in questions. For example, "kolos" means "kolokwium/test", and "oblać" may mean "fail".
        - If the retrieved documents do not contain enough information, say:
        "W dokumentach nie ma wystarczających informacji".
        """

        while True:
            query = input("Zadaj pytanie lub wyjdź (exit): \n")

            if query.lower() == "exit":
                break

            trace = {
                "question": query,
                "strategy": None,
                "tool_calls": [],
                "retriever_queries": [],
                "retrieved_chunk_ids": [],
                "sources": [],
                "grounded_in_sources": False
            }

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
            source_fragments = []
            step = 0
            while True:
                step += 1
                use_tools = step <= 5
                if not use_tools:
                    messages.append({"role": "user", "content": "Odpowiedz teraz na podstawie zebranych fragmentów."})
                response = chat(
                    model="qwen3.5:35b",
                    messages = messages,
                    tools = list(available_tools.values()) if use_tools else None,
                    options = {"num_ctx": 32768}
                )

                messages.append(response.message)

                if not response.message.tool_calls:
                    answer = response.message.content

                    used_tools = [call["tool"] for call in trace["tool_calls"]]
                    trace["strategy"] = " -> ".join(used_tools)

                    trace["retrieved_chunk_ids"] = list(
                        dict.fromkeys(trace["retrieved_chunk_ids"])
                    )

                    trace["sources"] = list(
                        dict.fromkeys(trace["sources"])
                    )

                    unique_fragments = {}

                    for chunk in source_fragments:
                        unique_fragments[chunk["chunk_id"]] = chunk

                    source_fragments = list(unique_fragments.values())[-15:]

                    judge_result = judge_answer(
                        question=query,
                        answer=answer,
                        source_fragments=source_fragments,
                        trace=trace
                    )

                    trace["grounded_in_sources"] = judge_result["grounded_in_sources"]

                    print(answer)
                    print("\nTRACE:")
                    print(json.dumps(trace, indent=2, ensure_ascii=False))

                    print("\nJUDGE:")
                    print(json.dumps(judge_result, indent=2, ensure_ascii=False))

                    break

                for tool_call in response.message.tool_calls:
                    function_name = tool_call.function.name
                    arguments = tool_call.function.arguments
                    trace["tool_calls"].append({
                        "tool": function_name,
                        "arguments": arguments
                    })
                    if function_name == "search_vector":
                        trace["retriever_queries"].append(arguments["query"])

                    function = available_tools.get(function_name)

                    if function is None:
                        result = f"Unknown tool: {function_name}"
                    else:
                        try:
                            result = function(**arguments)
                        except Exception as e:
                            result = f"Tool error: {e}"
                        if isinstance(result, dict):
                            result = [result]

                    if isinstance(result, list) and function_name in [
                        "search_vector",
                        "search_by_metadata",
                        "get_neighboring_chunks",
                        "get_chunk"
                    ]:
                        source_fragments.extend(result)
                        for chunk in result:
                            trace["retrieved_chunk_ids"].append(chunk["chunk_id"])
                            trace["sources"].append(f'{chunk["source"]} {chunk.get("article") or chunk.get("paragraph") or ""}')

                    messages.append({
                        "role": "tool",
                        "tool_name": function_name,
                        "content": str(result)
                    })

    finally:
        client.close()

if __name__ == "__main__":
    main()