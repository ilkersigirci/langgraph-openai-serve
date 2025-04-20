from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequestMessage


def convert_to_lc_messages(
    messages: list[ChatCompletionRequestMessage],
) -> list[BaseMessage]:
    """Convert OpenAI messages to LangChain messages.

    This function converts a list of OpenAI-compatible message objects to their
    LangChain equivalents for use with LangGraph.

    Args:
        messages: A list of OpenAI chat completion request messages to convert.

    Returns:
        A list of LangChain message objects.
    """

    lc_messages = []
    for m in messages:
        if m.role == "system":
            lc_messages.append(SystemMessage(content=m.content or ""))
        elif m.role == "user":
            lc_messages.append(HumanMessage(content=m.content or ""))
        elif m.role == "assistant":
            lc_messages.append(AIMessage(content=m.content or ""))
    return lc_messages
