"""Demo graph that exercises nested LangGraph subgraphs."""

from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from demo.api.graphs.subgraphs.specialist_team import create_specialist_team_graph
from langgraph_openai_serve import GraphConfig
from langgraph_openai_serve.api.chat.schemas import ChatCompletionRequest


class QuestionInput(BaseModel):
    question: str


def request_to_input(
    request: ChatCompletionRequest,
    messages: list[BaseMessage],
) -> dict[str, str]:
    last_message = messages[-1]
    input_model = QuestionInput(question=str(last_message.content or ""))
    return {"question": input_model.question}


def output_to_text(output: dict[str, Any]) -> str:
    return output["answer"]


def create_complex_subgraphs_graph_config() -> GraphConfig:
    """Create the complex subgraphs demo config during app setup."""

    return GraphConfig(
        graph=create_specialist_team_graph(),
        request_to_input=request_to_input,
        output_to_text=output_to_text,
        streamable_node_names=[
            "extract_keywords",
            "summarize_contract",
            "summarize_docs",
        ],
    )
