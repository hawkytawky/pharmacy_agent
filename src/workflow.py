"""Langgraph workflow definition for the Pharmacy Assistant chatbot."""

import logging
from typing import Generator, Literal, Union

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from config.chat_config import (
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
)
from config.secrets import get_secret
from src.prompts import SYSTEM_PROMPT
from src.schemas import AgentState
from src.tools import pharmacy_tools

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PharmacyAssistant:
    """
    AI-powered pharmacy assistant chatbot using Langgraph.

    The agent follows a ReAct pattern:
    1. Receives user message
    2. Decides whether to use a tool or respond directly
    3. If tool needed, executes tool and returns to step 2
    4. If no tool needed, responds to user

    Attributes:
        model: The LLM model name to use.
        temperature: The temperature setting for the LLM.
        chat_history: Internal message history (includes ToolMessages for API).
        display_history: Messages for UI display (user input and AI responses).
    """

    def __init__(self, model: str = LLM_MODEL, temperature: float = LLM_TEMPERATURE):
        """
        Initializes the pharmacy assistant.

        Args:
            model: The LLM model name to use.
            temperature: The temperature setting for the LLM.
        """
        self.model = model
        self.temperature = temperature
        self._llm_with_tools = self._create_llm()
        self.graph = self._build_agent()

    def _create_llm(self) -> Union[ChatOpenAI, ChatOllama]:
        """
        Creates the LLM with tools bound.

        Supports both OpenAI and Ollama providers based on config.

        Returns:
            ChatOpenAI or ChatOllama instance with tools bound.
        """
        if LLM_PROVIDER == "ollama":
            llm = ChatOllama(
                model=self.model,
                temperature=self.temperature,
            )
        elif LLM_PROVIDER == "openai":
            llm = ChatOpenAI(
                model=self.model,
                api_key=get_secret("OPENAI_API_KEY"),
                temperature=self.temperature,
            )
        return llm.bind_tools(pharmacy_tools)

    def _should_continue(self, state: AgentState) -> Literal["tools", END]:
        """
        Determines if the agent should continue to tools or end.

        Args:
            state: Current agent state containing messages.

        Returns:
            "tools" if the last message has tool calls, END otherwise.
        """
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    def _call_model(self, state: AgentState) -> dict:
        """
        Invokes the LLM with the current conversation state.

        Args:
            state: Current agent state containing messages.

        Returns:
            Dictionary with updated messages including LLM response.
        """
        messages = state["messages"]

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        else:
            first_system = messages[0]
            history = [m for m in messages[1:] if not isinstance(m, SystemMessage)]
            messages = [first_system] + history

        response = self._llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _build_agent(self) -> StateGraph:
        """
        Builds and compiles the Langgraph agent.

        Returns:
            Compiled StateGraph representing the pharmacy assistant workflow.
        """
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(pharmacy_tools))
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", self._should_continue, {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def stream_chat(
        self, user_input: str, thread_id: str
    ) -> Generator[str, None, None]:
        """
        Generator function for Streamlit's st.write_stream.
        Yields chunks of text from the LLM.
        """
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=user_input)]}

        # LangGraph Stream Loop
        for msg, metadata in self.graph.stream(inputs, config, stream_mode="messages"):
            if msg.content and isinstance(msg, AIMessage) and not msg.tool_calls:
                yield msg.content

    def clear_memory(self, thread_id: str):
        """Clears the LangGraph memory for this thread."""
        pass
