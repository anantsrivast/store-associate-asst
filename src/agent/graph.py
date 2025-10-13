
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.store.mongodb import MongoDBStore
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, AIMessage
from src.agent.state import AgentState
from src.database.mongodb_client import db_manager
from src.config import config
import logging

logger = logging.getLogger(__name__)


def create_agent_graph():
    """
    Create and compile the complete agent graph with memory.
    
    Uses LangMem tools for agent-controlled memory management.
    """
    logger.info("Building agent graph")
    
    # Get MongoDB components
    checkpointer = db_manager.get_checkpointer()
    store = db_manager.get_store()
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", lambda s: agent_node_with_memory(s, store))
    workflow.add_node("extract_memories", lambda s: extract_semantic_memories_node(s, store))
    
    # Define edges
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "extract_memories")
    workflow.add_edge("extract_memories", END)
    
    # Compile the graph
    graph = workflow.compile(
        checkpointer=checkpointer,
        store=store
    )
    
    logger.info("Agent graph compiled successfully")
    return graph



def agent_node_with_memory(state: AgentState, store: MongoDBStore) -> AgentState:
    """
    Main agent node with LangMem memory tools.
    Properly handles tool call execution loop.
    """
    try:
        from langmem import create_manage_memory_tool, create_search_memory_tool
        from langchain_core.messages import ToolMessage
        
        customer_id = state["customer_id"]
        messages = state["messages"]
        
        logger.info(f"Agent processing query for customer {customer_id}")
        
        # Create memory tools for this customer
        memory_namespace = ("customers", customer_id, "memories")
        
        manage_memory = create_manage_memory_tool(
            namespace=memory_namespace
        )
        
        search_memory = create_search_memory_tool(
            namespace=memory_namespace
        )
        
        # System prompt
        system_prompt = f"""You are a helpful store associate at a retail store.

You have access to memory tools:
- Use manage_memory to save customer preferences, sizes, and important details
- Use search_memory to recall past information about the customer

Guidelines:
- Be friendly and professional
- Save important customer details using manage_memory
- Search past interactions using search_memory when relevant
- Provide personalized recommendations

Current customer: {customer_id}
"""
        
        # Initialize LLM with tools
        from langchain.chat_models import init_chat_model
        model = init_chat_model(
            config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens
        )
        
        # Bind tools
        tools = [manage_memory, search_memory]
        tools_by_name = {tool.name: tool for tool in tools}
        model_with_tools = model.bind_tools(tools)
        
        # Prepare messages
        from langchain_core.messages import SystemMessage
        llm_messages = [SystemMessage(content=system_prompt)] + messages
        
        # Tool execution loop
        max_iterations = 5
        for iteration in range(max_iterations):
            # Get response from LLM
            response = model_with_tools.invoke(llm_messages)
            llm_messages.append(response)
            
            # Check if there are tool calls
            if not response.tool_calls:
                # No more tool calls, we're done
                break
            
            # Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                
                try:
                    # Get the tool and execute it
                    tool = tools_by_name[tool_name]
                    
                    # Execute with store context
                    tool_result = tool.invoke(
                        tool_args,
                        config={"store": store}
                    )
                    
                    # Create tool message with result
                    tool_message = ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                    
                    llm_messages.append(tool_message)
                    logger.info(f"Tool {tool_name} executed successfully")
                    
                except Exception as e:
                    logger.error(f"Error executing tool {tool_name}: {e}")
                    # Send error back to agent
                    error_message = ToolMessage(
                        content=f"Error: {str(e)}",
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                    llm_messages.append(error_message)
        
        # Get final response (last message from LLM)
        final_response = llm_messages[-1]
        
        # Add all new messages to state (including tool calls and responses)
        state["messages"] = messages + llm_messages[len(messages) + 1:]
        
        logger.info("Agent response generated successfully")
        return state
        
    except Exception as e:
        logger.error(f"Error in agent_node: {e}")
        import traceback
        traceback.print_exc()
        
        from langchain_core.messages import AIMessage
        error_message = AIMessage(
            content="I apologize, but I encountered an error. Please try again."
        )
        state["messages"] = state["messages"] + [error_message]
        return state


def extract_semantic_memories_node(state: AgentState, store: MongoDBStore) -> AgentState:
    """
    Background memory extraction - simplified version.
    The agent already saves memories via tools, so we don't need this.
    """
    logger.info("Background extraction disabled - agent manages memory via tools")
    return state

class StoreAssistantAgent:
    """
    High-level wrapper for the store assistant agent.
    """
    
    def __init__(self):
        """Initialize the agent with compiled graph"""
        self.graph = create_agent_graph()
        logger.info("StoreAssistantAgent initialized")
    
    def chat(
        self,
        customer_id: str,
        message: str,
        thread_id: str,
        session_active: bool = True
    ) -> str:
        """
        Send a message to the agent and get a response.
        
        Args:
            customer_id: Customer identifier
            message: User's message
            thread_id: Conversation thread identifier
            session_active: Whether the session is still active
            
        Returns:
            Agent's response text
        """
        from langchain_core.messages import HumanMessage
        
        # Create config with thread_id for checkpointing
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        # Create input state
        input_state = {
            "messages": [HumanMessage(content=message)],
            "customer_id": customer_id,
            "session_active": session_active,
            "needs_summarization": False,
            "context": {},
            "metadata": {}
        }
        
        # Invoke the graph
        result = self.graph.invoke(input_state, config)
        
        # Extract the last assistant message
        last_message = result["messages"][-1]
        
        return last_message.content
    
    def end_session(self, customer_id: str, thread_id: str):
        """
        End a conversation session.
        
        Args:
            customer_id: Customer identifier
            thread_id: Conversation thread identifier
        """
        # For now, just log the session end
        # In a full implementation, this would create an episode summary
        logger.info(f"Session ended for thread {thread_id}")
