from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.store.mongodb import MongoDBStore
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.agent.state import AgentState
from src.database.mongodb_client import db_manager
from src.config import config
import logging

logger = logging.getLogger(__name__)


def agent_node(state: AgentState, store: MongoDBStore) -> AgentState:
    """
    Main agent node with LangMem memory tools and improved tool selection.
    This is the function name that matches the graph definition.
    """
    try:
        from langmem import create_manage_memory_tool, create_search_memory_tool
        
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
        
        # Enhanced system prompt with clear tool selection guidelines
        system_prompt = f"""You are a helpful and friendly store associate at a retail store.

CUSTOMER INFORMATION:
You are currently helping customer_id: {customer_id}

TOOL SELECTION GUIDELINES - VERY IMPORTANT:

1. **For factual customer data** (shoe size, name, email, preferred brands, loyalty tier):
   → ALWAYS use get_customer_profile('{customer_id}')
   → This retrieves structured data stored in the customer's profile
   → Example: "What's my shoe size?" → Use get_customer_profile

2. **To update customer data** when they share new facts:
   → Use update_customer_profile('{customer_id}', updates)
   → Example: User says "I wear size 8" → update_customer_profile('{customer_id}', {{"shoe_size": 8}})
   → Then ALSO save context to memory with manage_memory()

3. **For past purchases**:
   → Use get_purchase_history('{customer_id}')
   → Shows what they've bought before

4. **For product searches**:
   → Use search_products(query, category)
   → Example: "Show me running shoes" → search_products("running shoes", "shoes")

5. **For conversational context and preferences** (not structured facts):
   → Use search_memory(query)
   → Example: "What did I say about marathon training?" → search_memory("marathon training")

6. **To save new conversational insights**:
   → Use manage_memory(content)
   → Save contextual information, preferences, conversation summaries
   → Example: "Customer mentioned training for a marathon"

WORKFLOW EXAMPLE:
- User: "I wear size 8 Nike shoes"
- Step 1: update_customer_profile('{customer_id}', {{"shoe_size": 8, "preferred_brands": ["Nike"]}})
- Step 2: manage_memory("Customer prefers Nike brand and wears size 8")
- Response: "Got it! I've saved that you wear size 8 and prefer Nike. I'll remember this for next time!"

- User: "What's my shoe size?"
- Step 1: get_customer_profile('{customer_id}')
- Step 2: Read shoe_size from the returned profile
- Response: "Your shoe size is 8!"

GUIDELINES:
- Be friendly, helpful, and professional
- Reference past information naturally when relevant
- Make personalized recommendations based on preferences
- Ask clarifying questions when needed
- Always use the appropriate tool for the type of information requested
"""
        
        # Initialize LLM
        llm = ChatAnthropic(
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            api_key=config.llm.anthropic_api_key
        )
        
        # Import and bind all tools
        from src.agent.tools import (
            search_products, 
            get_customer_profile, 
            update_customer_profile,
            get_purchase_history
        )
        
        tools = [
            get_customer_profile,
            update_customer_profile,
            search_products,
            get_purchase_history,
            manage_memory,
            search_memory
        ]
        
        llm_with_tools = llm.bind_tools(tools)
        
        # Prepare messages with system prompt
        full_messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add conversation messages
        for msg in messages:
            if hasattr(msg, 'content'):
                if msg.__class__.__name__ == 'HumanMessage':
                    full_messages.append({"role": "user", "content": msg.content})
                elif msg.__class__.__name__ == 'AIMessage':
                    full_messages.append({"role": "assistant", "content": msg.content})
        
        # Get response from LLM
        response = llm_with_tools.invoke(full_messages)
        
        # Handle tool calls if present
        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.info(f"Agent wants to call {len(response.tool_calls)} tool(s)")
            
            # Execute each tool call
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                
                # Find and execute the tool
                tool_to_execute = None
                for tool in tools:
                    if tool.name == tool_name:
                        tool_to_execute = tool
                        break
                
                if tool_to_execute:
                    try:
                        result = tool_to_execute.invoke(tool_args)
                        tool_results.append({
                            "tool_call_id": tool_call['id'],
                            "result": result
                        })
                        logger.info(f"Tool {tool_name} executed successfully")
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {e}")
                        tool_results.append({
                            "tool_call_id": tool_call['id'],
                            "error": str(e)
                        })
            
            # Create tool result messages
            tool_messages = []
            for tr in tool_results:
                tool_messages.append(
                    ToolMessage(
                        content=str(tr.get("result", tr.get("error"))),
                        tool_call_id=tr["tool_call_id"]
                    )
                )
            
            # Add original response and tool results to messages
            state["messages"] = messages + [response] + tool_messages
            
            # Get final response after tool execution
            final_response = llm_with_tools.invoke(
                full_messages + [
                    {"role": "assistant", "content": response.content, "tool_calls": response.tool_calls}
                ] + [
                    {"role": "tool", "content": str(tm.content), "tool_call_id": tm.tool_call_id}
                    for tm in tool_messages
                ]
            )
            
            state["messages"] = state["messages"] + [final_response]
        else:
            # No tool calls, just add the response
            state["messages"] = messages + [response]
        
        logger.info("Agent response generated successfully")
        return state
        
    except Exception as e:
        logger.error(f"Error in agent_node: {e}")
        import traceback
        traceback.print_exc()
        
        # Add error message to conversation
        error_message = AIMessage(
            content="I apologize, but I encountered an error. Please try again."
        )
        state["messages"] = state["messages"] + [error_message]
        return state


def extract_semantic_memories_node(state: AgentState, store: MongoDBStore) -> AgentState:
    """
    Background memory extraction - disabled for now.
    The agent manages memory via tools instead.
    """
    logger.info("Background extraction disabled - agent manages memory via tools")
    return state


def create_agent_graph():
    """
    Create and compile the complete agent graph.
    
    The graph flow:
    1. Entry → Agent
    2. Agent → Extract Memories
    3. Extract Memories → END
    
    Returns:
        Compiled graph ready for execution
    """
    logger.info("Building agent graph")
    
    # Get MongoDB components
    checkpointer = db_manager.get_checkpointer()
    store = db_manager.get_store()
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes - using lambda to pass store to the node functions
    workflow.add_node("agent", lambda s: agent_node(s, store))
    workflow.add_node("extract_memories", lambda s: extract_semantic_memories_node(s, store))
    
    # Define edges
    workflow.set_entry_point("agent")
    workflow.add_edge("agent", "extract_memories")
    workflow.add_edge("extract_memories", END)
    
    # Compile the graph with checkpointer and store
    graph = workflow.compile(
        checkpointer=checkpointer,
        store=store
    )
    
    logger.info("Agent graph compiled successfully")
    
    return graph


class StoreAssistantAgent:
    """
    High-level wrapper for the store assistant agent.
    
    This provides a clean interface for interacting with the agent,
    handling configuration and thread management.
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
        messages = result["messages"]
        
        # Find the last AI message
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg.content
        
        return "I apologize, but I couldn't generate a response."
    
    def end_session(self, customer_id: str, thread_id: str):
        """
        End a conversation session.
        
        Args:
            customer_id: Customer identifier
            thread_id: Conversation thread identifier
        """
        logger.info(f"Session ended for thread {thread_id}")