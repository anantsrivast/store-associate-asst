from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.store.mongodb import MongoDBStore
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool as create_tool
from src.agent.state import AgentState
from src.database.mongodb_client import db_manager
from src.config import config
import logging

logger = logging.getLogger(__name__)


def agent_node(state: AgentState, store: MongoDBStore) -> AgentState:
    """
    Main agent node using LangGraph MongoDBStore with vector search.
    """
    try:
        customer_id = state["customer_id"]
        messages = state["messages"]
        
        logger.info(f"Agent processing query for customer {customer_id}")
        
        # Memory namespace for this customer
        memory_namespace = ("customers", customer_id, "memories")
        
        # Create memory tools using LangGraph store
        @create_tool
        def manage_memory(content: str) -> str:
            """
            Store important information about the customer for future reference.
            
            Use this to save:
            - Customer preferences and interests
            - Important context from conversations
            - Things the customer wants you to remember
            
            Args:
                content: The information to remember (as a clear, concise string)
            """
            try:
                import uuid
                from datetime import datetime
                
                # Create a unique key
                key = f"memory_{uuid.uuid4().hex[:8]}"
                
                # Store directly to avoid LangMem's internal data structure issues
                value = {
                    "content": str(content),
                    "timestamp": datetime.now().isoformat(),
                    "type": "user_preference"
                }
                
                # ADD ALL THESE DEBUG LINES
                logger.info(f"=== DEBUG manage_memory ===")
                logger.info(f"Namespace: {memory_namespace}")
                logger.info(f"Key: {key}")
                logger.info(f"Value: {value}")
                logger.info(f"Value type: {type(value)}")
                logger.info(f"Content type: {type(value['content'])}")
                logger.info(f"Content length: {len(value['content'])}")
                logger.info(f"=== END DEBUG ===")
                
                # Put directly into store
                store.put(
                    namespace=memory_namespace,
                    key=key,
                    value=value
                )
                
                return f"Successfully saved memory: {content[:100]}..."
            except Exception as e:
                logger.error(f"Error saving memory: {e}", exc_info=True)
                return f"Error saving memory: {str(e)}"
        
        @create_tool
        def search_memory(query: str) -> str:
            """
            Search for relevant past interactions and stored information about the customer.
            
            Use this to recall:
            - What you know about the customer
            - Past conversations and context
            - Customer preferences and history
            
            Args:
                query: What to search for in past memories
            """
            try:
                # Search in the store - namespace is POSITIONAL argument
                results = store.search(
                    memory_namespace,  # First positional argument
                    query=query,
                    limit=5
                )
                
                if not results:
                    return "No relevant memories found."
                
                # Format results with more detail
                memories = []
                for idx, item in enumerate(results, 1):
                    content = item.value.get("content", "")
                    timestamp = item.value.get("timestamp", "")
                    memory_type = item.value.get("type", "general")
                    
                    # Format the timestamp nicely
                    timestamp_str = timestamp[:10] if timestamp else "unknown date"
                    
                    memories.append(f"{idx}. {content} (saved: {timestamp_str}, type: {memory_type})")
                
                logger.info(f"Found {len(results)} memories for query: {query}")
                
                # Return formatted memories
                return "Found relevant memories:\n" + "\n".join(memories)
                
            except Exception as e:
                logger.error(f"Error searching memories: {e}", exc_info=True)
                return f"Error searching memories: {str(e)}"
        
        # Enhanced system prompt with clear tool selection guidelines
        system_prompt = f"""You are a helpful and friendly store associate at a retail store.

CUSTOMER INFORMATION:
You are currently helping customer_id: {customer_id}

MULTI-STEP WORKFLOW:
When a customer asks for recommendations, follow this process:
1. FIRST: Use search_memory to understand their preferences
2. SECOND: Use search_products based on what you learned
3. THIRD: Present the products with personalized context

Example:
Customer: "Show me products based on my interests"
Step 1: search_memory("interests preferences activities") 
Step 2: search_products("running shoes hiking gear") based on memory results
Step 3: Present: "Based on your love of running and hiking, here are some great options..."

CRITICAL INSTRUCTION - SHOWING SEARCH RESULTS:
When you use search_products and it returns results, YOU MUST immediately show those products to the customer in your response. Do NOT stop after just searching memory - continue to search products and show results.

TOOL SELECTION GUIDELINES:

1. **For STRUCTURED FACTS** (shoe size, email, name):
   → Use get_customer_profile('{customer_id}')

2. **For CONVERSATIONAL CONTEXT and PREFERENCES**:
   → Use search_memory(query)
   → Then USE the results to inform product searches

3. **For PRODUCT SEARCHES**:
   → Use search_products(query)
   → Can call multiple times with different queries
   → ALWAYS present the results immediately

4. **When customer asks for recommendations "based on my interests"**:
   → Step 1: search_memory("interests preferences activities")
   → Step 2: search_products(relevant_query) using what you learned
   → Step 3: Present products with personalized context

5. **To save information**:
   → Structured facts → update_customer_profile
   → Conversational context → manage_memory

Remember: Complete the full workflow before responding. If you search memory, use those results to search products too!

Your goal is to be proactive, helpful, and show actual products, not just acknowledge the request.
"""
        
        # Initialize LLM with proper API key handling
        if config.llm.anthropic_api_key:
            llm = ChatAnthropic(
                model=config.llm.model,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                api_key=config.llm.anthropic_api_key
            )
        elif config.llm.openai_api_key:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
                api_key=config.llm.openai_api_key
            )
        else:
            raise ValueError("Either ANTHROPIC_API_KEY or OPENAI_API_KEY must be set in .env file")
        
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
        
        # Prepare initial messages with system prompt
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
        
        # ============= AGENT LOOP FOR MULTI-TOOL CALLS =============
        max_iterations = 5
        current_messages = messages.copy()
        
        for iteration in range(max_iterations):
            logger.info(f"Agent iteration {iteration + 1}/{max_iterations}")
            
            # Get response from LLM
            response = llm_with_tools.invoke(full_messages)
            current_messages.append(response)
            
            # Check if there are tool calls
            if not (hasattr(response, 'tool_calls') and response.tool_calls):
                # No tool calls - agent is done
                logger.info(f"Agent completed after {iteration + 1} iteration(s)")
                break
            
            # Execute tool calls
            logger.info(f"Agent wants to call {len(response.tool_calls)} tool(s)")
            
            tool_messages = []
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                
                # Find and execute the tool
                tool_to_execute = None
                for t in tools:
                    if t.name == tool_name:
                        tool_to_execute = t
                        break
                
                if tool_to_execute:
                    try:
                        result = tool_to_execute.invoke(tool_args)
                        tool_messages.append(
                            ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call['id']
                            )
                        )
                        logger.info(f"Tool {tool_name} executed successfully")
                    except Exception as e:
                        logger.error(f"Error executing tool {tool_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        tool_messages.append(
                            ToolMessage(
                                content=f"Error: {str(e)}",
                                tool_call_id=tool_call['id']
                            )
                        )
            
            # Add tool results to messages
            current_messages.extend(tool_messages)
            
            # Update full_messages for next iteration
            full_messages.append({
                "role": "assistant", 
                "content": response.content if response.content else "",
                "tool_calls": response.tool_calls
            })
            for tm in tool_messages:
                full_messages.append({
                    "role": "tool", 
                    "content": str(tm.content), 
                    "tool_call_id": tm.tool_call_id
                })
        
        # Update state with all messages
        state["messages"] = current_messages
        
        logger.info("Agent response generated successfully")
        return state
        
    except Exception as e:
        logger.error(f"Error in agent_node: {e}")
        import traceback
        traceback.print_exc()
        
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
    
    # Compile the graph with checkpointer
    graph = workflow.compile(
        checkpointer=checkpointer
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
            Agent's response text (only the final text, not tool calls)
        """
        # Create config with thread_id for checkpointing
        cfg = {
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
        result = self.graph.invoke(input_state, cfg)
        
        # Extract the last AIMessage with actual text content
        result_messages = result["messages"]
        
        # Find the last AI message that has text content (not tool calls)
        for msg in reversed(result_messages):
            if isinstance(msg, AIMessage):
                # Check if it has text content (not just tool calls)
                if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
                    return msg.content
                # If content is a list (like tool calls), skip it
                elif hasattr(msg, 'content') and isinstance(msg.content, list):
                    # Look for text blocks in the content
                    for block in msg.content:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            text = block.get('text', '').strip()
                            if text:
                                return text
        
        return "I apologize, but I couldn't generate a response."
    
    def end_session(self, customer_id: str, thread_id: str):
        """
        End a conversation session.
        
        Args:
            customer_id: Customer identifier
            thread_id: Conversation thread identifier
        """
        logger.info(f"Session ended for thread {thread_id}")