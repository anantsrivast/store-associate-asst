from langgraph.graph import StateGraph, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.store.mongodb import MongoDBStore
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool as create_tool
from langchain_core.messages.utils import count_tokens_approximately
from langmem.short_term import SummarizationNode
from langchain.chat_models import init_chat_model
from src.agent.state import AgentState
from src.database.mongodb_client import db_manager
from src.config import config
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


def check_summarization_node(state: AgentState) -> AgentState:
    """
    Check if conversation needs summarization.
    
    Counts tokens and sets needs_summarization flag.
    """
    try:
        token_count = count_tokens_approximately(state["messages"])
        logger.info(f"Current conversation has {token_count} tokens")
        
        needs_summary = token_count > config.memory.summarization_threshold
        state["needs_summarization"] = needs_summary
        
        if needs_summary:
            logger.info(f"Conversation exceeds threshold ({config.memory.summarization_threshold}), will summarize")
        
        return state
    except Exception as e:
        logger.error(f"Error in check_summarization_node: {e}")
        state["needs_summarization"] = False
        return state


def summarize_conversation_node(state: AgentState) -> AgentState:
    """
    Compress the conversation using rolling summarization.
    """
    try:
        if not state.get("needs_summarization", False):
            return state
        
        logger.info("Starting conversation summarization")
        
        model = init_chat_model(config.llm.model)
        
        summarization_node = SummarizationNode(
            token_counter=count_tokens_approximately,
            model=model.bind(max_tokens=config.memory.max_summary_tokens),
            max_tokens=config.memory.summarization_threshold,
            max_tokens_before_summary=config.memory.summarization_threshold,
            max_summary_tokens=config.memory.max_summary_tokens,
        )
        
        summarized_state = summarization_node(state)
        logger.info("Conversation summarized successfully")
        
        return summarized_state
    except Exception as e:
        logger.error(f"Error in summarize_conversation_node: {e}")
        return state


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
                key = f"memory_{uuid.uuid4().hex[:8]}"
                
                value = {
                    "content": str(content),
                    "timestamp": datetime.now().isoformat(),
                    "type": "user_preference"
                }
                
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
                results = store.search(
                    memory_namespace,
                    query=query,
                    limit=5
                )
                
                if not results:
                    return "No relevant memories found."
                
                memories = []
                for idx, item in enumerate(results, 1):
                    content = item.value.get("content", "")
                    timestamp = item.value.get("timestamp", "")
                    timestamp_str = timestamp[:10] if timestamp else "unknown date"
                    
                    memories.append(f"{idx}. {content} (saved: {timestamp_str})")
                
                logger.info(f"Found {len(results)} memories for query: {query}")
                
                return "Found relevant memories:\n" + "\n".join(memories)
                
            except Exception as e:
                logger.error(f"Error searching memories: {e}", exc_info=True)
                return f"Error searching memories: {str(e)}"
        
        # Enhanced system prompt
        system_prompt = f"""You are a helpful and friendly store associate at a retail store.

CUSTOMER INFORMATION:
You are currently helping customer_id: {customer_id}

MULTI-STEP WORKFLOW:
When a customer asks for recommendations, follow this process:
1. FIRST: Use search_memory to understand their preferences
2. SECOND: Use search_products based on what you learned
3. THIRD: Present the products with personalized context

TOOL SELECTION GUIDELINES:

1. **For STRUCTURED FACTS** (shoe size, email, name):
   → Use get_customer_profile('{customer_id}')

2. **For CONVERSATIONAL CONTEXT and PREFERENCES**:
   → Use search_memory(query)

3. **For PRODUCT SEARCHES**:
   → Use search_products(query)

4. **For PURCHASE HISTORY**:
   → Use get_purchase_history('{customer_id}')

5. **To save information**:
   → Structured facts → update_customer_profile
   → Conversational context → manage_memory

Your goal is to be proactive, helpful, and show actual products.
"""
        
        # Initialize LLM
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
        
        # Prepare messages
        full_messages = [{"role": "system", "content": system_prompt}]
        
        for msg in messages:
            if hasattr(msg, 'content'):
                if msg.__class__.__name__ == 'HumanMessage':
                    full_messages.append({"role": "user", "content": msg.content})
                elif msg.__class__.__name__ == 'AIMessage':
                    full_messages.append({"role": "assistant", "content": msg.content})
        
        # Agentic loop (up to 5 iterations)
        current_messages = list(messages)
        max_iterations = 5
        
        for iteration in range(max_iterations):
            logger.info(f"Agent iteration {iteration + 1}/{max_iterations}")
            
            response = llm_with_tools.invoke(full_messages)
            current_messages.append(response)
            
            if not (hasattr(response, 'tool_calls') and response.tool_calls):
                logger.info(f"Agent completed after {iteration + 1} iteration(s)")
                break
            
            logger.info(f"Agent wants to call {len(response.tool_calls)} tool(s)")
            
            tool_messages = []
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                
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
                        tool_messages.append(
                            ToolMessage(
                                content=f"Error: {str(e)}",
                                tool_call_id=tool_call['id']
                            )
                        )
            
            current_messages.extend(tool_messages)
            
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
    Extract semantic memories from conversation.
    Currently simplified - agent handles via tools.
    """
    logger.info("Extract memories node (agent manages memory via tools)")
    return state


def create_episode_node(state: AgentState, store: MongoDBStore) -> AgentState:
    """
    Create an episode summary when conversation ends.
    """
    try:
        if state.get("session_active", True):
            return state
        
        customer_id = state["customer_id"]
        messages = state["messages"]
        
        logger.info(f"Creating episode summary for customer {customer_id}")
        
        episode_key = f"episode_{uuid.uuid4().hex[:8]}"
        
        episode_summary = {
            "date": datetime.now().isoformat(),
            "summary": f"Conversation with {len(messages)} messages",
            "message_count": len(messages),
            "created_at": datetime.now().isoformat()
        }
        
        store.put(
            namespace=("customers", customer_id, "episodes"),
            key=episode_key,
            value=episode_summary
        )
        
        logger.info(f"Episode stored successfully: {episode_key}")
        return state
        
    except Exception as e:
        logger.error(f"Error in create_episode_node: {e}")
        return state


def route_after_summarization(state: AgentState) -> str:
    """
    Route after checking summarization.
    Returns: "summarize" or "agent"
    """
    if state.get("needs_summarization", False):
        logger.info("Routing to summarize node")
        return "summarize"
    logger.info("Routing directly to agent node")
    return "agent"


def route_after_agent(state: AgentState) -> str:
    """
    Route after agent processing.
    Returns: "create_episode" or "extract_memories"
    """
    if not state.get("session_active", True):
        logger.info("Session ending - routing to create_episode")
        return "create_episode"
    logger.info("Session active - routing to extract_memories")
    return "extract_memories"


def create_agent_graph():
    """
    Create and compile the complete agent graph with full workflow.
    
    Graph Flow:
    START → check_summarization → (conditional)
        ├─> [needs_summarization=True] → summarize → agent
        └─> [needs_summarization=False] → agent
    
    agent → (conditional)
        ├─> [session_active=False] → create_episode → END
        └─> [session_active=True] → extract_memories → END
    
    Returns:
        Compiled graph ready for execution
    """
    logger.info("Building agent graph with full workflow")
    
    checkpointer = db_manager.get_checkpointer()
    store = db_manager.get_store()
    
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("check_summarization", check_summarization_node)
    workflow.add_node("summarize", summarize_conversation_node)
    workflow.add_node("agent", lambda s: agent_node(s, store))
    workflow.add_node("extract_memories", lambda s: extract_semantic_memories_node(s, store))
    workflow.add_node("create_episode", lambda s: create_episode_node(s, store))
    
    # Set entry point
    workflow.set_entry_point("check_summarization")
    
    # Conditional edge from check_summarization
    workflow.add_conditional_edges(
        "check_summarization",
        route_after_summarization,
        {
            "summarize": "summarize",
            "agent": "agent"
        }
    )
    
    # Edge from summarize to agent
    workflow.add_edge("summarize", "agent")
    
    # Conditional edge from agent
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "create_episode": "create_episode",
            "extract_memories": "extract_memories"
        }
    )
    
    # Edges to END
    workflow.add_edge("extract_memories", END)
    workflow.add_edge("create_episode", END)
    
    graph = workflow.compile(checkpointer=checkpointer)
    
    logger.info("Agent graph compiled successfully with full workflow")
    return graph


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
        cfg = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        input_state = {
            "messages": [HumanMessage(content=message)],
            "customer_id": customer_id,
            "session_active": session_active,
            "needs_summarization": False,
            "context": {},
            "metadata": {}
        }
        
        result = self.graph.invoke(input_state, cfg)
        
        result_messages = result["messages"]
        
        for msg in reversed(result_messages):
            if isinstance(msg, AIMessage):
                if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
                    return msg.content
                elif hasattr(msg, 'content') and isinstance(msg.content, list):
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
