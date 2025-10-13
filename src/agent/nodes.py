
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain.chat_models import init_chat_model
from langgraph.store.mongodb import MongoDBStore
from langmem.short_term import SummarizationNode
from src.agent.state import AgentState, LLMInputState
from src.memory.managers import MemoryManagers
from src.memory.models import RunningSummary
from src.config import config
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def create_summarization_node():
    """
    Create the conversation summarization node.

    This node compresses long conversations to fit within context limits.
    It uses LangMem's SummarizationNode to create rolling summaries.

    Returns:
        SummarizationNode instance configured for our use case
    """
    model = init_chat_model(config.llm.model)

    return SummarizationNode(
        token_counter=count_tokens_approximately,
        model=model.bind(max_tokens=config.memory.max_summary_tokens),
        max_tokens=config.memory.summarization_threshold,
        max_tokens_before_summary=config.memory.summarization_threshold,
        max_summary_tokens=config.memory.max_summary_tokens,
    )


def check_summarization_node(state: AgentState) -> AgentState:
    """
    Check if conversation needs summarization.

    This node counts tokens in the conversation and sets a flag
    if summarization is needed.

    Args:
        state: Current agent state

    Returns:
        Updated state with needs_summarization flag set
    """
    try:
        # Count tokens in the conversation
        token_count = count_tokens_approximately(state["messages"])

        logger.info(f"Current conversation has {token_count} tokens")

        # Check if we exceed the threshold
        needs_summary = token_count > config.memory.summarization_threshold

        # Update state
        state["needs_summarization"] = needs_summary

        if needs_summary:
            logger.info("Conversation exceeds threshold, will summarize")

        return state

    except Exception as e:
        logger.error(f"Error in check_summarization_node: {e}")
        # Continue without summarization on error
        state["needs_summarization"] = False
        return state


def summarize_conversation_node(state: AgentState) -> AgentState:
    """
    Compress the conversation using rolling summarization.

    This node is called when the conversation gets too long. It:
    1. Creates a summary of older messages
    2. Keeps recent messages intact
    3. Stores the summary in state["context"]

    Args:
        state: Current agent state

    Returns:
        Updated state with compressed conversation
    """
    try:
        if not state.get("needs_summarization", False):
            # No summarization needed
            return state

        logger.info("Starting conversation summarization")

        # Get or create summarization node
        summarization_node = create_summarization_node()

        # Apply summarization
        # This modifies state["context"] with running summary
        summarized_state = summarization_node(state)

        logger.info("Conversation summarized successfully")

        return summarized_state

    except Exception as e:
        logger.error(f"Error in summarize_conversation_node: {e}")
        # Return original state on error
        return state


def agent_node_1(state: AgentState, store: MongoDBStore) -> AgentState:
    """
    Main agent node that processes user queries.

    This is the core of the agent. It:
    1. Retrieves relevant memories from the store
    2. Builds context from memories and conversation
    3. Calls the LLM to generate a response
    4. Returns the response in updated state

    Args:
        state: Current agent state
        store: MongoDBStore for accessing memories

    Returns:
        Updated state with agent's response
    """
    try:
        customer_id = state["customer_id"]
        messages = state["messages"]

        # Get the latest user message
        latest_message = messages[-1].content if messages else ""

        logger.info(f"Agent processing query for customer {customer_id}")

        # Retrieve relevant memories
        # 1. Search episodic memories (past interactions)
        episodic_memories = store.search(
            ("customers", customer_id, "episodes"),
            query=latest_message,
            limit=3
        )

        # 2. Get semantic facts (preferences)
        semantic_memories = store.search(
            ("customers", customer_id, "preferences"),
            limit=10
        )

        # 3. Get consolidated insights
        insights = store.search(
            ("customers", customer_id, "insights"),
            limit=5
        )

        logger.info(
            f"Retrieved {len(episodic_memories)} episodes, "
            f"{len(semantic_memories)} preferences, "
            f"{len(insights)} insights"
        )

        # Build memory context for the LLM
        memory_context = _build_memory_context(
            episodic_memories,
            semantic_memories,
            insights
        )

        # Create system prompt with memory context
        system_prompt = f"""You are a helpful store associate at a retail store.
        
You have access to the customer's history and preferences to provide personalized service.

{memory_context}

Guidelines:
- Be friendly, helpful, and professional
- Reference past interactions naturally when relevant
- Make personalized recommendations based on preferences
- Ask clarifying questions when needed
- Use tools to search products or get more information
- Be honest if you don't have information

Current conversation:
"""

        # Initialize LLM
        model = init_chat_model(
            config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens
        )

        # Bind tools to the model
        from src.agent.tools import search_products, get_customer_profile, get_purchase_history
        from langmem import create_manage_memory_tool, create_search_memory_tool

        # Create memory tools
        manage_memory = create_manage_memory_tool()
        search_memory = create_search_memory_tool()

        tools = [
            search_products,
            get_customer_profile,
            get_purchase_history,
            manage_memory,
            search_memory
        ]

        model_with_tools = model.bind_tools(tools)

        # Prepare messages with system prompt
        llm_messages = [
            HumanMessage(content=system_prompt)
        ] + messages

        # Get response from LLM
        response = model_with_tools.invoke(llm_messages)

        # Add response to messages
        state["messages"] = messages + [response]

        logger.info("Agent response generated successfully")

        return state

    except Exception as e:
        logger.error(f"Error in agent_node: {e}")
        # Add error message to conversation
        error_message = AIMessage(
            content="I apologize, but I encountered an error. Please try again."
        )
        state["messages"] = state["messages"] + [error_message]
        return state

def agent_node(state: AgentState, store: MongoDBStore) -> AgentState:
    """
    Main agent node with LangMem memory tools and improved tool selection.
    """
    try:
        from langchain_core.messages import HumanMessage, AIMessage
        from langchain_anthropic import ChatAnthropic
        from langmem import create_manage_memory_tool, create_search_memory_tool
        
        customer_id = state["customer_id"]
        messages = state["messages"]
        
        logger.info(f"Agent processing query for customer {customer_id}")
        latest_message = messages[-1].content if messages else ""
        
        # ADD THIS DEBUG
        logger.info(f"=== AGENT DECISION DEBUG ===")
        logger.info(f"User asked: {latest_message}")
        logger.info(f"Customer ID: {customer_id}")
        logger.info(f"Available tools: {[tool.name for tool in tools]}")
        logger.info(f"=== END DEBUG ===")
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
            from langchain_core.messages import ToolMessage
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
def agent_node_b(state: AgentState, store: MongoDBStore) -> AgentState:
    """Main agent node - simplified version"""
    try:
        from langchain_core.messages import AIMessage
        from langchain.chat_models import init_chat_model

        customer_id = state["customer_id"]
        messages = state["messages"]

        logger.info(f"Agent processing query for customer {customer_id}")

        # Simple system prompt without memories for now
        system_prompt = """You are a helpful store associate at a retail store.
        Be friendly, helpful, and professional."""

        # Initialize LLM
        model = init_chat_model(
            config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens
        )

        # Get response from LLM
        from langchain_core.messages import SystemMessage
        llm_messages = [SystemMessage(content=system_prompt)] + messages
        response = model.invoke(llm_messages)

        # Add response to messages
        state["messages"] = messages + [response]

        logger.info("Agent response generated successfully")
        return state

    except Exception as e:
        logger.error(f"Error in agent_node: {e}")
        from langchain_core.messages import AIMessage
        error_message = AIMessage(
            content="I apologize, but I encountered an error. Please try again."
        )
        state["messages"] = state["messages"] + [error_message]
        return state


def _build_memory_context(episodic, semantic, insights) -> str:
    """
    Build a formatted memory context string for the LLM.

    Args:
        episodic: List of episodic memories
        semantic: List of semantic memories
        insights: List of consolidated insights

    Returns:
        Formatted string with memory context
    """
    context_parts = []

    # Add episodic memories
    if episodic:
        context_parts.append("## Past Interactions:")
        for memory in episodic[:2]:  # Limit to 2 most relevant
            context_parts.append(f"- {memory.value.get('summary', 'N/A')}")

    # Add semantic preferences
    if semantic:
        context_parts.append("\n## Customer Preferences:")
        for memory in semantic[:5]:  # Limit to 5 most relevant
            pref_type = memory.value.get('preference_type', 'unknown')
            value = memory.value.get('value', 'N/A')
            context_parts.append(f"- {pref_type}: {value}")

    # Add consolidated insights
    if insights:
        context_parts.append("\n## Behavioral Patterns:")
        for insight in insights[:2]:  # Limit to 2 most relevant
            pattern = insight.value.get('pattern', 'N/A')
            context_parts.append(f"- {pattern}")

    return "\n".join(context_parts) if context_parts else "No previous memory available."


def extract_semantic_memories_node(state: AgentState, store: MongoDBStore) -> AgentState:
    """
    Extract semantic memories from the conversation (real-time).

    This node runs during the conversation (hot path) to capture
    important facts and preferences as they're mentioned.

    Args:
        state: Current agent state
        store: MongoDBStore for storing memories

    Returns:
        Updated state (unchanged, memories stored in background)
    """
    try:
        customer_id = state["customer_id"]
        messages = state["messages"]

        # Only process if we have recent messages
        if len(messages) < 2:
            return state

        logger.info("Extracting semantic memories from recent messages")

        # Get semantic memory manager
        semantic_manager = MemoryManagers.get_semantic_manager()

        # Extract memories from last 2 messages (user + assistant)
        recent_messages = messages[-2:]
        extracted_memories = semantic_manager.extract_memories(
            messages=recent_messages
        )

        logger.info(f"Extracted {len(extracted_memories)} semantic memories")

        # Store each memory in the preferences namespace
        for memory in extracted_memories:
            store.put(
                namespace=("customers", customer_id, "preferences"),
                key=memory.preference_type,
                value={
                    "preference_type": memory.preference_type,
                    "value": memory.value,
                    "confidence": memory.confidence,
                    "source": memory.source,
                    "first_observed": memory.first_observed.isoformat(),
                    "last_confirmed": memory.last_confirmed.isoformat(),
                    "times_observed": memory.times_observed
                }
            )

            logger.info(
                f"Stored preference: {memory.preference_type} = {memory.value}"
            )

        return state

    except Exception as e:
        logger.error(f"Error in extract_semantic_memories_node: {e}")
        # Don't fail the flow on memory extraction errors
        return state


def create_episode_node(state: AgentState, store: MongoDBStore) -> AgentState:
    """
    Create an episode summary when the conversation ends.

    This node runs when session_active becomes False. It:
    1. Summarizes the entire conversation
    2. Extracts key information
    3. Stores the episode with embeddings for future search

    Args:
        state: Current agent state
        store: MongoDBStore for storing episodes

    Returns:
        Updated state (unchanged, episode stored in background)
    """
    try:
        # Only create episode if session is ending
        if state.get("session_active", True):
            return state

        customer_id = state["customer_id"]
        messages = state["messages"]

        logger.info(f"Creating episode summary for customer {customer_id}")

        # Get episode memory manager
        episode_manager = MemoryManagers.get_episode_manager()

        # Extract episode from full conversation
        episodes = episode_manager.extract_memories(messages=messages)

        if not episodes:
            logger.warning("No episode extracted from conversation")
            return state

        # Take the first (and typically only) episode
        episode = episodes[0]

        logger.info(f"Episode summary: {episode.summary[:100]}...")

        # Store episode in MongoDB
        episode_key = f"episode_{datetime.now().isoformat()}"

        store.put(
            namespace=("customers", customer_id, "episodes"),
            key=episode_key,
            value={
                "date": episode.date,
                "summary": episode.summary,
                "customer_needs": episode.customer_needs,
                "products_discussed": episode.products_discussed,
                "outcome": episode.outcome,
                "key_insights": episode.key_insights,
                "sentiment": episode.sentiment,
                "duration_minutes": episode.duration_minutes,
                "associate_id": episode.associate_id,
                "created_at": datetime.now().isoformat()
            },
            index=True  # Enable vector search on this episode
        )

        logger.info(f"Episode stored successfully: {episode_key}")

        return state

    except Exception as e:
        logger.error(f"Error in create_episode_node: {e}")
        # Don't fail the flow on episode creation errors
        return state
