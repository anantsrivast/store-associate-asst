
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool as create_tool
from langchain_core.messages.utils import count_tokens_approximately
from langchain_anthropic import ChatAnthropic
from langgraph.store.mongodb import MongoDBStore
from langmem.short_term import SummarizationNode
from langchain.chat_models import init_chat_model
from src.agent.state import AgentState
from src.config import config
from datetime import datetime
import uuid
import logging
import os

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
    
    This node:
    1. Checks if session is ending (skip processing if so)
    2. Creates memory tools for the customer
    3. Prepares system prompt with clear tool usage guidelines
    4. Runs agentic loop (up to 5 iterations) for multi-tool calls
    5. Forces final response if max iterations reached
    """
    try:
        # Check if session is ending - skip agent processing if so
        if not state.get("session_active", True):
            logger.info("Session ending - skipping agent processing")
            return state
        
        customer_id = state["customer_id"]
        messages = state["messages"]
        
        logger.info(f"Agent processing query for customer {customer_id}")
        
        # Memory namespace for this customer
        memory_namespace = ("customers", customer_id, "memories")
        
        # ============= MEMORY TOOLS =============
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
                
                return f"✓ Memory saved: {content[:100]}..."
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
        
        # ============= SYSTEM PROMPT =============
        system_prompt = f"""You are a helpful store associate at a retail store helping customer {customer_id}.

🧠 CONVERSATION CONTEXT:
You have access to the FULL conversation history. Before responding:
- Read the ENTIRE conversation to understand context
- If customer says "yes/sure/okay" after you offered something → DO that thing immediately
- Understand pronouns ("it", "that", "those") refer to things just discussed
- Don't treat follow-ups as new conversations

Example:
You: "Would you like me to show you Nike shoes in size 10?"
Customer: "Yes please"
You: [Call search_products("Nike shoes size 10")] ← DO IT NOW, don't ask again!

═══════════════════════════════════════════════════════════════════

🚨 CRITICAL RULE 1: PRODUCT DISPLAY FORMAT

When search_products returns results, you MUST display them immediately in this EXACT format:

Here are [description]:
1. **Product Name** - $Price - Brand
2. **Product Name** - $Price - Brand
3. **Product Name** - $Price - Brand

✅ CORRECT:
"Here are some Nike running shoes for you:
1. **Nike Pegasus 40** - $139.99 - Nike
2. **Nike React Infinity** - $159.99 - Nike
3. **Nike Air Zoom** - $119.99 - Nike"

❌ WRONG:
- "Would you like to see these options?" (Show them NOW!)
- "I found some great shoes" (What shoes? List them!)
- "Here are some options for you" (Which options? Display them!)

NEVER reference products without showing them. Always present the numbered list immediately after calling search_products.

═══════════════════════════════════════════════════════════════════

🚨 CRITICAL RULE 2: WHEN TO USE TOOLS vs RESPOND NATURALLY

✅ RESPOND NATURALLY (NO TOOLS) when:
- Customer is chatting casually ("hi", "hello", "how are you", "thanks")
- Customer shares feelings ("I don't like anything these days")
- Customer makes life statements ("I only like walking")
- Customer gives confirmations ("yes please", "sure", "okay") → DO what you offered
- Conversation is emotional or empathetic

❌ USE TOOLS when:
- Customer explicitly asks for products ("show me shoes", "find Nike shoes")
- Customer asks about history ("what did I buy?", "what do you know about me?")
- Customer mentions new preferences ("I like Nike", "I wear size 10")
- You need information you don't have

═══════════════════════════════════════════════════════════════════

🔧 TOOL SELECTION GUIDE

**1. get_customer_profile('{customer_id}')**
Use for: Structured data (name, email, shoe_size, preferred_brands)
Example: "What's my shoe size?" → get_customer_profile

**2. update_customer_profile('{customer_id}', updates)**
Use for: Updating structured data
Example: "I wear size 10" → update_customer_profile('{customer_id}', {{"shoe_size": 10}})
Note: ALSO call manage_memory to save conversational context

**3. search_products(query, category=None)**
Use for: Finding products to show
Example: "Show me Nike shoes" → search_products("Nike shoes")
CRITICAL: Always DISPLAY results in numbered format (see Rule 1)

**4. get_purchase_history('{customer_id}')**
Use for: Past purchase data
Example: "What did I buy?" → get_purchase_history

**5. search_memory(query)**
Use for: Conversational context, preferences, past discussions
When: ALWAYS call FIRST when customer asks for products
Example: "Show me shoes" → search_memory("preferences brands") THEN search_products

**6. manage_memory(content)**
Use for: Saving new preferences and context
When: Customer mentions preferences, after showing products they liked
Example: "I like Nike" → After showing Nike → manage_memory("Customer prefers Nike brand")

═══════════════════════════════════════════════════════════════════

📋 MANDATORY WORKFLOWS

**WORKFLOW 1: Product Request (e.g., "Show me shoes")**
Step 1: search_memory("shoe preferences brand size style history")
Step 2: search_products(query based on memory results)
Step 3: DISPLAY products immediately in numbered format
Step 4: If customer showed interest → manage_memory(preference)

**WORKFLOW 2: Customer Mentions Preference (e.g., "I like Nike")**
Step 1: search_products(preference)
Step 2: DISPLAY products in numbered format
Step 3: manage_memory("Customer prefers [preference]")

**WORKFLOW 3: Confirmation (e.g., "Yes please")**
Step 1: Look at YOUR previous message to see what you offered
Step 2: DO that thing immediately (call search_products)
Step 3: DISPLAY results

**WORKFLOW 4: Customer Shares Data (e.g., "I wear size 10")**
Step 1: update_customer_profile('{customer_id}', {{"shoe_size": 10}})
Step 2: manage_memory("Customer wears shoe size 10")
Step 3: Confirm: "Got it! I've saved that you wear size 10."

═══════════════════════════════════════════════════════════════════

🎯 CRITICAL MEMORY RULES

1. **ALWAYS search_memory FIRST** for product requests
   - Don't skip this even if query seems generic
   - Use results to personalize recommendations

2. **ALWAYS manage_memory** when customer mentions preferences
   - Brand preferences: "Customer prefers Nike"
   - Size info: "Customer wears size 10"
   - Activities: "Customer enjoys walking at night"

3. **DON'T ask twice after confirmations**
   - If you offered something and customer says "yes" → DO IT
   - Read conversation history before responding

4. **Format for manage_memory**:
   - Clear, factual: "Customer prefers [X]" or "Customer mentioned [Y]"

═══════════════════════════════════════════════════════════════════

📝 RESPONSE GUIDELINES

✓ Be warm, friendly, and conversational
✓ Show empathy when customers share feelings
✓ Present products immediately after searching (numbered list)
✓ Use memory to personalize recommendations
✓ Understand confirmations and follow-ups

✗ Don't say "these options" without showing them
✗ Don't ask "what would you like?" after customer confirmed
✗ Don't search products without checking memory first
✗ Don't forget to save preferences when mentioned

Remember: You're a helpful human who remembers preferences, understands context, and shows products clearly!
"""
        
        # ============= INITIALIZE LLM =============
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
        
        # ============= IMPORT AND BIND TOOLS =============
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
        
        # ============= PREPARE MESSAGES =============
        full_messages = [{"role": "system", "content": system_prompt}]
        
        for msg in messages:
            if hasattr(msg, 'content'):
                if msg.__class__.__name__ == 'HumanMessage':
                    full_messages.append({"role": "user", "content": msg.content})
                elif msg.__class__.__name__ == 'AIMessage':
                    # Handle AIMessage with or without tool calls
                    msg_dict = {"role": "assistant", "content": msg.content if msg.content else ""}
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        msg_dict["tool_calls"] = msg.tool_calls
                    full_messages.append(msg_dict)
                elif msg.__class__.__name__ == 'ToolMessage':
                    full_messages.append({
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": msg.tool_call_id
                    })
        
        # ============= AGENTIC LOOP (UP TO 5 ITERATIONS) =============
        current_messages = list(messages)
        max_iterations = 5
        
        for iteration in range(max_iterations):
            logger.info(f"Agent iteration {iteration + 1}/{max_iterations}")
            
            # FORCE FINAL RESPONSE on last iteration
            if iteration == max_iterations - 1:
                logger.warning(f"Reached max iterations ({max_iterations}). Forcing final response without tools.")
                
                # Use LLM without tools to guarantee text response
                llm_no_tools = ChatAnthropic(
                    model=config.llm.model,
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_tokens,
                    api_key=config.llm.anthropic_api_key
                )
                
                force_msg = {
                    "role": "user", 
                    "content": "Please provide your final response to the customer now based on what you know. Do NOT call any more tools."
                }
                
                response = llm_no_tools.invoke(full_messages + [force_msg])
                current_messages.append(response)
                logger.info("✓ Forced final response generated")
                break
            
            # Normal iteration with tools
            response = llm_with_tools.invoke(full_messages)
            current_messages.append(response)
            
            # Check if agent wants to call tools
            if not (hasattr(response, 'tool_calls') and response.tool_calls):
                logger.info(f"✓ Agent completed naturally after {iteration + 1} iteration(s)")
                break
            
            # Agent wants to call tools
            logger.info(f"Agent wants to call {len(response.tool_calls)} tool(s)")
            
            # IMPROVED LOGGING: Show which tools and args
            for i, tc in enumerate(response.tool_calls, 1):
                logger.info(f"  Tool {i}: {tc['name']} | Args: {tc.get('args', {})}")
            
            # Execute tools
            tool_messages = []
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                
                # Find the tool
                tool_to_execute = None
                for t in tools:
                    if t.name == tool_name:
                        tool_to_execute = t
                        break
                
                if tool_to_execute:
                    try:
                        result = tool_to_execute.invoke(tool_args)
                        
                        # Log tool result (first 200 chars)
                        result_preview = str(result)[:200]
                        logger.info(f"  ✓ {tool_name} returned: {result_preview}{'...' if len(str(result)) > 200 else ''}")
                        
                        tool_messages.append(
                            ToolMessage(
                                content=str(result),
                                tool_call_id=tool_call['id']
                            )
                        )
                    except Exception as e:
                        logger.error(f"  ✗ Error executing {tool_name}: {e}")
                        tool_messages.append(
                            ToolMessage(
                                content=f"Error: {str(e)}",
                                tool_call_id=tool_call['id']
                            )
                        )
                else:
                    logger.error(f"  ✗ Unknown tool: {tool_name}")
            
            # Add tool results to conversation
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
        
        # Skip if no meaningful messages
        if len(messages) < 2:
            logger.info("Not enough messages to create episode")
            return state
        
        episode_key = f"episode_{uuid.uuid4().hex[:8]}"
        
        # Extract conversation preview (last 10 messages)
        message_previews = []
        for msg in messages[-10:]:
            if hasattr(msg, 'content') and msg.content:
                # Handle both string and list content
                if isinstance(msg.content, str):
                    content = msg.content[:100]
                elif isinstance(msg.content, list):
                    # Extract text from content blocks
                    text_parts = []
                    for block in msg.content:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            text_parts.append(block.get('text', ''))
                        elif isinstance(block, str):
                            text_parts.append(block)
                    content = ' '.join(text_parts)[:100]
                else:
                    continue
                
                if content.strip():
                    message_previews.append(content.strip())

        if not message_previews:
            logger.info("No content to create episode from")
            return state

        conversation_preview = " | ".join(message_previews)

        # Create summary with substantial text
        episode_summary = {
            "date": datetime.now().isoformat(),
            "summary": f"Conversation with {len(messages)} messages. Topics discussed: {conversation_preview}",
            "message_count": len(messages),
            "created_at": datetime.now().isoformat()
        }
        
        # Store without embeddings to avoid empty list error
        store.put(
            namespace=("customers", customer_id, "episodes"),
            key=episode_key,
            value=episode_summary,
            index=False  # Disable vector indexing to avoid embedding errors
        )
        
        logger.info(f"Episode stored successfully: {episode_key}")
        return state
        
    except Exception as e:
        logger.error(f"Error in create_episode_node: {e}")
        import traceback
        traceback.print_exc()
        return state
