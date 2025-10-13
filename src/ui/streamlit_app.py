# ============================================================================
# FILE: src/ui/streamlit_app.py
# DESCRIPTION: Streamlit UI for the Store Assistant demo
# ============================================================================

import streamlit as st
from datetime import datetime
from src.agent.graph import StoreAssistantAgent
from src.database.mongodb_client import db_manager
from src.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Store Associate Assistant",
    page_icon="🛍️",
    layout="wide"
)

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = StoreAssistantAgent()
    logger.info("Agent initialized")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_customer" not in st.session_state:
    st.session_state.current_customer = None

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def load_customers():
    """Load customer list from MongoDB"""
    try:
        collection = db_manager.get_collection(config.mongodb.customers_collection)
        customers = list(collection.find({}, {"customer_id": 1, "name": 1, "email": 1}))
        return customers
    except Exception as e:
        logger.error(f"Error loading customers: {e}")
        return []


def load_customer_memories(customer_id: str):
    """Load memories for a specific customer"""
    try:
        store = db_manager.get_store()
        
        # Get preferences
        preferences = store.search(
            namespace_prefix=("customers", customer_id, "preferences"),
            limit=20
        )
        
        # Get episodes
        episodes = store.search(
            namespace_prefix=("customers", customer_id, "episodes"),
            limit=10
        )
        
        # Get insights
        insights = store.search(
            namespace_prefix=("customers", customer_id, "insights"),
            limit=10
        )
        
        return {
            "preferences": preferences,
            "episodes": episodes,
            "insights": insights
        }
    except Exception as e:
        logger.error(f"Error loading memories: {e}")
        return {"preferences": [], "episodes": [], "insights": []}


# Header
st.title("🛍️ Store Associate Assistant")
st.markdown("### AI-Powered Customer Service with Long-Term Memory")

# Sidebar - Customer Selection
st.sidebar.header("Customer Selection")

customers = load_customers()

if customers:
    customer_options = {f"{c['name']} ({c['customer_id']})": c['customer_id'] 
                       for c in customers}
    
    selected = st.sidebar.selectbox(
        "Select Customer",
        options=list(customer_options.keys())
    )
    
    if selected:
        customer_id = customer_options[selected]
        
        # Update current customer if changed
        if st.session_state.current_customer != customer_id:
            st.session_state.current_customer = customer_id
            st.session_state.messages = []
            st.session_state.thread_id = f"thread_{customer_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Switched to customer: {customer_id}")
else:
    st.sidebar.warning("No customers found. Please seed the database first.")
    st.stop()

# Sidebar - Session Controls
st.sidebar.header("Session Controls")

if st.sidebar.button("🔄 New Session"):
    st.session_state.messages = []
    st.session_state.thread_id = f"thread_{st.session_state.current_customer}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.rerun()

if st.sidebar.button("🛑 End Session"):
    if st.session_state.current_customer and st.session_state.messages:
        # End the session (triggers episode creation)
        st.session_state.agent.end_session(
            customer_id=st.session_state.current_customer,
            thread_id=st.session_state.thread_id
        )
        st.sidebar.success("Session ended. Episode summary created!")
        logger.info(f"Session ended for {st.session_state.current_customer}")

# Sidebar - Memory View
st.sidebar.header("Customer Memories")

if st.session_state.current_customer:
    memories = load_customer_memories(st.session_state.current_customer)
    
    # Show memory counts
    st.sidebar.metric("Preferences", len(memories["preferences"]))
    st.sidebar.metric("Past Episodes", len(memories["episodes"]))
    st.sidebar.metric("Insights", len(memories["insights"]))
    
    # Expandable sections for memory details
    with st.sidebar.expander("View Preferences"):
        if memories["preferences"]:
            for pref in memories["preferences"][:5]:
                st.write(f"**{pref.value.get('preference_type', 'N/A')}**: {pref.value.get('value', 'N/A')}")
        else:
            st.info("No preferences yet")
    
    with st.sidebar.expander("View Episodes"):
        if memories["episodes"]:
            for ep in memories["episodes"][:3]:
                st.write(f"📅 {ep.value.get('date', 'N/A')}")
                st.write(ep.value.get('summary', 'N/A'))
                st.write("---")
        else:
            st.info("No episodes yet")
    
    with st.sidebar.expander("View Insights"):
        if memories["insights"]:
            for insight in memories["insights"][:3]:
                st.write(f"💡 {insight.value.get('pattern', 'N/A')}")
                st.write(f"Confidence: {insight.value.get('confidence', 0):.2f}")
                st.write("---")
        else:
            st.info("No insights yet")

# Main Chat Interface
st.header("💬 Conversation")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    if not st.session_state.current_customer:
        st.error("Please select a customer first")
        st.stop()
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent.chat(
                    customer_id=st.session_state.current_customer,
                    message=prompt,
                    thread_id=st.session_state.thread_id,
                    session_active=True
                )
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                logger.error(f"Error in chat: {e}")

