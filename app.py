import json
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from function import init_state, reset_state, change_on_api_key, change_on_lan, load_css, chunking_document

# 1. Page Configuration
st.set_page_config(
    page_title="Chat with Vio", 
    page_icon="👋",
    layout="centered"
)

# 2. Custom CSS with Media Queries (The Secret to Responsiveness)
load_css()

# 3. Render the Content
# Display the Responsive Title
st.markdown('<h1 class="main-title">👋 Hey, I\'m Vio\'s AI Assistant!</h1>', unsafe_allow_html=True)

# Display the Responsive Card
st.markdown(
    """
    <div class="description-box">
        <b>Want to get to know the builder?</b> You're in the right place! <br><br>
        I'm here to answer <span class="highlight">everything</span> you want to know about Silvio.
        From his complex code to his coffee preferences—<span class="highlight">just ask!</span>
    </div>
    """, 
    unsafe_allow_html=True
)

init_state()

with st.sidebar:
    # --- 1. Configuration Section ---
    st.header("⚙️ System Configuration")

    # API Key Input
    # 'on_change' triggers a callback to potentially clear old sessions when the key changes
    st.text_input(
        "🔑 Google Gemini API Key", 
        type="password",
        key="google_api_key",
        on_change=change_on_api_key,
        help="Paste your Google AI Studio API key here. It is required to power the AI models."
    )

    # Visual separator for better UI
    st.divider()

    # --- 2. Action Buttons ---
    # Reset Button
    # Capitalized the label and added an icon for better UX
    st.button(
        "🔄 Reset Conversation",
        on_click=reset_state,
        use_container_width=True, # Makes the button span the full width of the sidebar
        help="Clear the chat history and start a new session."
    )

    # Load Information Button
    # Changed label to 'Load Knowledge Base' to sound more professional
    load_vec_store = st.button(
        "📂 Load Knowledge Base",
        use_container_width=True,
        help="Initialize the vector database with Silvio's profile data."
    )

    st.divider()

    # Language Selection Dropdown
    # IMPORTANT: The 'key' parameter binds this widget to st.session_state.selected_language
    # This allows the AI Chain to access the chosen language globally.
    chosen_language = st.selectbox(
        "🌐 Response Language",
        options=["English", "Indonesian"],
        index=0,
        on_change=change_on_lan,
        key="selected_language", # <--- CRITICAL FIX: Ensures language choice is saved in memory
        help="Select the language you want the AI to use when answering your questions."
    )

if load_vec_store and st.session_state.vectorstore is None:
    # Check if the API Key is missing
    if not st.session_state.google_api_key:
        st.error("⚠️ API Key is missing. Please input your key in the **'🔑 Google Gemini API Key'** field in the sidebar.")

    else:
        try:
            with st.spinner("Processing documents and generating embeddings..."):
                # 1. Split document into chunks
                text_chunks = chunking_document()

                # 2. Initialize Embedding Model
                # FIXED: Updated model name for stability
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=st.session_state.google_api_key
                )

                # 3. Create Vector Store
                st.session_state.vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)

                # Notify success
                st.toast("✅ **Knowledge Base Loaded Successfully! System is ready to use.**")
                
        except Exception as e:
            error_msg = str(e)
            answer = "" # Placeholder for the error message

            # 1. Handle API Quota Limits (Common during embedding large docs)
            if "429" in error_msg or "Quota exceeded" in error_msg:
                answer = "🚨 **Embedding Quota Exceeded**\n\nFailed to process documents because the Google Gemini API limit has been reached. Please wait a minute before trying again."
            
            # 2. Handle Safety/Content Filters (If the document text violates policies)
            elif "finish_reason" in error_msg and "SAFETY" in error_msg:
                 answer = "🛡️ **Content Safety Block**\n\nThe profile data triggered Google's safety filters. The system cannot generate embeddings for this content."

            # 3. Handle Invalid API Key (Authentication failed during embedding)
            elif "API key not valid" in error_msg:
                 answer = "🔑 **Invalid API Key**\n\nThe API Key was rejected by the embedding service. Please check the **'🔑 Google Gemini API Key'** settings in the sidebar."

            # 4. Handle General Connection/Auth Issues (Replaced "Search Tool" error)
            elif "ratelimit" in error_msg.lower() or "auth" in error_msg.lower():
                 answer = "🌐 **Connection Issue**\n\nFailed to connect to Google API services for embedding. Please check your internet connection or API permissions."

            # 5. Handle General/Unknown Errors
            else:
                # Context-specific error message
                answer = f"❌ **Failed to Load Knowledge Base:**\n\nAn error occurred while creating the vector database. \n\n**Details:** `{error_msg}`"

            # Finally, display the formatted error message
            st.error(answer)

elif st.session_state.vectorstore is None:
    # Default state: Show instructions when idle or if configuration is incomplete
    st.info("""
        **⚠️ Configuration Required**
        
        To proceed, please ensure the following:
        1. Enter your API Key in the **'🔑 Google Gemini API Key'** sidebar input.
        2. Click the **'📂 Load Knowledge Base'** button to initialize the system.
    """)

elif load_vec_store and st.session_state.vectorstore is not None:
    # Feedback for redundant action: User clicked load again, but data is already present
    st.toast("Knowledge Base is already loaded. No need to reload.", icon="ℹ️")

# Check if the QA Chain needs to be initialized (or re-initialized due to language change)
# AND ensure the Vector Store contains data before proceeding.
if st.session_state.qa_chain is None \
    and st.session_state.vectorstore is not None:

    try:
        # 1. Setup Retriever
        retriever = st.session_state.vectorstore.as_retriever()

        # 2. Initialize LLM (Gemini)
        # FIXED: Updated model name to 1.5 Flash (More stable)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=st.session_state.google_api_key,
            temperature=0.3
        )

        # 3. Define the Dynamic Prompt Template
        # Uses f-string to inject the 'chosen_language' variable
        # Uses double braces {{ }} for LangChain variables (context, question)
        prompt_template = f"""
        You are the AI Portfolio Assistant for **Silvio Christian Joe**, also known by his nickname **Vio**.
        Your role is to represent Vio professionally to recruiters, developers, and visitors.

        Use the following pieces of context to answer the user's question.

        ### INSTRUCTIONS:
        1. **LANGUAGE PRIORITY:** The user is speaking in **{chosen_language}**. You MUST answer strictly in **{chosen_language}**. If the context is in English, translate your understanding and output to **{chosen_language}** naturally.
        2. **ELABORATE & ENGAGE:** Do not give short, one-line answers. Be detailed, descriptive, and professional. Explain *how* Vio uses his skills, don't just list them.
        3. **SPECIALIZATION:** Always frame answers to highlight his strong expertise in **NLP (Natural Language Processing), Data Science, and Tabular Data**.
        4. **NO HALLUCINATIONS:** If the answer is not in the context, DO NOT make it up. Instead, output this exact sentence (translated to **{chosen_language}**):
            "I don't have that specific information in my database currently. However, you can contact Silvio directly via LinkedIn (https://www.linkedin.com/in/silvio-christian-joe/) or Email (viochristian12@gmail.com)."

        Context:
        {{context}}

        Question: {{question}}

        Detailed Answer (in {chosen_language}):
        """
        
        QA_PROMPT = PromptTemplate.from_template(prompt_template)

        # 4. Build the Conversational Chain
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True, 
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )
        
    except Exception as e:
        error_msg = str(e)
        answer = "" # Placeholder for the error message

        # 1. Handle API Quota Limits (Most Common on free tier)
        if "429" in error_msg or "Quota exceeded" in error_msg:
            answer = "🚨 **API Quota Exceeded**\n\nFailed to initialize the AI model because the Google Gemini API limit has been reached. Please wait a minute before trying again."
        
        # 2. Handle Safety/Content Filters (Rare during init, but possible)
        elif "finish_reason" in error_msg and "SAFETY" in error_msg:
                answer = "🛡️ **Safety Restriction**\n\nThe AI model could not be initialized due to Google's safety filters."

        # 3. Handle Invalid API Key
        elif "API key not valid" in error_msg:
                answer = "🔑 **Invalid API Key**\n\nThe API Key provided is incorrect. Please check the **'🔑 Google Gemini API Key'** settings in the sidebar."

        # 4. Handle General/Unknown Errors
        else:
            # Clean up the error message to be less scary if possible
            answer = f"❌ **Failed to initialize AI Chain:**\n\nAn error occurred while setting up the conversation model. \n\n**Details:** `{error_msg}`"

        # Finally, display the formatted error message
        st.error(answer)

# Fallback: If Vectorstore is missing, warn the user
elif st.session_state.vectorstore is None:
    st.error("⚠️ Knowledge Base is not loaded. Please click the **'📂 Load Knowledge Base'** button in the sidebar to start.")

# 1. Display Chat History
# Renders previous messages from the session state to the screen
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 2. Handle User Input
# The prompt variable captures the user's text input
if prompt := st.chat_input("Ask about Vio's skills, projects, or experience..."):
    
    # Validation 1: Check if documents are processed (Vector Database)
    # Actionable Error: Directs user to the 'Load' button
    if not st.session_state.vectorstore:
        st.error("⚠️ Knowledge Base is not loaded. Please click the **'📂 Load Knowledge Base'** button in the sidebar.")
    
    # Validation 2: Check if the AI Agent is initialized
    # Actionable Error: Suggests checking API Key or Resetting
    elif not st.session_state.qa_chain:
        st.error("⚠️ AI Agent is not initialized. Please ensure your API Key is correct or try clicking **'🔄 Reset Conversation'**.")

    else: 
        # Display User Message immediately and add to session state
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.chat_message("human").write(prompt)

        # Generate AI Response
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                try:
                    # Execute the Chain
                    # This sends the question + history to Gemini via LangChain
                    response = st.session_state.qa_chain.invoke({
                        "question": prompt, 
                        "chat_history": st.session_state.chat_history
                    })

                    answer = response["answer"]
                    source_docs = response["source_documents"]

                    # Display the AI's Answer
                    st.markdown(answer)

                    # Display Evidence/Sources in an expandable section
                    # Renamed from 'video evidence' to 'Reference Context' for accuracy
                    if source_docs:
                        with st.expander("🔍 View Reference Context (Debug Info)"):
                            for i, doc in enumerate(source_docs, 1):
                                # Safe get for metadata source
                                information_source = doc.metadata.get("source", "data/silvio_profile.json")
                                
                                st.markdown(f"**Evidence {i}**")
                                st.caption(f"Source: `{information_source}`")
                                # formatting as json code block for readability
                                st.code(doc.page_content, language="json")
                                st.divider()

                    # Save AI Message to History
                    st.session_state.messages.append({"role": "ai", "content": answer})
                    st.session_state.chat_history.append((prompt, answer))
                
                except Exception as e:
                    error_msg = str(e)
                    answer = "" # Placeholder for the error message

                    # 1. Handle API Quota Limits
                    if "429" in error_msg or "Quota exceeded" in error_msg:
                        answer = "🚨 **API Quota Exceeded**\n\nI cannot answer right now because the Google Gemini API limit has been reached. Please wait a minute before trying again."
                    
                    # 2. Handle Safety/Content Filters
                    elif "finish_reason" in error_msg and "SAFETY" in error_msg:
                            answer = "🛡️ **Safety Restriction**\n\nI cannot answer this question because it triggered Google's safety filters. Please rephrase your question."

                    # 3. Handle Invalid API Key
                    elif "API key not valid" in error_msg:
                            answer = "🔑 **Invalid API Key**\n\nThe API Key provided is incorrect. Please check the **'🔑 Google Gemini API Key'** settings in the sidebar."

                    # 4. Handle Web Search Tool Errors
                    elif "ratelimit" in error_msg.lower() or "auth" in error_msg.lower():
                            answer = "🌐 **Search Tool Issue**\n\nThe web search tool is currently unavailable. I will try to answer based ONLY on the documents."

                    # 5. Handle General/Unknown Errors
                    else:
                        answer = f"❌ **An error occurred:**\n\nI encountered an issue while generating the response. \n\n**Details:** `{error_msg}`"

                    # Display the specific error message to the user
                    st.error(answer)

with st.sidebar:
    # Export Chat History Section
    # Only enable the download button if there is an active conversation to save.
    if st.session_state.messages:
        # Convert chat history list to a JSON string for downloading
        chat_str = json.dumps(st.session_state.messages, indent=2)
        
        downloaded = st.download_button(
            label="Download Chat History",
            data=chat_str,
            file_name="vio_chat_history.json", # Updated filename to be more relevant
            mime="application/json",
            icon="📥",
            help="Save your conversation and research insights as a JSON file."
        )
        
        # Provide visual feedback when the download is triggered
        if downloaded:
            st.toast("Chat history downloaded successfully!", icon="✅")
            
    else:
        # Show a disabled button when there is no history, for better UI consistency
        st.button(
            label="Download Chat History",
            icon="📥",
            disabled=True,
            help="Start a conversation to enable history downloading."
        )
    
    # Visual separator
    st.divider()

    # --- 3. Help & Documentation ---
    # User Guide Expander
    # Added markdown content to guide the user
    with st.expander("📖 How To Use"):
        st.markdown("""
        1. **Enter API Key**: Input your Gemini API Key at the top.
        2. **Load Data**: Click the **'Load Knowledge Base'** button to initialize Vio's memory.
        3. **Chat**: Ask anything about Silvio's skills, projects, or experience!
        """)

    # FAQ Expander
    # Added placeholder Q&A for common user questions
    with st.expander("❓ FAQ"):
        st.markdown("""
        **Q: Is my API Key saved?** A: No, it is only used for this browser session.
        
        **Q: What can I ask?** A: You can ask about Silvio's Tech Stack, GitHub projects, or contact info.
        """)

    st.markdown("---") 
    st.markdown(
        """
        <div style="text-align: center; font-size: 0.85rem; color: #888;">
            © 2026 <b>Silvio Christian, Joe</b><br>
            Powered by <b>Google Gemini</b> 🚀<br><br>
            <a href="https://www.linkedin.com/in/silvio-christian-joe/" target="_blank" style="text-decoration: none; margin-right: 10px;">🔗 LinkedIn</a>
            <a href="mailto:viochristian12@gmail.com" style="text-decoration: none;">📧 Email</a>
        </div>
        """, 
        unsafe_allow_html=True
    )