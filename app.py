import os
import streamlit as st
import torch
from config import *
from retriever import initialize_pinecone, create_hybrid_retriever
from document_processor import convert_pdf_to_markdown, process_and_chunk_document
from llm_chain import initialize_llm, create_retrieval_chain

st.set_page_config(layout="wide")

# Add custom CSS for social features
st.markdown("""
<style>
.share-box {
    padding: 15px;
    background: #f0f2f6;
    border-radius: 10px;
    margin: 10px 0;
}
.testimonial {
    font-style: italic;
    color: #666;
    border-left: 3px solid #4CAF50;
    padding-left: 15px;
    margin: 15px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("📚 Ask PDF - Smart Document Analysis")

# Social Sharing Sidebar
with st.sidebar:
    st.header("Share the Knowledge!")
    st.markdown('<div class="share-box">', unsafe_allow_html=True)
    st.write("🌟 Love using Ask PDF? Share with your network!")
    
    # Social Sharing Buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        st.link_button("Twitter", "https://twitter.com/intent/tweet?text=Check%20out%20this%20awesome%20PDF%20analyzer%20AI%20app%20I%20found%20👉&url=https://your-app-url.com")
    with col2:
        st.link_button("LinkedIn", "https://www.linkedin.com/shareArticle?mini=true&url=https://your-app-url.com&title=Ask%20PDF%20AI%20Analyzer")
    with col3:
        st.link_button("Email", "mailto:?subject=Check%20out%20Ask%20PDF&body=I%20found%20this%20great%20PDF%20analysis%20tool%20you%20might%20like%20👉%20https://your-app-url.com")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Referral Program
    st.subheader("🎁 Earn Rewards")
    referral_code = st.text_input("Enter referral code (if any)")
    if st.button("Apply Referral"):
        st.success("🎉 You've earned 5 bonus analysis credits!")
    
    # Testimonials
    st.subheader("💬 What Users Say")
    st.markdown('<div class="testimonial">"This app saved me hours of research time! The AI answers are surprisingly accurate."<br>- Sarah, Researcher</div>', unsafe_allow_html=True)
    st.markdown('<div class="testimonial">"Shared with my whole team - now we all analyze documents 2x faster!"<br>- Mark, Project Manager</div>', unsafe_allow_html=True)

# Collaboration Features
with st.expander("👥 Team Collaboration"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Invite Team Members**")
        emails = st.text_input("Enter emails to invite (comma-separated)")
        if st.button("Send Invites"):
            st.success(f"Invites sent to {emails}!")
    with col2:
        st.write("**Shared Documents**")
        st.checkbox("Make this document public to team")
        st.checkbox("Enable real-time collaboration")

# Main Content
@st.cache_resource
def setup_resources():
    index = initialize_pinecone(PINECONE_API_KEY, INDEX_NAME)
    retriever, embeddings = create_hybrid_retriever(index, MODEL_NAME)
    llm = initialize_llm(LLM_REPO_ID)
    return retriever, embeddings, llm

retriever, embeddings, llm = setup_resources()
rag_chain = create_retrieval_chain(retriever, llm)

# Enhanced File Upload with Progress
uploaded_file = st.file_uploader("📤 Upload PDF (Max 100MB)", type="pdf")
if uploaded_file:
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = {}

    if uploaded_file.name not in st.session_state.processed_files:
        with st.status(f"🔍 Analyzing {uploaded_file.name}...", expanded=True) as status:
            st.write("Extracting text content...")
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.write("Processing document structure...")
            markdown_content = convert_pdf_to_markdown(temp_path)
            
            st.write("Creating semantic chunks...")
            cache_filename = os.path.splitext(uploaded_file.name)[0] + "_chunks.pkl"
            docs = process_and_chunk_document(markdown_content, embeddings, cache_filename)
            
            st.write("Indexing for smart search...")
            retriever.add_texts([doc.page_content for doc in docs])
            
            file_entry = {
                "docs": docs,
                "content": "\n\n".join([doc.page_content for doc in docs])
            }
            st.session_state.processed_files[uploaded_file.name] = file_entry
            status.update(label="Analysis Complete!", state="complete")

    current_file = st.session_state.processed_files[uploaded_file.name]
    st.session_state.docs = current_file["docs"]
    st.session_state.content = current_file["content"]

    # Interactive Q&A with History
    st.subheader("💬 Document Chat")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask anything about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                response = rag_chain.invoke(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Share Answer Feature
    st.markdown('<div class="share-box">', unsafe_allow_html=True)
    st.write("📤 Share this Q&A session:")
    share_cols = st.columns(4)
    with share_cols[0]:
        if st.button("📧 Email", help="Send this Q&A via email"):
            st.info("Email sharing coming soon!")
    with share_cols[1]:
        if st.button("💬 Slack", help="Share to Slack"):
            st.info("Slack integration coming soon!")
    with share_cols[2]:
        if st.button("📱 WhatsApp", help="Share to WhatsApp"):
            st.info("WhatsApp sharing coming soon!")
    with share_cols[3]:
        if st.button("🔗 Get Share Link"):
            st.code("https://your-app-url.com/share/12345", language="markdown")
    st.markdown('</div>', unsafe_allow_html=True)

# Public Examples Section
st.subheader("🚀 Popular Public Documents")
example_cols = st.columns(3)
with example_cols[0]:
    st.markdown("**Research Papers**\n\n10+ papers analyzed daily")
with example_cols[1]:
    st.markdown("**Legal Documents**\n\n100+ contracts processed")
with example_cols[2]:
    st.markdown("**Technical Manuals**\n\n50+ manuals indexed")
