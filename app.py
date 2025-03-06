import os
import streamlit as st
import torch
from config import *
from retriever import initialize_pinecone, create_hybrid_retriever
from document_processor import convert_pdf_to_markdown, process_and_chunk_document
from llm_chain import initialize_llm, create_retrieval_chain

st.set_page_config(layout="wide")

# Simplified CSS for social icons
st.markdown("""
<style>
.social-icons {
    display: flex;
    gap: 15px;
    justify-content: center;
    margin: 20px 0;
}
.social-icons a {
    font-size: 24px;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)

st.title("📚 Ask PDF - Smart Document Analysis")

# Social Sharing Sidebar (Icons Only)
with st.sidebar:
    st.header("Share the Knowledge!")
    st.markdown('<div class="share-box">', unsafe_allow_html=True)
    st.write("🌟 Love using Ask PDF? Share with your network!")
    st.markdown('<div class="social-icons">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.link_button("🐦", "https://twitter.com")
    with col2:
        st.link_button("🔗", "https://www.linkedin.com/shareArticle?mini=true&url=https://your-app-url.com&title=Ask%20PDF%20AI%20Analyzer")
    with col3:
        st.link_button("📧", "mailto:?subject=Check%20out%20Ask%20PDF&body=I%20found%20this%20great%20PDF%20analysis%20tool%20you%20might%20like%20👉%20https://your-app-url.com")
    st.markdown('</div>', unsafe_allow_html=True)

    # Testimonials
    st.subheader("💬 What Users Say")
    st.markdown('<div class="testimonial">"This app saved me hours of research time! The AI answers are surprisingly accurate."<br>- Sarah, Researcher</div>', unsafe_allow_html=True)
    st.markdown('<div class="testimonial">"Perfect for quick document analysis - my new go-to tool!"<br>- Mark, Project Manager</div>', unsafe_allow_html=True)

# Main Content
@st.cache_resource
def setup_resources():
    index = initialize_pinecone(PINECONE_API_KEY, INDEX_NAME)
    retriever, embeddings = create_hybrid_retriever(index, MODEL_NAME)
    llm = initialize_llm(LLM_REPO_ID)
    return retriever, embeddings, llm

retriever, embeddings, llm = setup_resources()
rag_chain = create_retrieval_chain(retriever, llm)

# File Upload Handling
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

    # Interactive Q&A
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
