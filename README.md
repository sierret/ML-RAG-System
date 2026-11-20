This is an AI Information System that uses Retrieval Augmented System(RAG) design to provide context to a an LLM. A PDF manual is loaded and chunked. The chunks are converted to embeddings and stored in a vector Chroma-based DB. Upon user query the request is embedded and matching embeddings are ranked by similarity. The retrieved chunks are used as context for ChatGroq LLM "openai/gpt-oss-safeguard-20b".

Uses:

langchain
ChatGroq(langchain-groq)
HuggingFace Embeddings(langchain_huggingface)

Requires a ChatGroq API key. The key is exported (using EXPORT CHATGROQ_API_KEY="your_key_here") and read at runtime.
