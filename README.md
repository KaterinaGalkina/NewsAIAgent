# Mini Tech News AI Agent

This is a tiny project that demonstrate fine-tuning a Falcon model on Sci/Tech news and building a retrieval-augmented AI agent.  

---

## Files

### `fine-tuning.py`
- Loads a small subset of Sci/Tech news from the `ag_news` dataset.
- Uses Falcon-7B-Instruct as the base model.
- Tokenizes the dataset for causal language modeling.
- Applies LoRA (parameter-efficient fine-tuning) to train the model quickly on a local machine.
- Saves the fine-tuned model to `./ft_model`.

Key points:
- Uses PyTorch + Hugging Face `Trainer`.
- Training is lightweight (`max_length=64`, batch size=1, 1 epoch) for quick experimentation.
- Fine-tunes only the `query_key_value` layers using LoRA.

---

### `aiagent.py`
- Loads a small subset of Sci/Tech news for retrieval.
- Builds a **FAISS** vector store for semantic search.
- Wraps the fine-tuned Falcon model in a Hugging Face pipeline.
- Creates a **LangChain agent** with a tool that searches top 3 tech news snippets.
- Runs a query through the agent and prints a response.

Key points:
- Uses LangChain and `langchain_community` modules for vector search and embeddings.
- HuggingFacePipeline + FAISS allows retrieval-augmented generation (RAG) for relevant news answers.
- Example query: `"Tell me the latest AI news please."`

---

## Usage

1. Fine-tune the model (optional for testing):
```bash
python fine-tuning.py
```

2. Run the AI agent:
```bash
python aiagent.py
```

You should get a short, humorous response with the latest AI/tech news snippets.
