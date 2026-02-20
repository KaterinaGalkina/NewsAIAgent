from datasets import load_dataset
from transformers import AutoTokenizer
from langchain.tools import tool
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline
from langchain import HuggingFacePipeline

# We use the tech news texts as the knowledge base
# Loading small subset of Sci/Tech news for RAG
dataset = load_dataset("ag_news", split="train[:1000]") # only 1000 examples for speed
tech_dataset = [item for item in dataset if item["label"] == 3] # label 3 = Sci/Tech
texts = [item["text"] for item in tech_dataset]

# Model for retrieval
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = FAISS.from_texts(texts, embeddings)

model_name = "tiiuae/falcon-7b-instruct"

# Loading tokenizer from the pretrained model
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Wrapping the fine-tuned model in pipeline
pipe = pipeline("text-generation", model=model_name, tokenizer=tokenizer, max_length=256)
llm = HuggingFacePipeline(pipeline=pipe)

# Search tool
def search_news(query: str) -> str:
    results = vectordb.similarity_search(query, k=3)
    return "\n".join([r.page_content for r in results])

search_tool = Tool(
    name="Search News",
    func=search_news,
    description="Search top 3 tech news snippets."
)

# Agent
agent = initialize_agent(
    tools=[search_tool],
    llm=llm,
    verbose=True
)

# Run the agent
query = "Tell me the latest AI news please."
response = agent.run(query)
print(response)