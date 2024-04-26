from langchain_community.llms import Ollama
from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

app = Flask(__name__)

folder_path = "db"

llm = Ollama(model="llama3")

embeddings = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200 )

score_threshold = 0.1

raw_prompt = PromptTemplate.from_template(""" 
    <s>[INST] You are a technical assistant good at searching docuemnts. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
""")

# GETTING INFO ABOUT OUR QUERY FROM INVOKING LLM
@app.route('/ai', methods=['POST'])
def ai():
    data = request.get_json(force=True)
    question = data.get("question")  # Get the question from the request
    response = llm.invoke(question)  # Generate response from the model

    print(response)
    response_answer = {"answer": response}
    return (response_answer)

# QUERYING OUR UPLOADED PDF DOC
@app.route('/askPDF', methods=['POST'])
def ask_pdf():
    data = request.get_json(force=True)
    query = data.get("query")  # Get the question from the request
    
    print(f"PDF_query: {query}")
   
    print("Loading Vector store")
    vecotor_store = Chroma(persist_directory=folder_path, embedding_function=embeddings
    )
    vecotor_store.persist()

    print("Creating chain")
    retriever = vecotor_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3})

    document_chain = create_stuff_documents_chain( llm,raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    result = chain.invoke({"input": query})
    print(result)

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )
    pdf_response = {"answer": result["answer"],"sources": sources}
    return (pdf_response)


# UPLOAD PDF AND SPLIT DOC INTO CHUNKS & STORE IN CHROMADB
@app.route('/pdf', methods=['POST']) #Access the pdf file and return the contents
def pdf():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename:{file_name}")

    loader = PyPDFLoader(save_file)
    documents = loader.load_and_split()
    print(f"length of documents = {len(documents)}")

    chunks = text_splitter.split_documents(documents)
    print(f"length of chunks = {len(chunks)}")

    vecotor_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=folder_path) 

    vecotor_store.persist()

    response = {"Status": "Successfully uploaded", 
                "filename": file_name, 
                "docs length": len(documents), 
                "chunks": len(chunks)}
    return (response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
