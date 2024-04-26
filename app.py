from langchain_community.llms import Ollama
from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader


app = Flask(__name__)

folder_path = "db"

llm = Ollama(model="llama3")

embeddings = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False)

# GETTING INFO ABOUT OUR QUERY
@app.route('/ai', methods=['POST'])
def ai():
    data = request.get_json(force=True)
    question = data.get("question")  # Get the question from the request
    response = llm.invoke(question)  # Generate response from the model

    print(response)
    response_answer = {"answer": response}
    return (response_answer)


@app.route('/ai', methods=['POST'])
def ai():
    data = request.get_json(force=True)
    question = data.get("question")  # Get the question from the request
    response = llm.invoke(question)  # Generate response from the model

    print(response)
    response_answer = {"answer": response}
    return (response_answer)


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

    vecotor_store = Chroma.from_documents(documents=chunks, embedding=embeddings) 

    vecotor_store.persist()

    response = {"Status": "Successfully uploaded", "filename": file_name, "docs length": len(documents), "chunks": len(chunks)}
    return (response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
