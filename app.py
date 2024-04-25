from langchain_community.llms import Ollama
from flask import Flask, request, jsonify

app = Flask(__name__)

llm = Ollama(model="llama3")

@app.route('/ai', methods=['POST'])
def ai():
    data = request.get_json(force=True)
    question = data.get("question")  # Get the question from the request
    response = llm.invoke(question)  # Generate response from the model

    print(response)
    response_answer = {"answer": response}
    return (response_answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
