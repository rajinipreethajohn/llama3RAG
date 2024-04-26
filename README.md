

# Creating a RAG system using FLASK and CHROMA DB:

## This Flask application provides an API for performing document search and answering questions using a language model.  

## To get started with this application, follow the steps below: 
  
1.  Make sure you have Python and pip installed on your system.  

2.  Navigate to the project directory.
    
3.  Install the required Python packages
    

### Usage

1.  Run the Flask application:
  
2.  The application will start running on `http://127.0.0.1:5000/` by default.
    
3.  You can now use the following endpoints:
    
    *   **POST /ai**: Send a question to get an LLM-generated response.
    *   **POST /askPDF**: Query the uploaded PDF documents.
    *   **POST /pdf**: Upload PDF files to the server.
    
    See the `app.py` file for more details on each endpoint.
    

Endpoints
---------

### POST /ai

Send a POST request to `/ai` with JSON data containing the question you want to ask:

json


### POST /askPDF

Send a POST request to `/askPDF` with JSON data containing the query you want to search for in the PDF documents:

json


### POST /pdf

Send a POST request to `/pdf` with a PDF file attached to the request:

python


`import requests  url = 'http://127.0.0.1:5000/pdf' files = {'file': open('path_to_your_pdf_file.pdf', 'rb')}  response = requests.post(url, files=files)  print(response.json())`

Acknowledgments
---------------

*   This application uses [Flask](https://flask.palletsprojects.com/) for building the web server (POSTMAN used to spin-up FLASK)
*   The language model and document processing utilities are from the [langchain](https://github.com/username/langchain) library.
*   The llm used here is llama3 which is locally installed
