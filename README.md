<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ul>
    <li><a href="#about-the-project">About the Project</a></li>
    <li><a href="#tools-used">Tools Used</a></li>
    <li><a href="#set-up-instruction">Set up instruction</a></li>
    <li><a href="#prepare-the-knowledge-base">Prepare the knowledge base</a></li>
    <li><a href="#run-the-application">Run the app</a></li>
  </ul>
</details>
<!-- END OF TABLE OF CONTENTS -->

<!-- ABOUT THE PROJECT -->
## About the Project
This is a practical implementation of RAG application from a production ready standpoint. 
It contains all the tools and frameworks that are required to make a RAG application more optimised and secure. 
You can provide private knowledge base as documents and the LLM will answer based on the private knowledge.

### Tools Used
List of tools/languages the product uses
- [x] Python
- [x] Ollama
- [x] LlamaIndex
- [x] ChromaDB
- [x] Langfuse


<!-- END OF ABOUT THE PROJECT -->


<!-- SET UP INSTRUCTION -->
## Set up instruction
Follow the steps below to run the application

### Install and run Ollama

1. Download Ollama from https://ollama.com and install following the instructions
2. Follow instructions to enable ollama cli command
3. Run the following command to pull llama2 7b locally
``` bash
ollama pull llama2
```
4. Once the model is pulled, run the following command to start running ollama service
``` bash
ollama serve
```
⚠️ This will start the application in default port. If you see an error that means Ollama is already running  

## Install and run with python virtual environment

1. Set up a virtual environment using the command 
``` bash
python3 -m venv venv
```
2. Activate the virtual environment using the command 
``` bash
source venv/bin/activate
```
3. Change your IDE settings accordingly to use the created virtual environment
4. Install the required dependencies using the command
``` bash 
pip install -r requirements.txt
```

### Set up the environment variables
1. Create a .env file in the root of the project directory
2. Refer the .env.example file for the environment variables to be set in the .env

### Configure the observability using langfuse
1. Follow the below steps to configure langfuse
``` bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up
```
2. Once it is up and running you can access the observability at `http://localhost:3000`
3. Create a new user id and password to sign in
4. Create an api key from settings and copy the secret key and public key in .env file as shown:
    `LANGFUSE_SECRET_KEY=sk-<secret_key>`
    `LANGFUSE_PUBLIC_KEY=pk-<public_key>`
    `LANGFUSE_HOST=http://127.0.0.1:3000`
5. Create a prompt (sample given under samples) from the UI and reference the prompt name in env variable under `PROMPT_TEMPLATE` variable name.
<!-- END OF SET UP INSTRUCTION -->

<!-- PREPARE KNOWLEDGE BASE -->
## Prepare the knowledge base
1. To prepare the knowledge base create a docs folder in the root of the project directory
2. Add pdf documents under the docs folder
<!-- END OF PREPARE KNOWLEDGE BASE -->

<!-- RUN THE APP -->
## Run the application
To run this from a terminal or command prompt, run the following command
``` bash
python app.py
```
<!-- END OF RUN THE APP -->
