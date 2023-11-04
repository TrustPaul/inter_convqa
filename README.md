# Configuration
You need Huggingface API for Opensource Language models and OpenAI API for Runnning Huggingface Models <br>
We are using Huggiingface Pro Subscription can be obtained from [Huggingface](https://huggingface.co/blog/inference-pro) <br>
Open API keys can be obtained from OpenAI website [OpenAI](https://openai.com/) <br>
## Huggingface API
REplace this statement at the top of each of the files listed with your Huggingface API
YOUR_TOKEN = 'Replace with your huggingface token'
- Basic_chatbot.py
- chat_with_vector_database.py
- chat_with_your_documents.py
- chatbot_with_internet_access.py
- utils.py

## OpenAI API
Replace this statement with your OpenAI api in <br>
OPENAI_API_KEY = "Replace with your OpenAI API"
- tool_agumented_with_chat_gpt.py



# Installations
## Create a virtual environment
This has been tested in Ubuntu or Windows WSL
## Create a virtual environment
````
sudo apt-get install python3-pip 
sudo pip3 install virtualenv
````

<br>

````
virtualenv venv
source venv/bin/activate
````

## Install the necessary packages
````
pip install -r requirements.txt 
````

# Embedding Documents for the Vector Database
If you have multiple documents that you would like to  embed in a vector database and chat with it <br>
YOu have to create a vectordatabase first <br>
We are using an opensource chroma database <br>

## Working with PDF Documents
Store all your pdf documents in your a folder called DOC in the same directory as this code and at the same level as embed_pdf.py
In your virtual environment, Run
````
python embed_pdf.py
````

## Working with CSV or Excel
Load in your data in the file embed_csv.py
Adjust the file name or column name accordingly
And run
````
python embed_csv.py
````
# Running the Program
Once everything is set, You can run the program using the command
````
streamlit run Home.py
````
