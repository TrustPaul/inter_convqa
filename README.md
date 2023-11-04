# Configuration
You need Huggingface API for Opensource Language models and OpenAI API for Runnning Huggingface Models <br>
We are using Huggiingface Pro Subscription can be obtained from [Huggingface](https://huggingface.co/blog/inference-pro) <br>
Open API keys can be obtained from OpenAI website [OpenAI](https://openai.com/) <br>

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

#Embedding Documents for the Vector Database
If you have multiple documents that you would like to  embed in a vector database and chat with it <br>
YOu have to create a vectordatabase first <br>
We are using an opensource chroma database <br>

## Working with PDF Documents
In your 

## Working with CSV or Excel
