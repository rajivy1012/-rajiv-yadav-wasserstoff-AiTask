# **AI Website Chatbot**

This is a Streamlit-based application that allows users to interact with the content of any website through a conversational chatbot interface.

## Features

- Fetch and extract text content from a given website URL
- Split the content into manageable chunks for processing
- Create a FAISS vector store from the text chunks using Google's Generative AI Embeddings
- Provide a conversational interface for users to ask questions about the website content
- Use a LangChain-based QA chain with Google's Gemini language model to generate answers

## Flowchart
![{66A008FA-E573-497D-99C1-5FB1EE978E9C}](https://github.com/user-attachments/assets/7f3e351b-bf68-43d6-94a8-353d706e607e)

## Output

1.![{7E80129B-8321-477D-9186-1FF50EA4966F}](https://github.com/user-attachments/assets/7f466607-cc16-49a1-abbb-304b1b6a48fd)
2. ![{31D69F13-3E67-4687-B750-AA53B3A7C0F6}](https://github.com/user-attachments/assets/ee997d26-a7fe-4c6b-9847-cb3901b07be4)


## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/ai-website-chatbot.git
```

2. Navigate to the project directory:

```
cd ai-website-chatbot
```

3. Create a virtual environment (optional, but recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

4. Install the required dependencies:

```
pip install -r requirements.txt
```

5. Obtain a Google Generative AI API key:
   - Visit the [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the Generative AI API
   - Create an API key and replace the `GOOGLE_API_KEY` value in the `app.py` file.

6. Run the Streamlit application:

```
streamlit run app.py
```

The application will start running, and you can access it in your web browser at `http://localhost:8501`.

## Usage

1. Enter the URL of the website you want to interact with in the sidebar.
2. Click the "Load Website" button to fetch and process the website content.
3. Once the content is loaded, you can start asking questions about the website in the main input field.
4. The chatbot will provide responses based on the website content.

## Dependencies

- [Streamlit](https://www.streamlit.io/) - For building the web application
- [Requests](https://requests.readthedocs.io/en/latest/) - For fetching website content
- [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) - For parsing HTML content
- [LangChain](https://langchain.readthedocs.io/en/latest/) - For building the QA chain
- [Google Generative AI](https://github.com/google-research/generative-ai) - For text embedding and language model integration
- [FAISS](https://github.com/facebookresearch/faiss) - For creating and managing the vector store


