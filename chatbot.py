from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS


class ChatBot:
    def __init__(self, csv_path, api_key, model_name):
        # Initialize attributes
        self.csv_path = csv_path
        self.api_key = api_key
        self.model_name = model_name

        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

        # Load data from CSV
        self.loader = CSVLoader(file_path=csv_path)
        self.data = self.loader.load()

        # Create FAISS index and retriever
        self.faiss_index = FAISS.from_documents(documents=self.data, embedding=self.embeddings)
        self.retriever = self.faiss_index.as_retriever()

        # Initialize the language model
        self.llm = ChatGoogleGenerativeAI(
            api_key=api_key,
            model=model_name,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that combines the following question answers to generate a single question and answer: {question_answer}."
                ),
                ("human", "{actual_question}")
            ]
        )

        # Chain prompt with the language model
        self.chain = self.prompt | self.llm

    def populate_result(self, question):
        try:
            # Retrieve relevant documents
            # faiss_results = self.faiss_index.similarity_search("apply for online student service in tatkal service?", k=1)
            # print(faiss_results)
            # Process retrieved documents into a structured format
            # relevant_docs = self.retriever.get_relevant_documents(question)
            relevant_docs = self.faiss_index.similarity_search(question,k=1)
            result = []

            for doc in relevant_docs:
                prompt = doc.page_content.split(r'|')[0].replace('prompt: ', '')
                response = doc.page_content.split('|')[1].replace('\nresponse: ', '')
                print(prompt)
                result.append({"question": prompt, "answer": response})

            return result
        except Exception as e:
            print(f"Error in populate_result: {e}")
            return []

    def ask_question(self, user_input):
        try:
            # Get relevant results
            result = self.populate_result(user_input)
            print(result)

            if not result:
                return {
                    "error": "No relevant documents found.",
                    "related_Doc": []
                }

            # Generate the AI response
            # ai_response = self.chain.invoke(
            #     {
            #         "question_answer": result,
            #         "actual_question": user_input
            #     }
            # ).content

            return {
                "AI": ai_response,
                "related_Doc": result
            }
        except Exception as e:
            print(f"Error in ask_question: {e}")
            return {"error": f"An error occurred: {str(e)}"}


# Example Usage

    
#     # Start the Flask app on the default port (5000)
# k =  ChatBot(csv_path=r"D:\ChatBot\python flask server\JNTUH_Student_Services_FAQ_updated.csv",
#         api_key="AIzaSyBAVrawMMPMRmUaXXV-mk-ujZpsbSZbtTQ",
#         model_name="gemini-1.5-pro"
#     )
# user_question = "about JNTUH college?"
# response = k.ask_question(user_question)
