import os
import pickle
# from langchain.llms import OpenAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.callbacks import get_openai_callback

# OPEN_AI_MODEL = 'gpt-3.5-turbo'

# Load Embedding from Disk
def load_embedding(store_name='APTRA Advance NDC_superviser_Guide.pkl'):
  if os.path.exists(store_name):
    with open(store_name, 'rb') as file:
      VectorStore = pickle.load(file)
      print("Embeddings loaded from the Disk!")
      return VectorStore
    
  else:
    print("Embeddings are not present in the Disk!")
    return None


def handle_query(query: str):
  print("[handle_query] = ", query)
  return "Success response"
  """
  # return "The abbreviation of SPVR TRANS is Supervisor Transactions."
  if query:

    VectorStore = load_embedding()
    if VectorStore != None:
      docs = VectorStore.similarity_search(query=query, k=3)

      # llm = OpenAI(temperature=0, model_name=OPEN_AI_MODEL)
      # chain = load_qa_chain(llm=llm, chain_type='stuff')
      # with get_openai_callback() as cb:
      #   response = "Success response" #chain.run(input_documents=docs, question=query)
      #   print(response)

      #   return response
      response = "Success response" #chain.run(input_documents=docs, question=query)
      print(response)

      return response
    else:
      print("VectorStore has not created.")
    """
