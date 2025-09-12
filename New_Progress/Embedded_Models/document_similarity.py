from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv("../.env")


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"

)

docs = [
    "Sachin Tendulkar is widely regarded as one of the greatest batsmen in cricket history and is often called the 'God of Cricket'.",
    "The legendary field hockey player Dhyan Chand was renowned for his extraordinary ball control and amazing goal-scoring feats.",
    "Neeraj Chopra is a celebrated track and field athlete who became the first Indian to win an Olympic gold medal in the javelin throw.",
    "P.V. Sindhu stands as one of India's most successful shuttlers and is the first Indian to become the Badminton World Champion.",
    "Viswanathan Anand is an acclaimed Indian chess Grandmaster who has been crowned the World Chess Champion five times.",
    "Olympic boxer Mary Kom holds the incredible record of being the only woman to win the World Amateur Boxing Championship six times.",
    "Former professional shooter Abhinav Bindra made history as the first Indian to secure an individual Olympic gold medal.",
    "Sunil Chhetri captains the Indian national football team and is recognized as one of the world's leading active international goalscorers.",
    "Indian weightlifter Mirabai Chanu proudly won a silver medal at the 2020 Tokyo Olympics, showcasing immense strength.",
    "Pankaj Advani is a dominant force in cue sports, having won the International Billiards and Snooker Federation world championship an incredible 25 times."
]

query = "Tell me about a billiards player."

query_embed = embeddings.embed_query(query)
docs_embed = embeddings.embed_documents(docs)

scores = cosine_similarity([query_embed],docs_embed)[0]

index,score = sorted(list(enumerate(scores)),key = lambda x:x[1])[-1]

print(query)
print(docs[index])
print("Cosine similarity is ",score)