from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm= HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        # tempertature decides how different the response would be when same propmt is given
        # if same prompt  and temp=0, it will give almost same response everytime the same prompt give 
        # but it will give different repose every time even with same response when temp>0
        temperature=0.5,
        max_new_tokens=100
    )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke(
    '''
        What is the Captial of India?
    '''
)

print(result.content)

