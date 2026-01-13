from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader = DirectoryLoader(
    path='Docs',
    glob='*pdf',
    loader_cls=PyPDFLoader
)

# lazy_load loads only when asked, works like generator function
docs = loader.lazy_load()

for doc in docs:
    print(doc.metadata)