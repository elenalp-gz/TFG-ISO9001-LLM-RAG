import os
import time
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 0. CONFIGURACIÓN DE RUTAS AUTOMÁTICAS
# ==========================================
# Detectamos dónde está este archivo (src/) y calculamos la raíz del proyecto
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(script_dir, "..")

# Definimos rutas exactas a tus carpetas
pdf_path = os.path.join(base_path, "data", "normativa", "NOM_ISO_9001-2015.pdf")
empresa_dir = os.path.join(base_path, "data", "empresa")
prompt_path = os.path.join(base_path, "prompts", "instrucciones.txt")

# ==========================================
# 1. INGESTA DE DATOS (SISTEMA RAG)
# ==========================================
print("--- INICIANDO CARGA DE DOCUMENTACIÓN ---")

# Cargador de PDF (Normativa)
if not os.path.exists(pdf_path):
    print(f"ERROR: No se encuentra el PDF en {pdf_path}")
pdf_loader = PyPDFLoader(pdf_path)

# Cargador de Carpeta (Datos Empresa .md)
markdown_loader = DirectoryLoader(empresa_dir, glob="*.md", loader_cls=TextLoader)

# Unimos y fragmentamos (Chunking)
docs = pdf_loader.load() + markdown_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs)

# Crear Base Vectorial Local
embeddings = HuggingFaceEmbeddings(model_name="hiiamsid/sentence_similarity_spanish_es")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ==========================================
# 2. DISEÑO DEL PROMPT Y FEW-SHOT
# ==========================================
# Ejemplos para guiar el formato de salida
ejemplos = [
    {
        "input": "Hay que montar una mesa de comedor con roble y barniz.",
        "output": """## FICHA DE PROCESO: Montaje de Mesa
**1. Objetivo:** Ensamblar piezas de roble.
**2. Requisito:** ISO 9001 Cláusula 8.5.1.
**3. Pasos:** Corte, lijado y barnizado ecológico.
**4. Control:** Verificación de estabilidad y acabado."""
    }
]

example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=ejemplos)

# Leer instrucciones del sistema desde archivo externo
with open(prompt_path, "r", encoding="utf-8") as f:
    instrucciones_sistema = f.read()

final_prompt = ChatPromptTemplate.from_messages([
    ("system", instrucciones_sistema),
    few_shot_prompt,
    ("human", "{question}")
])

# ==========================================
# 3. MODELO Y CADENA (LCEL)
# ==========================================
llm = ChatOllama(model="llama3", temperature=0.1)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Construcción de la cadena RAG
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | final_prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# 4. EJECUCIÓN Y MÉTRICAS
# ==========================================
if __name__ == "__main__":
    pregunta = "Redacta el proceso de empaquetado de una silla artesanal según las directrices logística de la empresa."
    
    print("\nPROCESANDO CONSULTA...")
    inicio = time.time()
    
    respuesta = rag_chain.invoke(pregunta)
    
    fin = time.time()
    
    print("-" * 30)
    print(respuesta)
    print("-" * 30)
    
    print(f"\n[MÉTRICA] Latencia: {fin - inicio:.2f} segundos")
    print(f"[MODELO] Llama 3 vía Ollama (Local)")