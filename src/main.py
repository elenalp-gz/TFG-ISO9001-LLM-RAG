import os
import time
import warnings

# Ignorar advertencias de deprecación para que la terminal se vea limpia
warnings.filterwarnings("ignore", category=UserWarning)

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
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(script_dir, "..")

pdf_path = os.path.join(base_path, "data", "normativa", "NOM_ISO_9001-2015.pdf")
empresa_dir = os.path.join(base_path, "data", "empresa")
prompt_path = os.path.join(base_path, "prompts", "instrucciones.txt")

# ==========================================
# 1. INGESTA DE DATOS (SISTEMA RAG)
# ==========================================
print("\n--- 📚 CARGANDO BASE DE CONOCIMIENTO (ISO 9001 + EMPRESA) ---")

# Cargamos PDF y Markdown
try:
    pdf_loader = PyPDFLoader(pdf_path)
    markdown_loader = DirectoryLoader(empresa_dir, glob="*.md", loader_cls=TextLoader)
    docs = pdf_loader.load() + markdown_loader.load()
except Exception as e:
    print(f"❌ Error al cargar documentos: {e}")
    docs = []

# Fragmentación (Chunking)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_documents(docs)

# Embeddings y Base Vectorial
embeddings = HuggingFaceEmbeddings(model_name="hiiamsid/sentence_similarity_spanish_es")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ==========================================
# 2. DISEÑO DEL PROMPT (FEW-SHOT Y SYSTEM)
# ==========================================
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

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | final_prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# 4. INTERFAZ INTERACTIVA POR TERMINAL
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("      🤖 SISTEMA RAG - CONSULTOR ISO 9001:2015 🤖")
    print("            Empresa: Muebles ArteLocal S.L.")
    print("="*60)
    print("Instrucciones: Escribe tu duda o el documento que quieres generar.")
    print("Escribe 'salir' para cerrar el programa.\n")

    while True:
        user_input = input("👉 TU PREGUNTA: ")

        if user_input.lower() in ["salir", "exit", "quit", "adiós"]:
            print("\nCerrando el sistema. ¡Buen trabajo con la calidad, Elena!")
            break

        if not user_input.strip():
            continue

        print("\n🔍 Analizando normativa y archivos locales...")
        
        inicio = time.time()
        try:
            respuesta = rag_chain.invoke(user_input)
            fin = time.time()

            print("\n" + "—"*50)
            print(respuesta)
            print("—"*50)
            print(f"⏱️ Latencia: {fin - inicio:.2f} segundos | 📌 Llama 3 Local\n")
            
        except Exception as e:
            print(f"❌ Error en la generación: {e}")