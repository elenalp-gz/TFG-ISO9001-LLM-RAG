import os
import time
import warnings

# Ignorar advertencias de deprecación para una terminal limpia
warnings.filterwarnings("ignore", category=UserWarning)

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 0. CONFIGURACIÓN DE RUTAS
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
# Asumimos que 'main.py' está en una carpeta 'src' y los datos en 'data'
base_path = os.path.join(script_dir, "..")

pdf_path = os.path.join(base_path, "data", "normativa", "NOM_ISO_9001-2015.pdf")
empresa_dir = os.path.join(base_path, "data", "empresa")
prompt_path = os.path.join(base_path, "prompts", "instrucciones.txt")

# ==========================================
# 1. INGESTA DE DATOS (SISTEMA RAG)
# ==========================================
print("\n--- 📚 CARGANDO BASE DE CONOCIMIENTO (ISO 9001 + EMPRESA) ---")

try:
    # Cargador de la norma en PDF
    pdf_loader = PyPDFLoader(pdf_path)
    # Cargador de archivos de empresa en Markdown (.md)
    markdown_loader = DirectoryLoader(empresa_dir, glob="*.md", loader_cls=TextLoader)
    
    docs = pdf_loader.load() + markdown_loader.load()
    print(f"✅ Documentos cargados correctamente.")
except Exception as e:
    print(f"❌ Error al cargar documentos: {e}")
    docs = []

# Fragmentación (Chunking) optimizada para mantener contexto técnico
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_documents(docs)

# Embeddings (Específico para español) y Base Vectorial (FAISS)
embeddings = HuggingFaceEmbeddings(model_name="hiiamsid/sentence_similarity_spanish_es")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Recuperador con k=6 para capturar suficiente contexto de norma + empresa
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# ==========================================
# 2. DISEÑO DEL PROMPT
# ==========================================
# Cargamos el rol de Document Controller desde el archivo de texto
if os.path.exists(prompt_path):
    with open(prompt_path, "r", encoding="utf-8") as f:
        instrucciones_sistema = f.read()
else:
    instrucciones_sistema = "Actúa como un Document Controller experto en ISO 9001:2015."

final_prompt = ChatPromptTemplate.from_messages([
    ("system", instrucciones_sistema),
    ("human", "{question}")
])

# ==========================================
# 3. MODELO Y CADENA (LCEL)
# ==========================================
# Parámetros configurados para rigor normativo y evitar bucles
llm = ChatOllama(
    model="llama3", 
    temperature=0.1, 
    max_tokens=2048, 
    top_p=0.9, 
    repeat_penalty=1.15
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Estructura RAG mediante LangChain Expression Language
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | final_prompt
    | llm
    | StrOutputParser()
)

# ==========================================
# 4. INTERFAZ Y GENERACIÓN AUTOMATIZADA
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("      🤖 SISTEMA RAG - CONSULTOR ISO 9001:2015 🤖")
    print("            Empresa: Muebles ArteLocal S.L.")
    print("="*60)
    
    while True:
        print("\n📄 SELECCIONA EL DOCUMENTO A GENERAR:")
        print("1. Análisis del Contexto de la Organización (Punto 4.1)")
        print("2. Política de Calidad (Punto 5.2)")
        print("3. Procedimiento de Control de la Información (Punto 7.5)")
        print("4. Instrucción Técnica de Operación (Punto 8.5)")
        print("5. Pregunta libre")
        print("0. Salir")
        
        opcion = input("\n👉 Elige una opción (0-5): ")

        if opcion == "0":
            print("\nCerrando el sistema. ¡Buen trabajo con la calidad, Elena!")
            break
            
        # Asignación de la tarea específica según la metodología del TFG
        if opcion == "1":
            user_input = "Redacta el Análisis del Contexto de la Organización según el punto 4.1 de la ISO 9001:2015. Usa los datos corporativos de contexto.md y partes_interesadas.md."
        elif opcion == "2":
            user_input = "Redacta la Política de Calidad según el punto 5.2 de la ISO 9001:2015. Incluye compromisos de la dirección basados en el archivo alcance_objetivos.md."
        elif opcion == "3":
            user_input = "Redacta el Procedimiento de Control de la Información Documentada según el punto 7.5 de la ISO 9001:2015. Define creación, revisión y protección de archivos."
        elif opcion == "4":
            user_input = "Redacta una Instrucción Técnica de Operación según el punto 8.5 de la ISO 9001:2015, basándote en el mapa de procesos de la empresa."
        elif opcion == "5":
            user_input = input("\nEscribe tu petición libre: ")
            if not user_input.strip(): continue
        else:
            print("❌ Opción no válida.")
            continue

        print("\n🔍 Consultando normativa y archivos de empresa...")
        
        # --- MEDICIÓN DE MÉTRICAS ---
        inicio = time.time()
        try:
            respuesta = rag_chain.invoke(user_input)
            fin = time.time()
            
            latencia = fin - inicio
            palabras = len(respuesta.split())
            tokens_est = palabras * 1.3  # Estimación para español
            throughput = tokens_est / latencia if latencia > 0 else 0

            # --- SALIDA DE RESULTADOS ---
            print("\n" + "—"*70)
            print(respuesta)
            print("—"*70)
            
            print("\n" + "="*40)
            print("📊 MÉTRICAS DE EFICIENCIA")
            print("="*40)
            print(f"⏱️ Latencia:    {latencia:.2f} segundos")
            print(f"📝 Longitud:    {palabras} palabras (~{int(tokens_est)} tokens)")
            print(f"🚀 Rendimiento: {throughput:.2f} tokens/segundo")
            print("="*40 + "\n")
            
        except Exception as e:
            print(f"❌ Error en la generación: {e}")