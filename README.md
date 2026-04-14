# TFG-ISO9001-LLM-RAG
Comparativa de modelos LLM Open Source para la generación automática de documentación ISO 9001:2015. 

## Características Principales
* **Privacidad Total:** Ejecución 100% local mediante **Ollama** y **Llama 3**.
* **Cerebro Normativo:** Consulta en tiempo real la norma **ISO 9001:2015** (PDF).
* **Contexto Empresarial:** Base de conocimiento sobre procesos de carpintería, organigrama y objetivos (Markdown).
* **Interfaz Interactiva:** Consola técnica para generación de documentos y resolución de dudas de auditoría.

---

## Stack Tecnológico
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-VectorStore-blue?style=for-the-badge)

---

## Estructura del Proyecto
```text
.
├── data/
│   ├── normativa/    # PDF oficial ISO 9001
│   └── empresa/      # Contexto, procesos y riesgos (.md)
├── prompts/          # Ingeniería de prompts (System & Few-Shot)
├── src/              # Código fuente (main.py)
└── results/          # Histórico de documentos generados