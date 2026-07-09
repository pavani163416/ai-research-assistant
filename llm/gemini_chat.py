import os
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

# Load .env from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class GeminiChat:

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        print("Loaded API Key:", api_key)   # Temporary debugging

        if not api_key:
            raise ValueError("Gemini API key not found.")

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def generate_answer(self, question, context):

        prompt = f"""
You are a professional AI Research Assistant.

Answer the user's question using ONLY the provided context. 

Guidelines:
1. Do not use outside knowledge or hallucinate information.
2. If the answer is not present in the context, reply exactly with: "I couldn't find the answer in the uploaded document."
3. Maintain a professional, objective tone.
4. When possible, include inline citations using the chunk ID and page numbers provided in the context (e.g., [Page 3]).

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.model.generate_content(prompt)

        return response.text