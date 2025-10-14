from google.generativeai import GenerativeModel
import asyncio
import json

async def extract_with_gemini(prompt: str):
    model = GenerativeModel("gemini-2.5-flash")
    # model = GenerativeModel("gemini-2.5-flash-lite")
    response = await asyncio.to_thread(model.generate_content, prompt)
    return response.text
