from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import os
load_dotenv()
# Azure creds
AZURE_ENDPOINT = os.getenv("endpoint")
AZURE_KEY = os.getenv("key")

client = DocumentIntelligenceClient(endpoint=AZURE_ENDPOINT, credential=AzureKeyCredential(AZURE_KEY))

def extract_text_azure(file_path: str) -> tuple[str, int]:
    with open(file_path, "rb") as f:
    # Read the bytes
        file_bytes = f.read()
    req = AnalyzeDocumentRequest(bytes_source=file_bytes)

    poller = client.begin_analyze_document(
        model_id="prebuilt-read", 
        analyze_request=req
    )
    result = poller.result()
    num_pages = 1
    return result.content,num_pages