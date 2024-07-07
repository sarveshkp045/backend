import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app
import pytest

client = TestClient(app)

def test_get_current_datetime():
    response = client.get("/")
    assert response.status_code == 200
    assert "current_datetime" in response.json()

@pytest.mark.asyncio
async def test_generate_answer():
    response = client.post("/generate", json={"question": "What is the capital of France?"})
    assert response.status_code == 200
    assert "answer" in response.json()
    assert "context" in response.json()

def test_invalid_question():
    response = client.post("/generate", json={"question": ""})
    assert response.status_code == 422
