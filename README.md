# Python AI Journey & IT Support AI Agent Capstone

A comprehensive repository documenting my journey from foundational Python to building a production-ready **IT Support AI Agent** using Google Gemini, RAG (Retrieval-Augmented Generation), SQLite persistence, Docker containerization, and automated CI/CD testing.

---

## 🚀 Capstone: IT Support AI Agent

The capstone project is an autonomous IT Support Agent built with Flask, Google Gemini (`gemini-3.6-flash`), Vector Embeddings (`gemini-embedding-001`), and SQLite.

### 🌟 Key Capabilities
- **Intelligent Triage & Decision Making**: Evaluates employee reports and classifies whether to provide a self-service fix or escalate to a human technician based on severity and urgency.
- **RAG Knowledge Retrieval**: Uses semantic vector embeddings with cosine similarity (and robust keyword fallback) to retrieve relevant troubleshooting steps from a curated knowledge base.
- **Persistent Ticket Management**: Escalated issues are automatically recorded in an SQLite database (`tickets.db`) with timestamps and ticket status.
- **Resilient & Production-Ready**:
  - Exponential backoff retry logic for LLM API calls.
  - Rate limiting via `Flask-Limiter` (`20 requests/hour` per IP).
  - Input validation (missing fields, blank strings, max 500 characters).
  - Health check endpoint (`/health`) and structured logging.
- **Containerized Deployment**: Dockerized with Gunicorn for seamless deployment to Render or cloud platforms.
- **Automated CI/CD**: Automated unit testing with `pytest` on GitHub Actions on every push.

---

## 🛣️ The 20-Day Learning Roadmap

| Phase | Days | Focus Topics |
| :--- | :--- | :--- |
| **Python Core Fundamentals** | Days 1–5 | Variables, control flow, functions, modular architecture, data structures (`list`, `dict`), file operations (reading/writing `.txt` & `.csv`), and robust error handling (`try/except`). |
| **Data & Scientific Computing** | Days 6–8 | NumPy vectorization, Pandas DataFrames (filtering, sorting, aggregations), and Matplotlib visualizations (histograms, distributions). |
| **Generative AI & LLM Systems** | Days 9–12 | Gemini API integration, prompt engineering, structured JSON outputs, autonomous agent loops, vector embeddings, and RAG semantic search. |
| **Capstone Engineering** | Days 13–20 | Flask RESTful service, secrets management (`python-dotenv`), Docker containerization, pytest suite & GitHub Actions CI, health monitoring & structured logging, rate limiting & input validation, SQLite persistence. |

---

## 📡 API Endpoints

### 1. Health Check
* **`GET /health`**
* Returns the current service status.
```bash
curl http://localhost:5000/health
```
```json
{
  "service": "it-support-agent",
  "status": "healthy"
}
```

### 2. Ask Support Agent
* **`POST /ask`**
* Submits an employee IT issue for resolution or escalation.
* **Headers**: `Content-Type: application/json`
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "My printer is not responding and won'\''t print."}'
```
* **Sample Self-Service Fix Response**:
```json
{
  "decision": {
    "action": "provide_fix",
    "reason": "Printer connectivity is a common, self-resolvable issue with no critical urgency."
  },
  "result": "Here's a suggested fix for 'My printer is not responding and won't print.': Printer not responding: Check the printer is powered on and connected to the network. Restart the print spooler service on your machine."
}
```
* **Sample Escalation Response**:
```json
{
  "decision": {
    "action": "escalate_to_technician",
    "reason": "Hardware failure with imminent client demo deadline requires technician intervention."
  },
  "result": "This issue has been logged and escalated to a technician: 'Laptop motherboard smoked before client pitch.'"
}
```

### 3. List Escalated Tickets
* **`GET /tickets`** or **`GET /tickets?status=open`**
* Retrieves all logged tickets from the SQLite database.
```bash
curl http://localhost:5000/tickets
```
```json
{
  "count": 2,
  "tickets": [
    {
      "id": 2,
      "employee_question": "Laptop motherboard smoked before client pitch.",
      "decision": "escalate_to_technician",
      "status": "open",
      "created_at": "2026-09-24T09:30:00.000000"
    }
  ]
}
```

---

## 🛠️ Local Development & Testing

### 1. Prerequisites & Virtual Environment
```bash
# Clone the repository
git clone https://github.com/bfawole23/python-ai-journey.git
cd python-ai-journey

# Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r day13/requirements.txt
```

### 2. Environment Variables
Create a `day13/.env` file with your Gemini API key:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
PORT=5000
```

### 3. Run Automated Tests
```bash
pytest day13/test_agent.py -v
```

### 4. Run the Service Locally
```bash
cd day13
python it_support_agent.py
```

---

## 🐳 Docker & Cloud Deployment

### Build and Run with Docker
```bash
# Build Docker image
docker build -t it-support-agent day13/

# Run container locally
docker run -p 5000:5000 -e GEMINI_API_KEY="your_api_key" it-support-agent
```

### Deploying to Render
1. Connect your GitHub repository to Render.
2. Create a new **Web Service** selecting Docker environment (pointing to `day13/Dockerfile`).
3. Set the environment variable `GEMINI_API_KEY` in the Render dashboard.
4. Render will automatically build the container and deploy the service.
