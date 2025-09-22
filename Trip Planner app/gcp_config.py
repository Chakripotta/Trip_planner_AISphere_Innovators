import vertexai

PROJECT_ID = "hackathon-trip-planner"
LOCATION = "us-central1"

def init_vertex_ai():
    """Initialize Vertex AI SDK."""
    vertexai.init(project=PROJECT_ID, location=LOCATION)
