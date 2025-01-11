import os

from dotenv import load_dotenv
import vertexai
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory, ChatVertexAI
from vertexai.preview import reasoning_engines

# Load the environment variables from the .env file
load_dotenv()

# VertexAI Environment Variables
vertex_project_id = os.getenv("VERTEXAI_PROJECT_ID")
vertex_location = os.getenv("VERTEXAI_LOCATION")
vertex_bucket = os.getenv("VERTEXAI_GCP_BUCKET")

# Setup VertexAI SDK
vertexai.init(
    project=vertex_project_id,
    location=vertex_location,
    staging_bucket=f"gs://{vertex_bucket}",
)

# Define model
model = "gemini-1.5-flash-001"

# (Optional) Configure Saftey settings of model 
safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

# (Optional) Configure model parameters
model_kwargs = {
    # temperature (float): The sampling temperature controls the degree of
    # randomness in token selection.
    "temperature": 0.28,
    # max_output_tokens (int): The token limit determines the maximum amount of
    # text output from one prompt.
    "max_output_tokens": 1000,
    # top_p (float): Tokens are selected from most probable to least until
    # the sum of their probabilities equals the top-p value.
    "top_p": 0.95,
    # top_k (int): The next token is selected from among the top-k most
    # probable tokens. This is not supported by all model versions. See
    # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding#valid_parameter_values
    # for details.
    "top_k": None,
    # safety_settings (Dict[HarmCategory, HarmBlockThreshold]): The safety
    # settings to use for generating content.
    # (you must create your safety settings using the previous step first).
    "safety_settings": safety_settings,
}

# Instantiate model
llm = ChatVertexAI(
    model=model, 
    model_kwargs=model_kwargs
)

# Invoke model
messages = [
    ("system", "You are a helpful translator. Translate the user sentence to Hindi."),
    ("human", "My name is Vikas"),
]
response = llm.invoke(messages)
print(response)
