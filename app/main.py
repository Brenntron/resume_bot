import json
import requests
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydantic import BaseModel, EmailStr, constr
from pydantic_settings import BaseSettings
from openai import OpenAI
from pypdf import PdfReader

class Settings(BaseSettings):
    openai_api_key: str
    pushover_token: str
    pushover_user: str
    allowed_origins: list[str]
    allowed_hosts: list[str]
    environment: str = "production"  # default to production for safety

    class Config:
        env_file = ".env"

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
settings = Settings()
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["post"],
    allow_headers=["*"],
    max_age=60,
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.allowed_hosts
)

# Only enable HTTPS redirect in production
if settings.is_production:
    app.add_middleware(HTTPSRedirectMiddleware)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": settings.pushover_token,
            "user": settings.pushover_user,
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]

class ChatMessage(BaseModel):
    message: constr(min_length=1, max_length=1000)
    history: list = []

class UserDetails(BaseModel):
    email: EmailStr
    name: str | None = None
    notes: str | None = None

class Me:

    def __init__(self):
        self.openai = OpenAI(api_key=settings.openai_api_key)
        self.name = "Brennan Willingham"
        reader = PdfReader("app/me/linkedin.pdf")
        self.linkedin = ""
        self.summary = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("app/me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results

    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt

    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
            return response.choices[0].message.content


me = Me()

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Srict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Error: {exec}", exc_info=True)
    return JSONResponse(
    status_code=500,
        content={"error": "An internal errror occurred"}
    )

# API for handling chat POST requests
@app.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request):
    body = await request.json()
    message = body.get("message")
    history = body.get("history", [])

    if len(str(request.body())) > 4096:
        raise HTTPException(status_code=413, detail="Request too large")

    if not message:
        return JSONResponse(content={"error": "Missing required 'message' field."}, status_code=400)

    try:
        # Pass the input message and optional chat history to Me.chat
        response = me.chat(message, history)
        return JSONResponse(content={"response": response})
    except Exception as e:
        # Handle unexpected errors gracefully
        return JSONResponse(content={"error": str(e)}, status_code=500)
