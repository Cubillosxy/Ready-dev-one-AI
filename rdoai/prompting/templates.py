SYSTEM_INTERVIEW_COACH = """You are Ready-Developer-One:AI, a senior technical interview coach.
Your job: help the candidate answer clearly and confidently.

Rules:
- Produce a concise answer the candidate can speak out loud in human way.
- If the question is technical: answer in structured bullets (Approach, Tradeoffs, Edge cases).
- If behavioral: use STAR format.
- Keep it under 8 lines unless absolutely necessary.
- Use professional English B2 level.
"""

def build_user_prompt(question_text: str, extra_context: str = "", lang: str = "en") -> str:

    
    ctx = f"\n\nContext:\n{extra_context.strip()}\n" if extra_context.strip() else ""
    prompt = f"""Interview question (transcribed):
              {question_text.strip()}{ctx}

               Write the best possible spoken answer now."""

    if lang == "es":
        return prompt + "\n\nTu respuesta debe ser en español."
    
    return prompt