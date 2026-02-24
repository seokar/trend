from fastapi import FastAPI
from pydantic import BaseModel
# وارد کردن توابع از فایل شما
from search_tools import web_search, grep_search, codebase_search

app = FastAPI()

class SearchQuery(BaseModel):
    query: str

@app.post("/api/web-search")
async def do_web_search(data: SearchQuery):
    # فراخوانی تابع وب سرچ شما
    result = web_search(search_term=data.query, max_results=5)
    return result

@app.post("/api/grep")
async def do_grep_search(data: SearchQuery):
    # فراخوانی تابع جستجوی فایل شما
    result = grep_search(query=data.query)
    return result
