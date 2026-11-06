# Importing environment variables
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI')

# https://www.geeksforgeeks.org/python/database-integration-with-fastapi/ 

from fastapi import FastAPI 

app = FastAPI()

@app.get('/')
def read_root():
    return {"Initialized"}

from pydantic import BaseModel



# Using MongoDB Atlas for backend

import asyncio
from pymongo import AsyncMongoClient

async def main():
    client = AsyncMongoClient(MONGODB_URI)

    await client.admin.command("ping")
    print("Connected successfully")
    await client.close()

asyncio.run(main())