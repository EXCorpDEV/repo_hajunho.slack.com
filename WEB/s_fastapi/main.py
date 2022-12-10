from fastapi import FastAPI
import re
import io
import datetime
import hashlib
import json
import requests

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/request2est")
async def post_test():
    req_data={
        "auth_info": {
            "partner_code": "sodn",
            "partner_auth": "7d003a"
        },
        "request_info": {
            "priceInfo": {
                "delivery_code": "korex",
                "rsv_gbn": "G",
                "p_div": "G",
                "s_zipcode": "123-822",
                "r_zipcode": "456-903",
                "p_amt": "1"
            }
        }
    }
    print(json.dumps(req_data))
    req_response = requests.post(
        url="https://sodoc.net/packages",
        headers={"Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
                 "User-Agent":"PostmanRuntime/7.29.2", "Accept-Encoding":"gzip, deflate, br"},
        json=json.dumps(req_data)
    )
    print("==req_response==")
    print(req_response)
    print("==req_response.content==")
    print(req_response.content, req_response.text, req_response.content)
    print("======================")
    return req_response
