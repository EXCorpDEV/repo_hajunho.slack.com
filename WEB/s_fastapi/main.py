from fastapi import FastAPI
import re
import io
import datetime
import hashlib
import json
import requests
# from d001.aesPkcs7 import AESCipher
from d001.aesgithub import AESCipher

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.get("/aes")
async def aes_test():
    # aesinput = AESCipher(key="e86c90618195f30ecc97c7a30f7d003a", iv="9393838383838383")
    # print(aesinput)
    # return aesinput.encrypt(raw="ladjflasjdffj") //TODO: object to string
    aesinput = AESCipher(key="asdfasfdafasadf")
    return aesinput.encrypt("asdfasdafafadf")



@app.get("/request2est")
async def post_test():
    req_data={
        "auth_info": {
            "partner_code": "sodn",
            "partner_auth": "e86c90618195f30ecc97c7a30f7d003a"
        },
        "request_info": {
            "priceInfo": {
                "delivery_code": "korex",
                "rsv_gbn": "G",
                "p_div": "G",
                "s_zipcode": "210-822",
                "r_zipcode": "699-903",
                "p_amt": "1"
            }
        }
    }
    print(json.dumps(req_data))
    req_response = requests.post(
        url="https://webcheck37.logii.com/interface/trsvapi/priceInfoProc.asp",
        headers={"Content-Type": "application/json;charset=utf-8",
                 "User-Agent":"PostmanRuntime/7.29.2", "Accept-Encoding":"gzip, deflate, br"},
        # json=json.dumps(req_data)
        data=json.dumps(req_data)
    )
    print("==req_response==")
    print(req_response)
    print("==req_response.content==")
    print(req_response.content, req_response.text, req_response.content)
    print("======================")
    return req_response.text
