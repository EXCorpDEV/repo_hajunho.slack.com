import requests
import json

authentication_api_url = "https://api-server-dev-z27272c3ca-du.a.run.app/open/authentication"
invoice_api_url = "https://api-server-dev-z27272c3ca-du.a.run.app/open/invoice/receipt"

invoice_req = {
  "receipt": {
    "applicant": {
      "name": "하준호_테스트서버",
      "phone": "010-2525-6337"
    },
    "arrivals": {
      "name": "하준호_테스트",
      "phone": "010-2525-6337",
      "address": "경기 오산시 독산성로 425",
      "detail": "521호 쏘닥 수신"
    },
    "departures": {
      "name": "하준호_테스트",
      "phone": "1644-2447",
      "address": "경기 수원시 영통구 영통로 154번길 51-16",
      "detail": "521호 쏘닥 발송"
    },
    "carType": "오토바이",
    "paymentType": "선불",
    "requested": "정산 요금은 일간/주간/월간 일괄 결제 입니다.",
    "boxType": "1"
  },
  "token": ""
}

def init():
    print("RestAPI : ")
    postAPI()

def postAPI():
    req_body = {
        "id": "쏘닥",
        "pw": "0000"
    }
    req_data = requests.post(
        url=f"{authentication_api_url}",
        headers={
            "Content-Type": "application/json"
        },
        data=json.dumps(req_body),
        timeout=10
    ).json()
    print(req_data)
    print(req_data['data']['token'])

    invoice_req['token'] = req_data['data']['token']
    invoice_req_json = json.dumps(invoice_req)

    req_data2 = requests.post(
        url=f"{invoice_api_url}",
        headers={
            "Content-Type": "application/json"
        },
        data=invoice_req_json,
        timeout=10
    ).json()

    print(req_data2)
    print(req_data2['data']['invoice'])


if __name__ == '__main__':
    init()