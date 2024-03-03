import json

def init():
    print('init')
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
        "token": "tokenlasdjflasjdflijaef"
    }
    invoice_req['token'] = 'abcd'
    print(invoice_req['token'])
    print(invoice_req['receipt']['arrivals']['address'])


if __name__ == '__main__':
    init()