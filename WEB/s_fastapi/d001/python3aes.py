import base64
import hashlib
from Crypto.Cipher import AES
import json
#AES/ECB/PKCS7
BS = 16
pad = (lambda s: s + (BS - len(s) % BS) * chr(BS - len(s) % BS).encode())
unpad = (lambda s: s[:-ord(s[len(s) - 1:])])

class AESCipher(object):
    def __init__(self, key):
        self.key = hashlib.sha256(key.encode()).digest()

    def encrypt(self, message):
        message = message.encode()
        raw = pad(message)
        cipher = AES.new(self.key, AES.MODE_ECB, self.__iv().encode('utf8'))
        enc = cipher.encrypt(raw)
        return base64.b64encode(enc).decode('utf-8')
        # return base64.b64encode(enc)

    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        cipher = AES.new(self.key, AES.MODE_ECB, self.__iv().encode('utf8'))
        dec = cipher.decrypt(enc)
        return unpad(dec).decode('utf-8')

    def __iv(self):
        return chr(0) * 16

# key = "vktmfaleldj_0017" #partner_key
key = "parcelmedia_rsv1" #server_key
data = {
        "reserveInfo": {
            "delivery_code": "korex",
            "s_name": "홍길동",
            "s_phone1": "02",
            "s_phone2": "5555",
            "s_phone3": "5555",
            "s_mobile1": "010",
            "s_mobile2": "6666",
            "s_mobile3": "6666",
            "s_zipcode": "210-822",
            "s_addr1": "서울특별시",
            "s_addr2": "성북동",
            "r_name": "공길동",
            "r_phone1": "02",
            "r_phone2": "6666",
            "r_phone3": "6666",
            "r_mobile1": "010",
            "r_mobile2": "7777",
            "r_mobile3": "7777",
            "r_zipcode": "699-903",
            "r_addr1": "서울특별시",
            "r_addr2": "성북동",
            "d_req_date": "2015-10-02",
            "p_div": "G",
            "p_amt": "1",
            "p_pric": "20000",
            "p_name": "건어물",
            "etc_info": "기타",
            "partner_reserve_num": "23489038523904u79023",
            "partner_user_id": "ssddccee"
        }
    }
print("json.dumps(data)", json.dumps(data))
aes = AESCipher(key)

encrypt = aes.encrypt(json.dumps(data))
# encrypt_utf = str(encrypt, encoding="utf-8")
# print("암호화:", encrypt_utf)
print("암호화:", encrypt)
print("-"*100)

decrypt = aes.decrypt(encrypt)
print("복호화:", decrypt)
print("-"*100)