import base64
import json

from Crypto.Util.Padding import pad, unpad
from Crypto.Cipher import AES

class MyCrypt():
    def __init__(self, key):
        self.key = key
        # self.iv = iv
        self.mode = AES.MODE_ECB


    # def encrypt(self, message):
    #     message = message.encode()
    #     raw = pad(message)
    #     cipher = AES.new(self.key, AES.MODE_ECB, self.__iv().encode('utf8'))
    #     enc = cipher.encrypt(raw)
    #     return base64.b64encode(enc).decode('utf-8')
    #     # return base64.b64encode(enc)

    def encrypt(self, text):
        # cryptor = AES.new(self.key, self.mode, self.__iv().encode('utf8'))
        cryptor = AES.new(self.key, self.mode)
        length = 16
        text = pad(text, 16)
        self.ciphertext = cryptor.encrypt(text)
        return self.ciphertext
    def __iv(self):
        return chr(0) * 16


# key = base64.b64decode('PyxZO31GlgKvWm+3GLySzAAAAAAAAAAAAAAAAAAAAAA=')
key = b'vktmfaleldj_0017'
# IV = base64.b64decode('AAAAAAAAAAAAAAAAAAAAAA==')
# IV1 = chr(0) * 16
# IV = str(IV1).encode('utf-8')
# print("IV = ", IV.encode('utf-8'))
plainText = 'y_device=y_C9DB602E-0EB7-4FF4-831E-8DA8CEE0BBF5'.encode('utf-8')
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
print('plainText = ', plainText)
print("json.dumps(data) = ", json.dumps(data).encode('utf-8'))
crypto = MyCrypt(key)
# encrypt_data = crypto.encrypt(plainText)
encrypt_data = crypto.encrypt(json.dumps(data).encode('utf-8'))
print('encrypt data = ', encrypt_data)
encoder = base64.b64encode(encrypt_data)
print('encoder = ', encoder)