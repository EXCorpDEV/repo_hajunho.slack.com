# Taken from http://stackoverflow.com/questions/12524994/encrypt-decrypt-using-pycrypto-aes-256/12525165#12525165
import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES

class AESCipher(object):

    def __init__(self, key):
        self.bs = 32
        self.key = hashlib.sha256(key.encode()).digest()

    def encrypt(self, raw):
        raw = self._pad(raw)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(raw))

    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size:])).decode('utf-8')

    def _pad(self, s):
        return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s)-1:])]

def main():
    SECRET_KEY = b"1234567890123456"
    print("Hello World!")
    # aesinput=AESCipher(key="adsfasdfasdfsdfsadfsafdasdf")
    # res = aesinput.encrypt(raw="adsfasdfasfasdfasdffd")
    # print(res)
    c = AESCipher(SECRET_KEY)
    enc_crypt = c.encrypt(raw='this is plain text')
    print(enc_crypt)

if __name__ == "__main__":
    main()