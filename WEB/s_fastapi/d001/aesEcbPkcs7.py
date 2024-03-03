import base64
import json
from Crypto import Random
from Crypto.Cipher import AES

class AESCrypt:
    """
    AES/ECB/PKCS#7 加密算法
    """
    BLOCK_SIZE = 16

    def pkcs7_pad(self, s):
        length = self.BLOCK_SIZE - (len(s) % self.BLOCK_SIZE)
        s += bytes([length]) * length
        return s

    def pkcs7_unpad(self, s):
        """
        unpadding according to PKCS #7
        @param s: string to unpad
        @type s: byte
        @rtype: byte
        """
        sd = -(s[-1])
        return s[0:sd]

    def __encrypt__(self, plain_text, secret_key):
        if (plain_text is None) or (len(plain_text) == 0):
            raise ValueError('input text cannot be null or empty set')

        if not secret_key or (len(secret_key) == 0):
            raise ValueError('密钥长度错误')

        plain_bytes = plain_text.encode('utf-8')
        raw = self.pkcs7_pad(plain_bytes)
        print(raw)
        print(len(raw))
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(secret_key, AES.MODE_ECB, iv)

        cipher_bytes = cipher.encrypt(raw)

        cipher_text = self.base64_encode(iv + cipher_bytes)
        return cipher_text

    def __decrypt__(self, cipher_text, secret_key):

        if not secret_key or (len(secret_key) == 0):
            raise ValueError('密钥长度错误')

        cipher_bytes = self.base64_decode(cipher_text)
        iv = cipher_bytes[:AES.block_size]
        cipher_data = cipher_bytes[AES.block_size:]

        cipher = AES.new(secret_key, AES.MODE_ECB, iv)
        plain_pad = cipher.decrypt(cipher_data)
        plain_text = self.pkcs7_unpad(plain_pad)
        return plain_text.decode()

    def base64_encode(self, bytes_data):
        """
        加base64
        :type bytes_data: byte
        :rtype 返回类型: string
        """
        return (base64.urlsafe_b64encode(bytes_data)).decode()

    def base64_decode(self, str_data):
        """
        解base64
        :type str_data: string
        :rtype 返回类型: byte
        """
        return base64.urlsafe_b64decode(str_data)


class HowToUse:
    def test_encrypt(self):
        # secret_key = 'hotel!!!!!'
        plain_text = 'test test test test aes'
        # secret_key md5后
        secret_key_md5 = '5373a944a96d5c656cd4307012b8e2d0'
        # secret_key_md5 = hash_md5(secret_key)

        aes_crypt = AESCrypt()
        cipher_text = aes_crypt.__encrypt__(plain_text, secret_key_md5)

        plain_data = aes_crypt.__decrypt__(cipher_text, secret_key_md5)

        assert plain_data == plain_text

HowToUse.test_encrypt()