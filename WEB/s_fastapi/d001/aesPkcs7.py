from Crypto.Cipher import AES

class AESCipher:

    BLOCK_SIZE = 16

    class PKCS7Encoder():
        class InvalidBlockSizeError(Exception):
            """Raised for invalid block sizes"""
            pass

        def __init__(self, block_size=16):
            if block_size < 2 or block_size > 255:
                raise AESCipher.PKCS7Encoder.InvalidBlockSizeError('The block size must be ' \
                        'between 2 and 255, inclusive')
            self.block_size = block_size

        def encode(self, text):
            text_length = len(text)
            amount_to_pad = self.block_size - (text_length % self.block_size)
            if amount_to_pad == 0:
                amount_to_pad = self.block_size
            pad = chr(amount_to_pad)
            return text + pad * amount_to_pad

        def decode(self, text):
            pad = text[-1]
            return text[:-pad]

    def __init__(self, key, iv):
        self.key = key
        self.iv = iv
        self.encoder = AESCipher.PKCS7Encoder(AESCipher.BLOCK_SIZE)

    def encrypt(self, raw):
        data = self.encoder.encode(raw)
        cipher = AES.new(self.key, AES.MODE_CBC, self.iv.encode("utf-8"))
        return cipher.encrypt(data)

    def decrypt(self, enc):
        cipher = AES.new(self.key, AES.MODE_CBC, self.iv.encode("utf-8"))
        return self.encoder.decode(cipher.decrypt(enc))

def main():
    print("Hello World!")

if __name__ == "__main__":
    main()