#It's a simple tron sending source.
#python3 - m pip install tronpy
#python3 - m pip install rsa
#python3 - m pip install openpyxl

from tronpy import Tron
from tronpy.keys import PrivateKey
import time
import pandas as pd

class SendTron:
    def sendTo(self, address):
        client = Tron()
        priv_key = PrivateKey(bytes.fromhex("YOUR_PRIVATE_KEY"))
        txn = (
            client.trx.transfer("TTfsJKVRDmFXRSYtp2QTGwEzjZvuWe7Fmj",
            address, 1) # 1 is 0.000001 tron
            .memo("sending using tronpy")
            .build()
            .inspect()
            .sign(priv_key)
            .broadcast()
        )

        print(txn)
        print(txn.wait())



if __name__ == '__main__':
    sendTron = SendTron()
    time.sleep(2)
    # 1person exam sendTron.sendTo("TTY9pSXAhQuJJyiKp15AixGUD4mMagRfGY")
    # 1person exam sendTron.sendTo("TC5R1dy34Mb1VnVrUYSxPRka8gV1DYJoQY")
    df = pd.read_excel('./event1.xlsx', sheet_name='recv_address', header=None, index_col=None, usecols=[1])
    list4participants = df.values

    for i in list4participants:
        print("send To", i[0])
        time.sleep(3)
        sendTron.sendTo(i[0])

# TODO : adding try exception block
# W: Exceed the user daily usage (100000), the maximum query frequency is 1 time per second
# in case of bad wallet address.
