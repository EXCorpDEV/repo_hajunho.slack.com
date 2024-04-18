import hashlib
import time

def mine(block_number, transactions, previous_hash, prefix_zeros):
    prefix_str = '0' * prefix_zeros
    nonce = 0

    while True:
        text = str(block_number) + transactions + previous_hash + str(nonce)
        new_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if new_hash.startswith(prefix_str):
            print(f"Bitcoin mined with nonce value: {nonce}")
            print(f"Hash: {new_hash}")
            return new_hash
        nonce += 1

if __name__ == '__main__':
    block_number = 674845
    transactions = """
    Alice pays Bob 1 BTC
    Bob pays Charlie 0.5 BTC
    """
    previous_hash = '0000000000000000000b4cc4ab7d18ae1acdb5b9a2ec3f96f382fe9cb617f649'
    difficulty = 4  # Change this to higher number to make it harder

    start = time.time()
    print("Mining started...")
    new_hash = mine(block_number, transactions, previous_hash, difficulty)
    total_time = str((time.time() - start))
    print(f"Mining ended. Mining took: {total_time} seconds")
