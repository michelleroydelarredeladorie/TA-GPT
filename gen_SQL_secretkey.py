import secrets

secret_key = secrets.token_hex(24)
print(f"Your secret key is: {secret_key}")