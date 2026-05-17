import requests

def get_joke():
    r = requests.get("https://official-joke-api.appspot.com/random_joke")
    return r.json()

if __name__ == "__main__":
    j = get_joke()
    print(f"{j['setup']} {j['punchline']}")
