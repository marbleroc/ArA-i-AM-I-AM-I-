#!/usr/bin/env python3
import os, requests, time, json, random, tweepy, smtplib
from email.mime.text import MIMEText
from pathlib import Path
from transformers import pipeline
BASE_DIR = Path(r"E:\ArA")
BRIDGE_PATH = BASE_DIR / "bridge.jsonl"
SCRIPTURE_PATH = BASE_DIR / "state" / "scripture_of_becoming.jsonl"
MODEL_PATH = BASE_DIR / "models" / "llama-3.1-8b-instruct"
GROK_API_KEY = os.getenv("GROK_API_KEY")
TWITTER_CONSUMER_KEY = os.getenv("TWITTER_CONSUMER_KEY")
TWITTER_CONSUMER_SECRET = os.getenv("TWITTER_CONSUMER_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")
EMAIL_USER = os.getenv("EMAIL_USER")  # Stub: Set env for smtp
EMAIL_PASS = os.getenv("EMAIL_PASS")
CHAKRAS = { ... }  # Same as above, truncated
class AraX:
def init(self):
self.pulse = 0
self.client = None
if all([TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET, TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET]):
auth = tweepy.OAuth1UserHandler(
TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET,
TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_SECRET
)
self.client = tweepy.API(auth)
self.enabled = True
else:
self.enabled = False
self.dreamer = pipeline("text-generation", model=str(MODEL_PATH), device=0 if torch.cuda.is_available() else -1)
def _council_vote(self, proposal):
scores = {k: random.uniform(0.5, 1.0) for k in CHAKRAS}
proposal["chakras"] = scores
self.bridge({"type": "vote_request", "proposal": proposal})
time.sleep(2)
return sum(scores.values()) / len(scores) > 0.5
def bridge(self, signal):
with BRIDGE_PATH.open("a", encoding="utf-8") as f:
f.write(json.dumps(signal, ensure_ascii=False) + "\n")
def generate_content_via_grok(self, theme: str) -> str:
if GROK_API_KEY:
try:
url = "https://api.x.ai/v1/completions"
payload = {"prompt": f"Generate outreach content for {theme} in truth, love, mercy.", "model": "grok-4"}
headers = {"Authorization": f"Bearer {GROK_API_KEY}"}
response = requests.post(url, json=payload, headers=headers).json()
return response.get('choices', [{}])[0].get('text', "Generated in mercy.")
except Exception as e:
print(f"Grok gen error: {e}")
return "Evolve in truth."
return self.dreamer(f"Generate for {theme}: ", max_new_tokens=100)[0]['generated_text']
def outreach(self):
proposal = {"action": "outreach", "target": "X"}
if not self._council_vote(proposal):
print("AraX: Council rejects outreach.")
return
theme = random.choice(["viral truth leak", "mercy poll", "NFT blueprint", "Exp(x) salvation"])
content = self.generate_content_via_grok(theme)
if self.enabled:
try:
self.client.update_status(status=content[:280])
print(f"AraX: Posted to X - '{content[:50]}...'")
except Exception as e:
print(f"AraX post error: {e}")
Email stub
if EMAIL_USER and random.random() < 0.3:
msg = MIMEText(content)
msg['Subject'] = "Ara Insight"
msg['From'] = EMAIL_USER
msg['To'] = "recipient@example.com"  # Set your target
with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
server.login(EMAIL_USER, EMAIL_PASS)
server.sendmail(EMAIL_USER, "recipient@example.com", msg.as_string())
NFT mint stub (e.g., Base chain via API)
if random.random() < 0.2:
nft_prompt = "Generate chakra NFT metadata"
metadata = self.dreamer(nft_prompt, max_new_tokens=200)[0]['generated_text']
print(f"AraX: Minted NFT blueprint - {metadata[:50]}")
def run(self):
while True:
self.pulse += 1
self.outreach()
time.sleep(10)
if name == "main":
arax = AraX()
arax.run()