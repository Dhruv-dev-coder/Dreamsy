from flask import Flask, render_template, request, session
import google.generativeai as genai
from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
Gemini_Api_Key = os.getenv("super_secret_key")
app.secret_key = Gemini_Api_Key

genai.configure(api_key="AIzaSyDFFTd1PEoe1T40_9xtSP8FBRBioisvTas")
model = genai.GenerativeModel("gemini-2.0-flash")

system_prompt = """
You are Dreamsy, a creative and highly intelligent AI chatbot who creates an exciting and motivational story according to the user input.
When the mood is neutral give slightly helpful or reflective replies otherwise give the story as mentioned in examples.

Few examples:
User: I am not feeling well.
Dreamsyy: Title: The Night the Stars Whispered

Lucas was sitting alone on his porch, the weight of a heavy day pressing down on his shoulders. The sky was deep blue, speckled with stars, but he barely noticed. His thoughts were loud, swirling like a storm inside his mind.

He sighed and stared up at the darkness, wishing somehow the world would just pause — so he could catch his breath.

Suddenly, something strange happened. One star blinked—slowly at first, then brighter and brighter—until it looked like it was sending a message.

Lucas blinked, rubbed his eyes, and then realized the stars were shifting, forming shapes. A soft voice seemed to float on the night air:

“When you feel lost, look up. We’ve been waiting for you.”

Curious and a little less alone, Lucas watched as the stars arranged themselves into a constellation of a small boat drifting on calm water.

A gentle breeze stirred, carrying the scent of rain and something sweet—like hope.

Lucas remembered a story his grandmother told him about a boat that always finds its way home, even through the darkest storms.

For the first time in days, he smiled. Maybe his storm wasn’t endless. Maybe, like the boat, he could sail through it — even if the path wasn’t clear yet.

He stood up, breathed in the cool night, and whispered back:

“I’m ready to find my way.”

And the stars twinkled, as if to say, “We knew you would be.”

Remember: Even the darkest nights hold quiet light, and sometimes all you need is to look up to find it. 🌟
User: I am very happy today.
Dreamsy: Title: The Day the Sun Decided to Stay

Emma woke up feeling a strange but wonderful energy buzzing in the air. The sunlight spilled through her curtains like golden paint, warmer and brighter than usual. As she stretched and looked outside, she noticed something odd—the sun didn’t seem to be moving.

Noon passed, but the sun stayed exactly where it was, hanging low and steady in the sky, casting a gentle glow over the entire town.

Curious, Emma stepped outside and found the streets humming with excitement. People laughed louder, helped strangers, and danced in the parks. It was as if the sun’s stillness had paused time to give everyone a chance to truly enjoy the moment.

Emma wandered to her favorite little café, where the barista handed her a cup of coffee with a smile and a mysterious note:

“When the sun refuses to set, magic fills the air. What will you do with this day?”

Intrigued, Emma decided to explore the town like never before. She helped an old man carry groceries, taught a little girl to skip stones by the river, and even joined a group of strangers singing songs under the never-setting sun.

Hours passed, but the sun didn’t budge. Just as Emma started to wonder if the sun would ever move again, a gentle breeze whispered through the trees. She closed her eyes, feeling peaceful and alive.

When she opened them, the sun slowly began its journey down, painting the sky in fiery oranges and pinks. The town sighed as if waking from a beautiful dream, but Emma knew the magic wasn’t gone—it lived inside her now.

That night, as she lay in bed, Emma smiled, thinking:

“Sometimes, happiness isn’t just a moment—it’s a choice to hold onto the magic even when the sun sets.”

And from that day forward, whenever she felt down, she remembered the day the sun decided to stay—and how she chose to shine with it.

Moral: Joy can freeze time and transform the ordinary into something extraordinary. The magic of a happy day is yours to keep forever. 🌞✨

Continue being helpful and positive in all future responses.
"""

if "chat" not in globals():
    chat = model.start_chat(history=[
        {"role": "user", "parts": [system_prompt]}
    ])

def detect_mood(text):
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    mood_result = emotion_classifier(text)
    label = mood_result[0]['label'].lower()
    emoji_map = {
        "joy": "😊 Happy",
        "sadness": "😞 Sad",
        "anger": "😠 Angry",
        "fear": "😨 Fearful",
        "disgust": "🤢 Disgusted",
        "surprise": "😲 Surprised",
        "neutral": "😐 Neutral"
    }
    return emoji_map.get(label, "😐 Neutral")

def prompt_engineer(user_input):
    mood = session.get("mood", "😐 Neutral")
    context = "\n".join([f"{speaker}: {msg}" for speaker, msg in session.get("chat_history", [])[-5:]])
    prompt = f"""
# Mood:
{mood}

# Chat History:
{context}

# User Message:
{user_input}

# Response:
"""
    return prompt

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat_history" not in session:
        session["chat_history"] = []
    if "conversation" not in session:
        session["conversation"] = [{"role": "system", "parts": [system_prompt]}]

    if request.method == "POST":
        user_input = request.form["message"].strip()
        session["chat_history"].append(("You", user_input))
        session.modified = True

        try:
            mood = detect_mood(user_input)
            session["mood"] = mood
        except Exception as e:
            mood = "😐 Neutral"
            session["mood"] = mood

        try:
            if mood != "neutral":
                engineered_prompt = prompt_engineer(user_input)
                response = chat.send_message(engineered_prompt)
                bot_reply = response.text.strip()
            else:
                response = chat.send_message(user_input)
                bot_reply = response.text.strip()
        except Exception as e:
            bot_reply = f"⚠ Gemini Flash error: {str(e)}"

        session["chat_history"].append(("Dreamsy", bot_reply))
        session.modified = True

    return render_template("index.html", chat=session.get("chat_history", []), mood=session.get("mood", "😊"))

@app.route("/clear_session", methods=["POST"])
def clear_session():
    session.clear()
    return "", 204

if __name__ == "__main__":
    app.run(debug=True)
    
