from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

model = None
tokenizer = None
max_seq_len = 0

def initialize_model():
    global model, tokenizer, max_seq_len
    
    data = [
        "Life is short and the world is wide",
        "To travel is to live",
        "Jobs fill your pocket but adventures fill your soul",
        "Travel far enough to meet yourself",
        "Wander often wonder always",
        "Live life with no excuses and travel with no regrets",
        "Collect moments not things",
        "Don't listen to what they say go see",
        "Take only memories leave only footprints",
        "Travel is the only thing you buy that makes you richer",
        "Travel makes one modest you see what a tiny place you occupy in the world",
        "The journey not the arrival matters",
        "Travel brings power and love back into your life",
        "Wherever you go go with all your heart",
        "It feels good to be lost in the right direction",
        "Once a year go someplace you've never been before",
        "Better to see something once than hear about it a thousand times",
        "Traveling leaves you speechless then turns you into a storyteller",
        "Not all those who wander are lost",
        "Adventure is worthwhile",
        "Take every chance you get in life because some things only happen once",
        "A journey of a thousand miles begins with a single step",
        "Wherever you go leave a trail of kindness",
        "Travel is fatal to prejudice bigotry and narrow-mindedness",
        "The world is a book and those who do not travel read only one page",
        "You don't need magic to disappear all you need is a destination",
        "Travel far travel wide and travel often",
        "Go where you feel most alive",
        "Jobs fill your pocket but travel fills your soul",
        "Let's find some beautiful place to get lost",
        "Live your life by a compass not a clock",
        "You have to get lost before you can find yourself",
        "Travel is an investment in yourself",
        "We travel not to escape life but for life not to escape us",
        "The goal is to die with memories not dreams",
        "There's no time to be bored in a world as beautiful as this",
        "You can always make money you can't always make memories",
        "Don't count the days make the days count",
        "Sometimes the most scenic roads in life are the detours you didn't mean to take",
        "Travel teaches tolerance and understanding"
                "Life is short and the world is wide",
        "To travel is to live",
        "Jobs fill your pocket but adventures fill your soul",
        "Travel far enough to meet yourself",
        "Wander often wonder always",
        "Live life with no excuses and travel with no regrets",
        "Collect moments not things",
        "Don't listen to what they say go see",
        "Take only memories leave only footprints",
        "Travel is the only thing you buy that makes you richer",
        "Travel makes one modest you see what a tiny place you occupy in the world",
        "The journey not the arrival matters",
        "Travel brings power and love back into your life",
        "Wherever you go go with all your heart",
        "It feels good to be lost in the right direction",
        "Once a year go someplace you've never been before",
        "Better to see something once than hear about it a thousand times",
        "Traveling leaves you speechless then turns you into a storyteller",
        "Not all those who wander are lost",
        "Adventure is worthwhile",
        "Take every chance you get in life because some things only happen once",
        "A journey of a thousand miles begins with a single step",
        "Wherever you go leave a trail of kindness",
        "Travel is fatal to prejudice bigotry and narrow-mindedness",
        "The world is a book and those who do not travel read only one page",
        "You don't need magic to disappear all you need is a destination",
        "Travel far travel wide and travel often",
        "Go where you feel most alive",
        "Let's find some beautiful place to get lost",
        "Live your life by a compass not a clock",
        "You have to get lost before you can find yourself",
        "Travel is an investment in yourself",
        "We travel not to escape life but for life not to escape us",
        "The goal is to die with memories not dreams",
        "There's no time to be bored in a world as beautiful as this",
        "You can always make money you can't always make memories",
        "Don't count the days make the days count",
        "Sometimes the most scenic roads in life are the detours you didn't mean to take",
        "Travel teaches tolerance and understanding",
        
        "Mountains are calling and I must go",
        "The ocean whispers secrets to those who listen",
        "Forests hold ancient wisdom in their roots",
        "Desert winds carry stories across endless sands",
        "Rivers flow toward infinite possibilities",
        "Sunrise paints the sky with golden promises",
        "Moonlight illuminates paths unknown",
        "Stars guide wanderers through darkest nights",
        "Clouds drift carrying dreams across horizons",
        "Thunder echoes nature's powerful voice",
        "Lightning splits darkness with brilliant truth",
        "Rain washes away yesterday's worries",
        "Snow blankets earth in peaceful silence",
        "Seasons change teaching us about renewal",
        "Wildflowers bloom in unexpected places",
        "Birds sing melodies of freedom",
        "Waves crash against shores of possibility",
        "Valleys hold secrets waiting to be discovered",
        "Cliffs offer perspectives from great heights",
        "Caves whisper mysteries of ancient times",
        
        "Happiness is a journey not a destination",
        "Dreams are blueprints for tomorrow's reality",
        "Courage is not absence of fear but action despite it",
        "Wisdom comes from experience and reflection",
        "Kindness costs nothing but means everything",
        "Patience is the key to unlocking many doors",
        "Gratitude transforms ordinary moments into blessings",
        "Hope is the anchor that keeps us steady",
        "Faith moves mountains and opens new paths",
        "Love is the universal language of humanity",
        "Peace begins with understanding ourselves",
        "Joy is found in simple everyday miracles",
        "Strength grows through overcoming challenges",
        "Success is measured by lives we touch",
        "Forgiveness frees both giver and receiver",
        "Growth happens outside our comfort zones",
        "Change is the only constant in life",
        "Time is the most precious gift we possess",
        "Laughter heals wounds and lightens burdens",
        "Silence speaks volumes when words fail",
        
        "Excellence is not a skill but an attitude",
        "Perseverance turns obstacles into stepping stones",
        "Innovation requires thinking beyond conventional limits",
        "Leadership means inspiring others to achieve greatness",
        "Success follows those who never give up",
        "Creativity flourishes when imagination runs free",
        "Determination overcomes seemingly impossible odds",
        "Passion fuels the engine of achievement",
        "Vision transforms dreams into tangible realities",
        "Focus channels energy into powerful results",
        "Discipline bridges the gap between goals and accomplishment",
        "Opportunity disguises itself as hard work",
        "Progress requires stepping into the unknown",
        "Mastery comes from consistent daily practice",
        "Confidence builds through small daily victories",
        "Commitment turns promises into achievements",
        "Resilience transforms setbacks into comebacks",
        "Purpose gives meaning to every action",
        "Ambition drives us toward our highest potential",
        "Effort multiplied by time equals transformation",
        
        "Friendship is the foundation of lasting happiness",
        "Community creates strength through shared bonds",
        "Family provides roots while giving us wings",
        "Connection transcends distance and time",
        "Empathy builds bridges between different hearts",
        "Trust forms the bedrock of meaningful relationships",
        "Loyalty stands the test of time and trial",
        "Respect creates harmony in human interactions",
        "Compassion heals wounds and mends broken spirits",
        "Understanding grows through genuine listening",
        "Support lifts others when they cannot stand alone",
        "Encouragement plants seeds of future success",
        "Honesty builds relationships on solid foundations",
        "Generosity multiplies when shared with others",
        "Acceptance embraces differences and celebrates uniqueness",
        "Collaboration achieves what individuals cannot accomplish alone",
        "Communication opens doors to deeper understanding",
        "Appreciation acknowledges the value in everyone",
        "Celebration magnifies joy through shared experiences",
        "Unity creates strength through diversity",
        
        "Knowledge is power when applied with wisdom",
        "Learning never stops for those who stay curious",
        "Questions lead to discoveries and breakthroughs",
        "Experience teaches lessons that books cannot convey",
        "Mistakes are stepping stones to mastery",
        "Reflection transforms experience into wisdom",
        "Growth requires leaving familiar territory behind",
        "Awareness is the first step toward transformation",
        "Mindfulness brings peace to chaotic moments",
        "Balance creates harmony in all aspects of life",
        "Adaptation helps us thrive in changing circumstances",
        "Flexibility allows us to bend without breaking",
        "Openness invites new possibilities into our lives",
        "Curiosity drives exploration and discovery",
        "Imagination creates realities that don't yet exist",
        "Inspiration strikes when we least expect it",
        "Motivation comes from within our deepest desires",
        "Transformation happens one small step at a time",
        "Evolution requires embracing continuous change",
        "Potential awaits those brave enough to pursue it"
    ]
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)
    total_words = len(tokenizer.word_index) + 1
    
    input_sequences = []
    for line in data:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(2, len(token_list) + 1):
            n_gram_sequence = token_list[:i]
            input_sequences.append(n_gram_sequence)
    
    max_seq_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre'))
    
    X = input_sequences[:, :-1]
    y = tf.keras.utils.to_categorical(input_sequences[:, -1], num_classes=total_words)
    
    model = Sequential([
        Embedding(total_words, 64, input_length=max_seq_len - 1),
        LSTM(64),
        Dense(total_words, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print("Training model... Please wait.")
    model.fit(X, y, epochs=100, verbose=0)
    print("✅ Model training complete.")

def generate_text(seed_text, next_words=10):
    """Generate text using the trained model"""
    global model, tokenizer, max_seq_len
    
    original_seed = seed_text
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)[0]
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        if output_word:
            seed_text += " " + output_word
        else:
            break
    return seed_text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        seed_text = data.get('seed_text', '').strip()
        num_words = int(data.get('num_words', 10))
        
        if not seed_text:
            return jsonify({'error': 'Please enter a seed text'}), 400
        
        if num_words < 1 or num_words > 50:
            return jsonify({'error': 'Number of words must be between 1 and 50'}), 400
        
        generated_text = generate_text(seed_text, num_words)
        
        return jsonify({
            'original_seed': seed_text,
            'generated_text': generated_text,
            'success': True
        })
    
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    initialize_model()
    
    print("Starting Flask app...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=True)
    