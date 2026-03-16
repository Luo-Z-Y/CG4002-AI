import requests
import json
import os
import base64
import random

# Your API Key
API_KEY = "AIzaSyDOljft5i49RMe9z3e4n9g9Xapt2fmUTMA"
URL = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={API_KEY}"

def generate_pokemon_voices():
    # Define the Pokemon and their target folders
    pokemons = ["Pikachu", "Charizard", "Bulbasaur"]
    
    # Reliable voice list (50% Singaporean Standard, 50% International)
    sg_voices = ["en-SG-Standard-A", "en-SG-Standard-B"]
    intl_voices = [
        "en-US-Neural2-A", "en-US-Neural2-D", "en-GB-Neural2-A", 
        "en-AU-Neural2-A", "en-IN-Wavenet-D", "en-US-Wavenet-C"
    ]

    for pokemon in pokemons:
        output_dir = pokemon.lower() # creates folders 'pikachu', 'charizard', 'bulbasaur'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"\n--- Generating 500 files for {pokemon} ---")

        for i in range(1, 501):
            # Select Voice (50% Singaporean)
            is_sg = random.random() < 0.5
            if is_sg:
                voice_name = random.choice(sg_voices)
                lang_code = "en-SG"
            else:
                voice_name = random.choice(intl_voices)
                lang_code = voice_name[:5]

            # High Variation Parameters
            pitch = random.uniform(-10.0, 10.0)
            rate = random.uniform(0.65, 1.45)
            volume = random.uniform(-5.0, 3.0)
            
            # Resolution Variation (8kHz lo-fi to 48kHz hi-fi)
            sample_rate = random.choice([8000, 11025, 16000, 22050, 44100, 48000])

            payload = {
                "input": {"text": pokemon},
                "voice": {
                    "languageCode": lang_code,
                    "name": voice_name
                },
                "audioConfig": {
                    "audioEncoding": "M4A",
                    "pitch": pitch,
                    "speakingRate": rate,
                    "volumeGainDb": volume,
                    "sampleRateHertz": sample_rate
                }
            }

            response = requests.post(URL, json=payload)
            
            if response.status_code == 200:
                audio_data = response.json()['audioContent']
                filename = os.path.join(output_dir, f"{pokemon.lower()}_{i:03d}.m4a")
                with open(filename, "wb") as f:
                    f.write(base64.b64decode(audio_data))
            else:
                # Simple fallback to a guaranteed US voice if the chosen one fails
                payload["voice"]["name"] = "en-US-Standard-C"
                payload["voice"]["languageCode"] = "en-US"
                retry_res = requests.post(URL, json=payload)
                if retry_res.status_code == 200:
                    audio_data = retry_res.json()['audioContent']
                    filename = os.path.join(output_dir, f"{pokemon.lower()}_{i:03d}.m4a")
                    with open(filename, "wb") as f:
                        f.write(base64.b64decode(audio_data))

            if i % 25 == 0:
                print(f"  {pokemon}: {i}/500 files done...")

    print(f"\nAll done! Folders 'pikachu', 'charizard', and 'bulbasaur' are ready.")

if __name__ == "__main__":
    generate_pokemon_voices()