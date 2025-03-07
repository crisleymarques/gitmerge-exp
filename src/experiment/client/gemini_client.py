from google import genai
from google.genai import types


client = genai.Client(api_key="AIzaSyAR_cJdtzgm654HTkUcl_JwM_pQNLOcf04")


def generate_content(model_name, prompt):
    return client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            # system_instruction=prompt,
            #top_k= 2,
            #top_p= 0.5,
            temperature= 0.3,
            seed=42,
        ),
    )