// for testing purpose only
// this code is not relevant ignore it

from google import genai
import os
import PIL.Image

image = PIL.Image.open('pics/test.jpeg')

client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["Return 1 if its single plant leaf. Else return 0. Give reason", image]
) 

print(response.text)
