import asyncio
from google.genai import types
from google.genai import Client
import dotenv

dotenv.load_dotenv()

safety_settings = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_NONE,
    )
]


async def main():  # 建立一個 async 函式
    client = Client()  # 實例化 AsyncClient
    response = await client.aio.models.generate_content(  # 使用 await
        model='gemini-2.5-flash-lite',
        contents='李开复的“五秒准则”',
        config=types.GenerateContentConfig(
            system_instruction='你是一位AI領域的專家',
            temperature=0.0,
            max_output_tokens=1000,
            safety_settings=safety_settings,
        ),
    )
    print(response.text)


if __name__ == "__main__":
    asyncio.run(main())  # 執行 async 函式
