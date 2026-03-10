import requests
import json
import sys

# Настройки API
API_KEY = "sk-Hou2Ci2dLLEZblDfTGM30PatSWUq9WhKD1YrHZnK64Boa"
MODEL = "gemini-3-flash"
BASE_URL = "http://127.0.0.1:8317/v1beta/models"

# Формируем полный URL для потоковой генерации контента
url = f"{BASE_URL}/{MODEL}:streamGenerateContent?key={API_KEY}"

headers = {
    "Content-Type": "application/json"
}

# Инициализируем историю сообщений (контекст диалога)
chat_history = []

print("\n=== Чат с Gemini запущен (Antigravity Proxy API) ===")
print("Введите 'выход', 'exit' или 'quit' для завершения.\n")

while True:
    try:
        user_input = input("Вы: ")
        
        # Проверка на команду выхода
        if user_input.strip().lower() in ['выход', 'exit', 'quit']:
            print("Завершение работы.")
            break
            
        # Пропускаем пустые вводы
        if not user_input.strip():
            continue

        # Добавляем сообщение пользователя в историю перед отправкой
        chat_history.append({"role": "user", "parts": [{"text": user_input}]})

        # Формируем тело запроса
        data = {
            "contents": chat_history
        }

        print("Gemini: ", end="", flush=True)

        # Переменная для сохранения полного текстового ответа от модели в текущем раунде
        full_assistant_response = ""

        # Отправляем POST запрос и включаем stream=True
        response = requests.post(url, headers=headers, json=data, stream=True)

        if response.status_code != 200:
            print(f"\n[Ошибка API {response.status_code}]: {response.text}")
            # Удаляем последний запрос пользователя из истории, раз он не прошел
            chat_history.pop()
            continue

        # Читаем ответ по кускам
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                
                # Обработка формата Server-Sent Events (SSE) или обычного JSON массива
                if decoded_line.startswith("data: "):
                    decoded_line = decoded_line[6:]
                
                if decoded_line.strip() == "[": continue
                if decoded_line.strip() == "]": continue
                if decoded_line.strip() == ",": continue

                try:
                    # Убираем запятую в конце JSON объекта, если она есть в массиве кусков
                    if decoded_line.endswith(","):
                        decoded_line = decoded_line[:-1]
                        
                    json_data = json.loads(decoded_line)
                    if "candidates" in json_data and len(json_data["candidates"]) > 0:
                        content = json_data["candidates"][0].get("content", {})
                        parts = content.get("parts", [])
                        if parts and "text" in parts[0]:
                            chunk_text = parts[0]["text"]
                            print(chunk_text, end="", flush=True)
                            full_assistant_response += chunk_text
                except json.JSONDecodeError:
                    # Игнорируем строки, которые не являются валидным JSON (например разделители массива)
                    pass

        print("\n")

        # Добавляем ответ ассистента в историю, чтобы модель помнила контекст разговора
        if full_assistant_response:
            chat_history.append({"role": "model", "parts": [{"text": full_assistant_response}]})

    except KeyboardInterrupt:
        print("\nЗавершение работы.")
        break
    except Exception as e:
        print(f"\n[Критическая ошибка]: {e}\n")
