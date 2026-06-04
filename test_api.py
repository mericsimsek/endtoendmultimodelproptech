import google.generativeai as genai


genai.configure(api_key="AIzaSyAodd_5rBqE9dcqOHDlIyoHwwgF-_P0lCQ")

print("Senin API Anahtarınla Çalışan Modeller Şunlar:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)