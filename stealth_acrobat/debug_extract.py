from celery_worker import extract_questions_task
import time

print("Starting debug extraction...")
start = time.time()
res = extract_questions_task("1. What is 2+2? A) 3 B) 4")
print(f"Finished in {time.time() - start} seconds. Result:", res)
