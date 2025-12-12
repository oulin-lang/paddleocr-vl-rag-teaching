import httpx
import asyncio
import json
import sys
import time

BASE_URL = "http://127.0.0.1:8901"

async def run_chat_test(case_name, session_id, question, net_search):
    print(f"\n{'='*50}")
    print(f"TEST CASE: {case_name}")
    print(f"Session ID: {session_id}")
    print(f"Question: {question}")
    print(f"Net Search: {net_search}")
    print(f"{'='*50}")
    
    url = f"{BASE_URL}/sessions/{session_id}/chat"
    payload = {
        "question": question, 
        "net_search": net_search
    }
    
    start_time = time.time()
    response_content = ""
    chunk_count = 0
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                print(f"Response status: {response.status_code}")
                if response.status_code != 200:
                    print(f"Error: {await response.read()}")
                    return False

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[PING]":
                            pass # Ignore pings in output to keep it clean
                        elif data_str.startswith("[END]"):
                            print(f"\n[END] Received. Duration info: {data_str}")
                        elif data_str == "[CANCELLED]":
                            print("\n[CANCELLED] Received")
                        elif data_str.startswith("ERROR"):
                            print(f"\n[ERROR] Received: {data_str}")
                            return False
                        else:
                            try:
                                data = json.loads(data_str)
                                content = data.get("content", "")
                                if content:
                                    print(content, end="", flush=True)
                                    response_content += content
                                    chunk_count += 1
                            except json.JSONDecodeError:
                                print(f"\n[RAW] {data_str}")
    except Exception as e:
        print(f"\nEXCEPTION: {e}")
        return False
    
    duration = time.time() - start_time
    print(f"\n\nTest finished in {duration:.2f}s")
    print(f"Total chunks received: {chunk_count}")
    print(f"Response length: {len(response_content)}")
    
    if len(response_content) > 0:
        print(" [PASS] Test PASSED: Content received.")
        return True
    else:
        print(" [FAIL] Test FAILED: No content received.")
        return False

async def main():
    print("Starting E2E Test Suite for Agent RAG Chat...")
    
    # Test Case 1: Pure Local Search (Simple calculation/knowledge)
    success_1 = await run_chat_test(
        case_name="Pure Local Search (net_search=False)",
        session_id=f"test_local_{int(time.time())}",
        question="Hello, what is 25 * 4?",
        net_search=False
    )
    
    # Test Case 2: Pure Net Search (Specific real-time info)
    success_2 = await run_chat_test(
        case_name="Pure Net Search (net_search=True)",
        session_id=f"test_net_{int(time.time())}",
        question="What is the current stock price of Tesla (TSLA)?",
        net_search=True
    )
    
    # Test Case 3: Mixed/Complex Search (Requires finding info and summarizing)
    success_3 = await run_chat_test(
        case_name="Mixed/Complex Search (net_search=True)",
        session_id=f"test_mixed_{int(time.time())}",
        question="Who is the current CEO of Microsoft and when did they take office?",
        net_search=True
    )
    
    print("\n" + "="*50)
    print("TEST SUITE REPORT")
    print("="*50)
    print(f"Case 1 (Local): {'[PASS]' if success_1 else '[FAIL]'}")
    print(f"Case 2 (Net):   {'[PASS]' if success_2 else '[FAIL]'}")
    print(f"Case 3 (Mixed): {'[PASS]' if success_3 else '[FAIL]'}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
