print("Hello, World! This is a test script.")
print("If you can see this, standard output is working correctly.")

# 引数処理も試してみる
import sys
if len(sys.argv) > 1:
    print(f"Argument received: {sys.argv[1]}") 