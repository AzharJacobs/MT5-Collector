"""Quick MT5 connection diagnostic."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()
import MetaTrader5 as mt5

login = int(os.getenv("MT5_LOGIN", 0)) if os.getenv("MT5_LOGIN") else 0
password = os.getenv("MT5_PASSWORD", "")
server = os.getenv("MT5_SERVER", "")

print(f"Login : {login}")
print(f"Server: {server}")

print("\n--- Method 1: mt5.initialize() then mt5.login() ---")
ok = mt5.initialize()
print(f"  initialize() : {ok}  {mt5.last_error()}")
if ok:
    ok2 = mt5.login(login, password=password, server=server)
    print(f"  login()      : {ok2}  {mt5.last_error()}")
    if ok2:
        acct = mt5.account_info()
        print(f"  Account      : {acct.login}  balance={acct.balance}")
    mt5.shutdown()

print("\n--- Method 2: mt5.initialize(login=, password=, server=) ---")
ok = mt5.initialize(login=login, password=password, server=server)
print(f"  initialize() : {ok}  {mt5.last_error()}")
if ok:
    acct = mt5.account_info()
    name = acct.login if acct else "N/A"
    bal  = acct.balance if acct else "N/A"
    print(f"  Account      : {name}  balance={bal}")
    mt5.shutdown()

print("\n--- Method 3: mt5.initialize() no login (use existing session) ---")
ok = mt5.initialize()
print(f"  initialize() : {ok}  {mt5.last_error()}")
if ok:
    acct = mt5.account_info()
    if acct:
        print(f"  Account      : {acct.login}  balance={acct.balance}")
    else:
        print(f"  No account info — terminal not logged in")
    mt5.shutdown()
