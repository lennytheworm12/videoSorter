from playwright.sync_api import sync_playwright
import pathlib 

STATE_PATH = pathlib.Path("auth_state.json")

#this file prompts us to log in for session cookies and then store then in auth_state to later be used 
with sync_playwright() as play:
    browser = play.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()

    page.goto("https://discord.com/login")

    print("log in manually throught the window")
    print("when done logging in and can see servers come back and press enter")
    input()

    #save cookies and localstorage to reuse login
    context.storage_state(path=str(STATE_PATH))
    print(f"(saved auth state to {STATE_PATH.resolve()}")
    
    context.close()
    browser.close()
