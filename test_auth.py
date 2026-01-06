#quick file to test that headless reuse works of the auth

from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    #set up the webpage with implicitly calls sync_playwright.start()
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(storage_state= "auth_state.json")
    page = context.new_page()

    #go to discord with the auth key 
    page.goto("https://discord.com/channels/@me")
    page.wait_for_timeout(3000)
    print("loaded page title: ", page.title())
    context.close()
    browser.close()
    
    


