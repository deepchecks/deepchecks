from urllib.parse import urlparse

import os
import selenium
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec


options = Options()
options.add_argument("--headless")
# following option is different for chrome driver
# options.set_preference("browser.download.dir", 1)
# options.set_preference("browser.download.lastDir", os.getcwd())
# options.set_preference("browser.helperApps.neverAsk.saveToDisk", "image/png")
# print(options.preferences)
# Don't put the path to geckodriver in the following. But the firefox executable
# must be in the path. If not, include the path to firefox, not geckodriver below.
driver = Firefox(options=options)

def fetch_thumbnails(links, crawled_urls):
    new_links = []
    for l in links:
        i = 0
        if l not in crawled_urls:
            driver.get(l)
            print("Crawling ", l)
            crawled_urls.append(l)        
            links_wes = driver.find_elements(By.XPATH, "//a[@class='reference internal']")
            for l1 in links_wes:
                l_dict = urlparse(l1.get_attribute('href'))
                if l_dict.path[0:4] != '/api':
                    link = l_dict.scheme + "://" + l_dict.netloc + l_dict.path
                    if link not in new_links:
                        new_links.append(link)
                        # we are looking for plot_ html files because those contain plots
            if l.rsplit('/', 1).pop()[0:5] == 'plot_' and i == 0:
                print(f"{l} is interesting; so we try to download image")
                i += 1
                try:
                    # this access is supposed to be on localhost so 10 seconds should be enough
                    download_png_link = WebDriverWait(driver, 10).until(
                        ec.visibility_of_element_located(
                            (By.XPATH, "//a[@data-title='Download plot as a png']")
                        )
                    )
                    download_png_link.click()
                    print(os.stat(os.path.expanduser('~') + '/Downloads/newplot.png'))
                except selenium.common.exceptions.TimeoutException as e:
                    # we ignore this as not all plots have figures so it will cause
                    # timeout exception
                    print(f"Timed out for {l}")
                    pass



    for l in new_links:
        if l not in links:
            links.append(l)

    return len(new_links)

def crawl():
    # there are no plots here so we are not interested
    print("Crawling ", "http://localhost/index.html")
    driver.get('http://localhost/index.html')
    crawled_urls = ['http://localhost/index.html']
    links = []
    links_wes = driver.find_elements(By.XPATH, "//a[@class='reference internal']")
    for l in links_wes:
        l_dict = urlparse(l.get_attribute('href'))
        if l_dict.path[0:4] != '/api':
            link = l_dict.scheme + "://" + l_dict.netloc + l_dict.path
            if link not in links:
                links.append(link)
    
    no_of_new_links = -1
    while no_of_new_links != 0:
        no_of_new_links = fetch_thumbnails(links, crawled_urls)


if __name__ == "__main__":
    crawl()
    driver.quit()
