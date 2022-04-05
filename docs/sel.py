import os
import shutil
from urllib.parse import urlparse

import selenium
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec


options = Options()
options.add_argument("--headless")
driver = Firefox(options=options)


def fetch_thumbnails(links, crawled_urls):
    """
    This crawls the links and downloads first plotly plot by clicking
    the "Download as png" from plotly toolbar to default folder which is
    typically ~/Downloads. The default filename is newplot.png and
    replaces the thumbnail file.

    If you want a custom thumbnail, read the instructions at
    https://sphinx-gallery.github.io/stable/configuration.html#providing-thumbnail
    Basically you will need following below docstring in the plot file:
    # sphinx_gallery_thumbnail_path = '_static/demo.png'
    This path is relative to conf.py
    """
    new_links = []
    for l in links:
        i = 0
        if l not in crawled_urls:
            driver.get(l)
            print("Crawling ", l)
            crawled_urls.append(l)
            links_wes = driver.find_elements(
                By.XPATH, "//a[@class='reference internal']"
            )
            for l1 in links_wes:
                l_dict = urlparse(l1.get_attribute("href"))
                if l_dict.path[0:4] != "/api":
                    link = l_dict.scheme + "://" + l_dict.netloc + l_dict.path
                    if link not in new_links:
                        new_links.append(link)
                        # we are looking for plot_ html files because those contain plots
            # we are interested in only those files which are named
            # with prefix /plot
            if l.rsplit("/", 1).pop()[0:5] == "plot_" and i == 0:
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
                    source = os.path.expanduser("~") + "/Downloads/newplot.png"
                    # we use the format which sphinx galler uses for filenames
                    filename = (
                        "sphx_glr_"
                        + l.rsplit("/", 1).pop().split(".")[0]
                        + "_thumb.png"
                    )
                    destination = "docs/build/html/_images/" + filename
                    shutil.move(source, destination)
                except selenium.common.exceptions.TimeoutException as e:
                    # we ignore this as not all plots have figures so it will cause
                    # timeout exception
                    print(f"Timed out for {l}")
                    pass
                except FileNotFoundError as e:
                    # if plot does not have image it will cause this exception
                    # so we pass
                    print(
                        "Seems plot does not have an image. Perhaps you would "
                        "want a custom thumbnail"
                    )
                    pass

    for l in new_links:
        if l not in links:
            links.append(l)

    return len(new_links)


def crawl():
    # there are no plots here so we are not interested
    print("Crawling ", "http://localhost/index.html")
    driver.get("http://localhost/index.html")
    crawled_urls = ["http://localhost/index.html"]
    links = []
    # find all the links in homepage
    links_wes = driver.find_elements(By.XPATH, "//a[@class='reference internal']")
    for l in links_wes:
        l_dict = urlparse(l.get_attribute("href"))
        # we are not interested in API reference
        if l_dict.path[0:4] != "/api":
            link = l_dict.scheme + "://" + l_dict.netloc + l_dict.path
            if link not in links:
                links.append(link)

    no_of_new_links = -1
    while no_of_new_links != 0:
        no_of_new_links = fetch_thumbnails(links, crawled_urls)


if __name__ == "__main__":
    crawl()
    driver.quit()
