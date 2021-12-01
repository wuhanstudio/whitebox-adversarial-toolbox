#!/usr/bin/env python3
from what import __version__

import shutil
from pathlib import Path

import pdoc

here = Path(__file__).parent

modules = [
    "what",
]

print('Building docs for v{}'.format(__version__))

# Render pdoc's documentation into docs/api...
pdoc.render.configure(
    footer_text="White-box Adversarial Toolbox v" + str(__version__),
    search=True,
    logo="https://what.wuhanstudio.uk/images/what.png",
    logo_link="https://github.com/wuhanstudio/whitebox-adversarial-toolbox")

pdoc.pdoc(*modules, output_directory = here / "docs" / str("v" + str(__version__)) )


from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Encountered a start tag:", tag)

    def handle_endtag(self, tag):
        print("Encountered an end tag :", tag)

    def handle_data(self, data):
        print("Encountered some data  :", data)

print('Generating index.html')

homepage ='docs/index.html' 
with open(homepage, 'w') as filetowrite:
    filetowrite.write('<!DOCTYPE html><html><head><script type="text/javascript">')
    filetowrite.write('window.location.href = window.location.href + "v')
    filetowrite.write(str(__version__))
    filetowrite.write('" + "/what.html";')
    filetowrite.write('</script></head><body></body></html>')
