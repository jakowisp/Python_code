"""
bsparser is a helper tool to take scrapped html pages for WGU coarsesr
and turn them into a single document
"""
import sys
import copy
import os
from os.path import isfile, join
from bs4 import BeautifulSoup


with open(sys.argv[1], encoding="utf-8") as fp:
    soup = BeautifulSoup(fp, 'html.parser')



# Accessing Elements
print("Title of the Page:", soup.title.text)  # Access the title element
print("Heading:", soup.h1.text)  # Access the heading element

# Accessing List Items
items = soup.find_all('button')  # Find all list items within the ul
print("<=======Mapping strings to hash===========>")
mapvals = {}
for item in items:
    datatestid = item.get('data-testid')
    if datatestid is not None:
        mapvals[str(item.get('data-testid')).split('block@', maxsplit=1)[-1]]=str(item.text)
print("<============Renaming cached names and prepending string to body==========")
MYPATH='temp'
onlyfiles = [f for f in os.listdir(MYPATH) if isfile(join(MYPATH, f))]
changedFiles={}
"""
for file in onlyfiles:
    index=str(file).split('block@')[-1].strip('.html')
    header=mapvals[index].strip()
    newname = header.replace('/','_').replace(':','').replace('.','_').replace(' ','_')+".html"
    print(str(file) + ":" +mapvals[index]+" : "+newname)
    with open("temp/"+file) as fp:
        souptemp = BeautifulSoup(fp, 'html.parser')
    souptemp.title.string = mapvals[str(file).split('block@')[-1].strip('.html')]
    body = souptemp.find('div')

    new_content = souptemp.new_tag('div')
    new_content.string = souptemp.title.text

    body.insert_before(new_content)
    html = souptemp.prettify("utf-8")
    with open("temp/"+newname, "wb") as filehtml:
        filehtml.write(html)
    changedFiles[index]=newname
"""
for file in onlyfiles:
    index=str(file).split('block@', maxsplit=1)[-1].strip('.html')
    changedFiles[index]=file
    print(str(file) + ":" +mapvals[index] )

print("<============= Begin Merge =============>")

first=None
cnt=0
for key in mapvals.keys():
    if key in changedFiles.keys():
        print(mapvals[key])
        if first is None:
            with open("temp/"+changedFiles[key], encoding="utf-8") as fp:
                first = BeautifulSoup(fp, 'html.parser')
            body=first.find('body')
            new_content = first.new_tag('div')
            new_content.string = mapvals[key]
            body.insert_before(new_content)
        else:
            with open("temp/"+changedFiles[key], encoding="utf-8") as fp:
                temp = BeautifulSoup(fp, 'html.parser')
            new_content = first.new_tag('div')
            new_content.string = mapvals[key]
            body.append(new_content)
            for element in temp.body:
                body.append(copy.deepcopy(element))
                cnt+=1
print(cnt)
print("<==========writing file=========>")
html = first.prettify("utf-8")
with open("temp/combined.html", "wb") as filehtml:
    filehtml.write(html)
