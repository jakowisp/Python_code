#!/bin/bash
mkdir temp
cp /mnt/c/Users/ddilb/Downloads/Unix*/*.html temp
pushd temp
for FILE in `ls *.html`; do
    sed -i -e '/Skip to main content/d' $FILE
    sed -i -e '/title="Watch Icon"/d' $FILE
    sed -i -e '/title="Read Icon"/d' $FILE
    sed -i -e 's/<span class="aria-tooltip">Opens in new tab<\/span>//g' $FILE
done
popd
python3 bsparser.py example5.html

