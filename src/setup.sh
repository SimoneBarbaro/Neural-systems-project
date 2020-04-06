#!/bin/bash
curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=1AvU1UDSeg4uWfPA8JynjrrNnvg8oa0AZ" > /tmp/intermezzo.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/intermezzo.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > dataset.txt
pip install py-rouge
pip install rank_bm25
## Install sent2vec
## Installing sent2vec
git clone https://github.com/epfml/sent2vec.git
cd sent2vec &&  make && pip install .