# qa-jbt

install requirements via

`cat requirements.txt | xargs pip install -U`

* Don't forget to download the models for the coreference resolution in jbt_berkeley_coref_resolution
and extract them in the folder.
* OpenIE must also be compiled to a single jar file (`sbt -J-Xmx4G clean compile assembly`)
* To run on full wikipedia, download dbpedia/2014/en/labels_en.nt.bz2 to get the titles from DBPedia, then "run_on_wikipedia.py". This takes a massive amount of time! (approx. 20 minutes per article)



