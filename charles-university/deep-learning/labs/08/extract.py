import gzip
import shutil
import os
import glob
from bs4 import BeautifulSoup

datadirs = ['dtest', 'etest'] + ['train-' + str(i+1) for i in range(8)]
datadirs = [os.path.join('pdt3.5-mw', datadir) for datadir in datadirs]

for datadir in datadirs:
    files = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]

    for hgx in glob.glob(os.path.join(datadir, '*.xml')):
        os.remove(hgx)

    for file in files:
        if not file.endswith('.m.gz'):
            continue

        extracted = file.rstrip('.gz') + '.xml'

        # Extract files
        with gzip.open(os.path.join(datadir, file), 'rb') as gzfile:
            with open(os.path.join(datadir, extracted), 'wb') as xmlfile:
                shutil.copyfileobj(gzfile, xmlfile)

        with open(os.path.join(datadir, extracted), 'rb') as xmlfile:
            data = BeautifulSoup(xmlfile, features="xml")

            ss = data.mdata.findAll('s')

            for s in ss:
                ms = s.findAll('m')

                for m in ms:
                    form, lemma, tag = m.form.string, m.lemma.string, m.tag.string
                    print(form, lemma, tag, sep='\t')
                print()
