git clone https://github.com/salesforce/DNNC-few-shot-intent.git
cd DNNC-few-shot-intent
pip install -r ./requirements.txt
wget https://storage.googleapis.com/sfr-dnnc-few-shot-intent/roberta_nli.zip
unzip roberta_nli.zip
rm roberta_nli.zip
cd ..