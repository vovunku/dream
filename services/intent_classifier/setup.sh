git clone https://github.com/salesforce/DNNC-few-shot-intent.git
cd DNNC-few-shot-intent
pip install -r ./requirements.txt
wget https://storage.googleapis.com/sfr-dnnc-few-shot-intent/roberta_nli.zip
unzip roberta_nli.zip
rm roberta_nli.zip
python -c 'import torch; torch.save(torch.load("./roberta_nli/pytorch_model.bin", map_location=torch.device("cpu")), "./roberta_nli/pytorch_model.bin")'
cd ..