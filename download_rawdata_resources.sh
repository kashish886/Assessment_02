set -eu

# Download ConceptNet
mkdir -p data/
mkdir -p data/cpnet/
echo "Downloading ConceptNet..."
wget -nc -P data/cpnet/ https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
cd data/cpnet/
if [ -f conceptnet-assertions-5.6.0.csv.gz ] && [ ! -f conceptnet-assertions-5.6.0.csv ]; then
	gzip -df conceptnet-assertions-5.6.0.csv.gz
fi
# download ConceptNet entity embedding
echo "Downloading ConceptNet entity embedding..."
wget -nc https://csr.s3-us-west-1.amazonaws.com/tzw.ent.npy
cd ../../


mkdir -p data/wiki
cd data/wiki
# Download Wikipedia passages
echo "Downloading Wikipedia passages"
if [ ! -f psgs_w100.tsv.gz ] && [ ! -f psgs_w100.tsv ]; then
	wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
fi
echo "Decompressing Wikipedia passages"
if [ ! -f psgs_w100.tsv ]; then
	gzip -df psgs_w100.tsv.gz
fi

# Generate passages embeddings
## Here we directly download passages embeddings pre-computed with Contriever or Contriever-msmarco
### wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever/wikipedia_embeddings.tar
wget -nc https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar
tar -xvf wikipedia_embeddings.tar
cd ../../


# download CREAK dataset
mkdir -p data/creak/

# create output folders for CREAK
mkdir -p data/creak/grounded/
mkdir -p data/creak/retrieval/
mkdir -p data/creak/graph/


# download CSQA2 dataset
mkdir -p data/csqa2/

# create output folders for CSQA2
mkdir -p data/csqa2/grounded/
mkdir -p data/csqa2/retrieval/
mkdir -p data/csqa2/graph/


