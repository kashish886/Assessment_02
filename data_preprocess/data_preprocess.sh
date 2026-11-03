
set -eu

# Part One: ground concepts.
# Input: ../data/<dataset>/<split>; output: grounded/<split>.grounded.jsonl.
python data_preprocess-grounded.py



## Part Two: Retrieve facts
# Part Two: retrieve Wikipedia facts.
python data_preprocess-retrieval.py \
    --model_name_or_path facebook/contriever \
    --passages ../data/wiki/psgs_w100.tsv \
    --passages_embeddings "../data/wiki/wikipedia_embeddings/*" \
    --data_path ../data/ \
    --dataset creak \
    --output_dir ../data/creak/retrieval

python data_preprocess-retrieval.py \
    --model_name_or_path facebook/contriever \
    --passages ../data/wiki/psgs_w100.tsv \
    --passages_embeddings "../data/wiki/wikipedia_embeddings/*" \
    --data_path ../data/ \
    --dataset csqa2 \
    --output_dir ../data/csqa2/retrieval




## Part Three: Process and Load graph information
# Part Three: convert retrieved facts and concepts into graphs.
python data_preprocess-graph.py
