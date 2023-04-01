import glob
import logging

import esprima
import tqdm

from nodes.main import Node


def load_corpus(path: str) -> list[Node]:
    files = glob.glob(f"{path}/*.js")
    logging.info(f"Found {len(files)} files in corpus")
    corpus = []

    for file in tqdm.tqdm(files):
        with open(file, "r") as f:
            code = f.read()

        try:
            ast = esprima.parseScript(code, tolerant=True, jsx=True)
        except esprima.error_handler.Error:  # type: ignore
            # logging.warning(f"Failed to parse {file}")
            continue

        corpus.append(ast)

    logging.info(f"Loaded {len(corpus)}/{len(files)} files from corpus")

    return corpus
