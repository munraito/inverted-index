from textwrap import dedent
import argparse

import io
import pytest

from inverted_index import *


# DATASET_BIG_FPATH = "data/wikipedia.sample"
# DATASET_SMALL_FPATH = "data/small_wikipedia.sample"
# DATASET_TINY_FPATH = "data/tiny_wikipedia.sample"
DATASET_TINY_STR = dedent("""\
    123\tsome words A_word and nothing
    2\tsome word B_word in this document
    5\tfamous_phrases to be or not to be
    37\tall words such as A_word and B_word\tare here
    """)


@pytest.fixture
def tiny_dataset_fio(tmpdir):
    dataset_fio = tmpdir.join("dataset.txt")
    dataset_fio.write(DATASET_TINY_STR)
    return dataset_fio


def test_can_load_documents(tiny_dataset_fio):
    documents = load_documents(tiny_dataset_fio)
    etalon_documents = {
        123: "some words A_word and nothing",
        2: "some word B_word in this document",
        5: "famous_phrases to be or not to be",
        37: "all words such as A_word and B_word are here"
    }
    assert etalon_documents == documents, (
        "load_documents incorrectly loaded dataset"
    )


@pytest.mark.parametrize(
    "query, etalon_answer",
    [
        pytest.param(["A_word"], [123, 37], id="A_word"),
        pytest.param(["B_word"], [2, 37], id="B_word"),
        pytest.param(["A_word", "B_word"], [37], id="both words"),
        pytest.param(["word_does_not_exist"], [], id="word does not exist"),
        pytest.param([], [], id="empty input"),
    ],
)
def test_query_inverted_index_intersect_results(tiny_dataset_fio, query, etalon_answer):
    documents = load_documents(tiny_dataset_fio)
    tiny_inverted_index = build_inverted_index(documents)
    answer = tiny_inverted_index.query(query)
    assert sorted(answer) == sorted(etalon_answer), (
        f"Expected answer is {etalon_answer}, but you got {answer}"
    )


@pytest.mark.skip
def test_can_load_wikipedia_sample():
    documents = load_documents(DATASET_BIG_FPATH)
    assert len(documents) == 4100, (
        "incorrectly loaded wiki sample"
    )


@pytest.fixture
def wikipedia_documents(tiny_dataset_fio):
    documents = load_documents(tiny_dataset_fio)  # DATASET_BIG_FPATH
    return documents


# @pytest.fixture
# def small_sample_wikipedia_documents():
#     documents = load_documents(DATASET_SMALL_FPATH)
#     return documents


def test_can_build_and_query_inverted_index(wikipedia_documents):
    wikipedia_inverted_index = build_inverted_index(wikipedia_documents)
    doc_ids = wikipedia_inverted_index.query(["wikipedia"])
    assert isinstance(doc_ids, list), (
        "inverted index should return list"
    )


@pytest.fixture
def wikipedia_inverted_index(wikipedia_documents):
    wikipedia_inverted_index = build_inverted_index(wikipedia_documents)
    return wikipedia_inverted_index


# @pytest.fixture
# def small_wikipedia_inverted_index(small_sample_wikipedia_documents):
#     wikipedia_inverted_index = build_inverted_index(small_sample_wikipedia_documents)
#     return wikipedia_inverted_index


def test_can_dump_and_load_inverted_index(tmpdir, wikipedia_inverted_index):
    index_fio = tmpdir.join('index.dump')
    wikipedia_inverted_index.dump(index_fio)
    loaded_inverted_index = InvertedIndex.load(index_fio)
    assert wikipedia_inverted_index == loaded_inverted_index, (
        "load should return the same inverted index"
    )


def test_callback_build_can_load_documents(tmpdir, capsys, tiny_dataset_fio):
    index_filepath = tmpdir.join('inverted.index')
    build_arguments = argparse.Namespace(
        dataset_filepath=tiny_dataset_fio,
        index_filepath=index_filepath,
    )
    callback_build(build_arguments)
    captured = capsys.readouterr()
    assert "building inverted index" not in captured.out


def test_can_encode_filetype(monkeypatch):
    monkeypatch.setattr('sys.stdin', TextIOWrapper(io.BytesIO(b"test"), encoding="utf-8"))
    encoded_type = EncodedFileType("r", encoding="utf-8")
    text_io = encoded_type('-')
    assert TextIOWrapper == text_io.__class__, (
        "encoded file type should work"
    )


def test_callback_query_can_process_all_queries_from_files(tmpdir, capsys, wikipedia_inverted_index):
    index_fio = tmpdir.join('index.dump')
    wikipedia_inverted_index.dump(index_fio)
    query_str = dedent("""\
            A_word
            B_word
            A_word B_word
            word_does_not_exists
        """)
    query_fio = tmpdir.join("queries.txt")
    query_fio.write(query_str)
    query_file = open(query_fio)
    query_arguments = argparse.Namespace(
        index_filepath=index_fio,
        query_file=query_file,
        query=None
    )
    callback_query(query_arguments)
    query_file.close()
    captured = capsys.readouterr()
    assert "loading inverted index" not in captured.out
    assert "loading inverted index" in captured.err
    assert "query inverted index with request" not in captured.out
    assert "query inverted index with request" in captured.err
    assert "123,37" in captured.out
    assert "123,37" not in captured.err


def test_callback_query_works_for_individual_query(tmpdir, wikipedia_inverted_index, capsys):
    index_fio = tmpdir.join("index.dump")
    wikipedia_inverted_index.dump(index_fio)
    query_arguments = argparse.Namespace(
        query=[["A_word"]],
        index_filepath=index_fio,
        query_file=None
    )
    callback_query(query_arguments)
    captured = capsys.readouterr()
    assert "123,37" in captured.out