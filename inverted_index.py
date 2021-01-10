#!/Users/munraito/opt/anaconda3/envs/bdt-python-course/bin/python
from io import TextIOWrapper
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, FileType, ArgumentTypeError
from collections import defaultdict
import struct

DEFAULT_DATASET_PATH = "data/small_wikipedia.sample"
DEFAULT_INVERTED_INDEX_STORE_PATH = "small_inverted.index"


class EncodedFileType(FileType):
    """hotfix for FileType class"""
    def __call__(self, string):
        """hotfix for FileType"""
        if string == '-':
            if 'r' in self._mode:
                stdin = TextIOWrapper(sys.stdin.buffer, encoding=self._encoding)
                return stdin
            elif 'w' in self._mode:
                stdout = TextIOWrapper(sys.stdout.buffer, encoding=self._encoding)
                return stdout
            else:
                msg = 'argument "-" with mode %r' % self._mode
                raise ValueError(msg)
        try:
            return open(string, self._mode, self._bufsize, self._encoding,
                        self._errors)
        except OSError as err:
            message = "can't open '%s': %s"
            raise ArgumentTypeError(message % (string, err))


class InvertedIndex:
    """class for building and querying inverted index"""
    def __init__(self, data: dict):
        """Initialize inverted index with 'data' dict"""
        self.data = data

    def query(self, words: list) -> list:
        """Search index for doc_ids which contains input words

        :param words: list of words to search
        :return: list of relevant documents for the given query
        """
        assert isinstance(words, list), (
            "query should be provided with a list of words, but user provided: "
            f"{repr(words)}"
        )
        print(f"query inverted index with request {repr(words)}", file=sys.stderr)
        res = []
        for word in words:
            doc_ids = set(self.data.get(word, []))
            res.append(doc_ids)
        if len(res) == 0:
            return []
        if len(res) == 1:
            return list(res[0])
        return list(res[0].intersection(*res[1:]))

    def dump(self, filepath: str):
        """Pack & write inverted index to hard drive

        :param filepath: desired path to dump index to
        :return: None
        """
        print("dumping inverted index to the ", filepath, file=sys.stderr)
        with open(filepath, 'wb') as f:
            # pack & write number of elements in dict
            count = len(self.data)
            d = struct.pack('>i', count)
            f.write(d)
            for word in self.data:
                # pack & write len of word
                len_encoded = len(word.encode())
                d = struct.pack('>H', len_encoded)
                f.write(d)
                # pack & write encoded word
                d = struct.pack('>' + str(len_encoded) + 's', word.encode())
                f.write(d)
                # pack & write count of elements in set
                count = len(self.data[word])
                d = struct.pack('>H', count)
                f.write(d)
                # pack & write every doc_id in set
                for doc_id in self.data[word]:
                    d = struct.pack('>H', doc_id)
                    f.write(d)

    @classmethod
    def load(cls, filepath: str) -> 'InvertedIndex':
        """Load & unpack inverted index from the disk

        :param filepath: path to dumped index
        :return: InvertedIndex() class object
        """
        print("loading inverted index from the ", filepath, file=sys.stderr)
        data = defaultdict(set)
        with open(filepath, 'rb') as f:
            # first 4 bytes - size of dict index
            d = f.read(4)
            count = struct.unpack('>i', d)[0]
            i = 0
            while i < count:
                # read & unpack len of word
                d_len = f.read(2)
                word_len = struct.unpack('>H', d_len)[0]
                # read, unpack & decode word
                d_word = f.read(word_len)
                word = struct.unpack('>' + str(word_len) + 's', d_word)[0].decode()
                # read & unpack count of elements in set
                d_set = f.read(2)
                count_set = struct.unpack('>H', d_set)[0]
                # unpack & add to set every doc_id
                j = 0
                while j < count_set:
                    d = f.read(2)
                    doc_id = struct.unpack('>H', d)[0]
                    # add to class variable
                    data[word].add(doc_id)
                    j += 1
                i += 1
        return cls(data)

    def __eq__(self, other):
        return self.data == other.data


def load_documents(filepath: str) -> dict:
    """Read documents from the disk and convert them to dict

    :param filepath: path to wikipedia documents
    :return: dict with a structure: {doc_id: "doc content"}
    """
    print(f"loading documents from path {filepath} to build inverted index...", file=sys.stderr)
    doc_dict = {}
    for line in open(filepath):
        doc = line.split('\t')
        if len(doc) > 1:
            content = " ".join(doc[1:])
            doc_dict[int(doc[0])] = content.strip()
    return doc_dict


def build_inverted_index(documents: dict) -> InvertedIndex:
    """Make inverted index out of documents dict

    :param documents: dict with a structure: {doc_id: "doc content"}
    :return: InvertedIndex() class object with data dict with a structure: {"word": set(doc_ids)}
    """
    print("building inverted index for provided documents...", file=sys.stderr)
    data = defaultdict(set)
    for doc_id, text in documents.items():
        words = text.split()
        for word in words:
            data[word].add(doc_id)
    return InvertedIndex(data)


def callback_build(arguments):
    """Callback for build"""
    print(f"call build subcommand with arguments: {arguments}", file=sys.stderr)
    documents = load_documents(arguments.dataset_filepath)
    inverted_index = build_inverted_index(documents)
    inverted_index.dump(arguments.index_filepath)


def callback_query(arguments):
    """Callback for query"""
    print(f"call query subcommand with arguments: {arguments}", file=sys.stderr)
    if arguments.query is not None:
        for query in arguments.query:
            process_arguments_query(arguments.index_filepath, query)
    else:  # arguments.query_file is not None
        for query in arguments.query_file:
            query = query.strip().split()
            process_arguments_query(arguments.index_filepath, query)


def process_arguments_query(index_filepath: str, query: list):
    """Support function for query callback"""
    inverted_index = InvertedIndex.load(index_filepath)
    document_ids = inverted_index.query(query)
    print(",".join(list(map(str, document_ids))))


def setup_parser(parser: ArgumentParser):
    """Parser configs"""
    subparsers = parser.add_subparsers(help="choose command")
    # build subcommand:
    build_parser = subparsers.add_parser(
        "build",
        help="build inverted index and save it in binary format to hard drive",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    build_parser.add_argument(
        "-d", "--dataset", default=DEFAULT_DATASET_PATH,
        dest="dataset_filepath",
        help="path to dataset to load"  # , default path is %(default)s
    )
    build_parser.add_argument(
        "-o", "--output", default=DEFAULT_INVERTED_INDEX_STORE_PATH,
        dest="index_filepath",
        help="path to store inverted index in a binary format"
    )
    build_parser.set_defaults(callback=callback_build)
    # query subcommand:
    query_parser = subparsers.add_parser(
        "query",
        help="query inverted index",
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    query_parser.add_argument(
        "-i", "--index", default=DEFAULT_INVERTED_INDEX_STORE_PATH,
        dest="index_filepath",
        help="path to read inverted index in a binary format"
    )
    query_file_group = query_parser.add_mutually_exclusive_group(required=False)  # required=True
    query_file_group.add_argument(
        "--query-file-utf8", dest="query_file", type=EncodedFileType("r", encoding="utf-8"),
        default=TextIOWrapper(sys.stdin.buffer, encoding="utf-8"),
        help="query file to get queries for inverted index"
    )
    query_file_group.add_argument(
        "--query-file-cp1251", dest="query_file", type=EncodedFileType("r", encoding="cp1251"),
        default=TextIOWrapper(sys.stdin.buffer, encoding="cp1251"),
        help="query file to get queries for inverted index"
    )
    query_file_group.add_argument(
        "-q", "--query",
        action="append", nargs="+", metavar="WORD",
        help="query to run against inverted index"
    )
    query_parser.set_defaults(callback=callback_query)


def main():
    parser = ArgumentParser(
        prog='inverted-index',
        description="tool to build, dump, load and query inverted index",
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    print(arguments, file=sys.stderr)
    arguments.callback(arguments)


if __name__ == "__main__":
    main()
